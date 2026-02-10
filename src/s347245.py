import networkx as nx
import numpy as np
from Problem import Problem
import random
import copy
import time
import os

# --- HYPERPARAMETERS ---
# Tuned for a balance between exploration (mutation) and exploitation (elitism).
POPULATION_SIZE = 50       # Number of individuals in the population
GENERATIONS = 80           # Number of evolution cycles
TOURNAMENT_SIZE = 5        # Size of the tournament for parent selection
MUTATION_RATE = 0.3        # Probability of flipping a sub-sequence
ELITISM_SIZE = 2           # Number of best individuals kept unchanged

# --- CONFIGURATION ---
# If True, calculates the baseline (Star Topology) to guarantee improvement >= 0%.
# Serves as a "Safety Net".
CHECK_BASELINE = True 

# If True, performs a final slow verification using exact edge traversal via Dijkstra.
# Necessary to get the exact score for the exam/report.
EXACT_VERIFICATION = True 

def solution(p: Problem, report_path: str = "report_s347245.txt"):
    """
    Entry point for the Computational Intelligence Project - Student s347245.
    
    Architectural Overview:
    1.  **Data Preparation**: Caches the graph and pre-computes distance matrices using 
        fast C-optimized algorithms (Floyd-Warshall via NumPy).
    2.  **Baseline Initialization**: Calculates the "Star Topology" cost.
    3.  **Optimization Phase (GA)**: Runs a Genetic Algorithm (Atomic or Split).
    4.  **Refinement Phase (Local Search)**: Applies 2-Opt local search.
    5.  **Verification & Reporting**: Calculates exact cost and generates a report.

    Args:
        p (Problem): The problem instance.
        report_path (str): The file path where the report will be saved.

    Returns:
        list: The optimal list of cities representing the path.
    """
    start_time = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] --- STARTING SOLVER ---")
    print(f"[{time.strftime('%H:%M:%S')}] Config: Size={len(p.graph.nodes)-1}, Alpha={p.alpha}, Beta={p.beta}, Density=Unknown")
    
    # 1. Graph Caching (Crucial Optimization)
    G = p.graph

    # 2. Matrix Pre-computation
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 0: Matrix Pre-computation (Floyd-Warshall)...")
    global_dist_matrix = nx.floyd_warshall_numpy(G, weight='dist')
    
    # Pre-calculate resistance upper bound (dist^beta)
    global_res_matrix = np.power(global_dist_matrix, p.beta)
    
    best_path = None
    best_cost_matrix = float('inf')

    # 3. Baseline Calculation (Star Topology)
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 1: Baseline Calculation (Star Topology)...")
    path_star = generate_star_path(G)
    cost_star_ub = calculate_path_cost_fast(p, path_star, global_dist_matrix, global_res_matrix)
    
    if CHECK_BASELINE:
        best_path = path_star
        best_cost_matrix = cost_star_ub
        print(f" -> Baseline Cost (Upper Bound): {cost_star_ub:,.2f}")
    
    # 4. Atomic Strategy (Standard GA)
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 2: Running Atomic GA...")
    path_atomic = run_ga_solver(p, G, global_dist_matrix, global_res_matrix, split_factor=0.0, label="Atomic")
    cost_atomic_ub = calculate_path_cost_fast(p, path_atomic, global_dist_matrix, global_res_matrix)
    
    # Compare with baseline
    if cost_atomic_ub < best_cost_matrix:
        best_path = path_atomic
        best_cost_matrix = cost_atomic_ub
        print(f" -> Improvement found (Atomic): {cost_atomic_ub:,.2f}")
    
    # 5. Split Strategy (Virtual Nodes)
    # Active only if Beta is high (>= 1.5).
    if p.beta >= 1.5:
        print(f"[{time.strftime('%H:%M:%S')}] PHASE 3: High Beta ({p.beta}) detected. Running Split GA...")
        path_split = run_ga_solver(p, G, global_dist_matrix, global_res_matrix, split_factor=0.25, label="Split")
        cost_split_ub = calculate_path_cost_fast(p, path_split, global_dist_matrix, global_res_matrix)
        
        if cost_split_ub < best_cost_matrix:
            best_path = path_split
            best_cost_matrix = cost_split_ub
            print(f" -> Improvement found (Split): {cost_split_ub:,.2f}")
    else:
        print(f"[{time.strftime('%H:%M:%S')}] PHASE 3: Skipped (Beta < 1.5)")
            
    # Fallback
    if best_path is None:
        best_path = path_atomic

    # 6. Final Polish (2-Opt Local Search)
    print(f"[{time.strftime('%H:%M:%S')}] PHASE 4: 2-Opt Refinement...")
    refined_path = refine_path_with_2opt(p, best_path, global_dist_matrix, global_res_matrix)
    
    # 7. Final Verification
    final_real_cost = 0
    if EXACT_VERIFICATION:
        print(f"[{time.strftime('%H:%M:%S')}] PHASE 5: Exact Cost Verification (Slow)...")
        final_real_cost = calculate_path_real_cost_exact_with_progress(p, G, refined_path)
        print(f" -> Official REAL Cost: {final_real_cost:,.2f}")
    else:
        final_real_cost = calculate_path_cost_fast(p, refined_path, global_dist_matrix, global_res_matrix)

    total_time = time.time() - start_time
    print(f"[{time.strftime('%H:%M:%S')}] --- FINISHED --- Total Time: {total_time:.2f}s")
    print(f"[{time.strftime('%H:%M:%S')}] Saving report to '{report_path}'...")
    
    write_report_to_file(p, G, refined_path, final_real_cost, total_time, report_path)
    print(f"[{time.strftime('%H:%M:%S')}] Report saved.\n")
    
    return refined_path

# --- REPORTING FUNCTIONS ---

def write_report_to_file(p, G, path, my_cost, total_time, filename):
    """
    Generates a detailed text file comparing solution against the Professor's baseline.
    
    Args:
        filename (str): Path to the output file.
    """
    # Calculate Professor's Baseline (Exact calculation)
    # We create a new Problem instance to ensure clean baseline calculation if needed,
    # but using 'p' and 'G' is more efficient.
    prof_path = generate_star_path(G)
    prof_cost = calculate_path_real_cost_exact_with_progress(p, G, prof_path, silent=True)
    
    improvement = prof_cost - my_cost
    improvement_pct = (improvement / prof_cost) * 100 if prof_cost > 0 else 0
    
    # Analyze gold collection
    city_stats = {} 
    for n in G.nodes():
        if n == 0: continue
        city_stats[n] = {'available': G.nodes[n].get('gold', 0), 'collected': 0.0, 'visits': 0}
        
    for node, gold in path:
        if node != 0:
            city_stats[node]['collected'] += gold
            city_stats[node]['visits'] += 1

    # Ensure directory exists
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w", encoding='utf-8') as f:
        f.write("=================================================================\n")
        f.write(f"           COMPUTATIONAL INTELLIGENCE PROJECT REPORT             \n")
        f.write("=================================================================\n\n")
        
        f.write(f"GENERAL DATA:\n")
        f.write(f" - Student ID: s347245\n")
        f.write(f" - Execution Time: {total_time:.2f} seconds\n")
        f.write(f" - Total Cities: {len(G.nodes) - 1}\n")
        f.write(f" - Parameters: Alpha={p.alpha}, Beta={p.beta}\n\n")
        
        f.write("PERFORMANCE COMPARISON:\n")
        f.write(f" - Baseline Cost (Prof):  {prof_cost:,.2f}\n")
        f.write(f" - Algorithm Cost (My):   {my_cost:,.2f}\n")
        f.write(f" - Absolute Improvement:  {improvement:,.2f}\n")
        if improvement > 0:
             f.write(f" - OUTCOME: VICTORY (+{improvement_pct:.2f}%)\n")
        else:
             f.write(f" - OUTCOME: DEFEAT ({improvement_pct:.2f}%)\n")
        f.write("\n" + "-"*65 + "\n\n")
        
        f.write("CITY COVERAGE DETAIL:\n")
        f.write(f"{'City ID':<10} | {'Available':<12} | {'Collected':<12} | {'% Taken':<8} | {'Visits':<6}\n")
        f.write("-" * 65 + "\n")
        
        total_gold_map = 0
        total_gold_taken = 0
        
        for cid in sorted(city_stats.keys()):
            s = city_stats[cid]
            pct = (s['collected'] / s['available'] * 100) if s['available'] > 0 else 0
            f.write(f"{cid:<10} | {s['available']:<12.1f} | {s['collected']:<12.1f} | {pct:<7.1f}% | {s['visits']:<6}\n")
            total_gold_map += s['available']
            total_gold_taken += s['collected']
            
        f.write("-" * 65 + "\n")
        tot_pct = (total_gold_taken / total_gold_map * 100) if total_gold_map > 0 else 0
        f.write(f"{'TOTAL':<10} | {total_gold_map:<12.1f} | {total_gold_taken:<12.1f} | {tot_pct:<7.1f}% | -\n\n")
        
        f.write("TRIP LOG:\n")
        current_trip = []
        trip_count = 1
        
        for node, gold in path:
            if node == 0:
                if current_trip:
                    trip_gold = sum(g for _, g in current_trip)
                    path_str = " -> ".join([str(n) for n, _ in current_trip])
                    f.write(f" Trip #{trip_count:<3} | Load: {trip_gold:<8.1f} | Path: 0 -> {path_str} -> 0\n")
                    current_trip = []
                    trip_count += 1
            else:
                current_trip.append((node, gold))

# --- HELPER FUNCTIONS ---

def generate_star_path(graph_obj):
    """Generates a Star Topology path: Depot -> City 1 -> Depot -> City 2 -> Depot..."""
    path = []
    nodes = list(range(1, len(graph_obj.nodes)))
    for node in nodes:
        gold = graph_obj.nodes[node].get('gold', 0)
        path.append((node, gold))
        path.append((0, 0))
    return path

def calculate_path_cost_fast(p, path, dist_matrix, res_matrix):
    """Calculates the 'Fictitious' Cost (Upper Bound) using O(1) Matrix Lookups."""
    cost = 0
    current_w = 0
    prev_node = 0
    alpha_pow_beta = p.alpha ** p.beta
    
    for node, gold in path:
        d = dist_matrix[prev_node][node]
        r = res_matrix[prev_node][node]
        w_term = (current_w ** p.beta) * alpha_pow_beta
        cost += d + w_term * r
        if node == 0: current_w = 0
        else: current_w += gold
        prev_node = node
    return cost

def calculate_path_real_cost_exact_with_progress(p, graph_obj, path, silent=False):
    """Calculates the EXACT physical cost required by the professor."""
    cost = 0
    current_w = 0
    prev_node = 0
    total_steps = len(path)
    log_step = max(1, total_steps // 5)
    
    for i, (node, gold) in enumerate(path):
        if not silent and i % log_step == 0 and i > 0:
            print(f"    ...verifying progress: {i}/{total_steps} steps...")
            
        real_path_nodes = nx.shortest_path(graph_obj, prev_node, node, weight='dist')
        for j in range(len(real_path_nodes) - 1):
            u = real_path_nodes[j]
            v = real_path_nodes[j+1]
            d = graph_obj[u][v]['dist']
            cost += d + (p.alpha * d * current_w) ** p.beta
            
        if node == 0: current_w = 0
        else: current_w += gold
        prev_node = node
    return cost

def refine_path_with_2opt(p, path, dist_matrix, res_matrix):
    """Applies 2-Opt local search on each sub-trip (Depot -> ... -> Depot)."""
    final_path_structure = []
    current_trip = []
    for node_info in path:
        node, gold = node_info
        if node == 0:
            if current_trip:
                optimized_trip = two_opt_on_sequence(p, current_trip, dist_matrix, res_matrix)
                final_path_structure.extend(optimized_trip)
                current_trip = []
            final_path_structure.append((0, 0))
        else:
            current_trip.append(node_info)
    return final_path_structure

def two_opt_on_sequence(p, sequence, dist_matrix, res_matrix):
    """Performs the 2-Opt swapping mechanism."""
    best_seq = sequence
    alpha_pow_beta = p.alpha ** p.beta
    
    def fast_seq_cost(seq):
        c = 0
        w = 0
        prev = 0
        for n, g in seq:
            d = dist_matrix[prev][n]
            r = res_matrix[prev][n]
            w_term = (w ** p.beta) * alpha_pow_beta
            c += d + w_term * r
            w += g
            prev = n
        d = dist_matrix[prev][0]
        r = res_matrix[prev][0]
        w_term = (w ** p.beta) * alpha_pow_beta
        c += d + w_term * r
        return c

    best_cost = fast_seq_cost(best_seq)
    improved = True
    max_iter = 50 
    iter_count = 0
    
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        n = len(best_seq)
        for i in range(n - 1):
            for j in range(i + 1, n):
                new_seq = best_seq[:i] + best_seq[i:j+1][::-1] + best_seq[j+1:]
                new_cost = fast_seq_cost(new_seq)
                if new_cost < best_cost:
                    best_seq = new_seq
                    best_cost = new_cost
                    improved = True
                    break 
            if improved: break
    return best_seq

def run_ga_solver(p: Problem, graph_obj, base_dist, base_res, split_factor=0.0, label="GA"):
    """Main Genetic Algorithm Engine."""
    original_num_cities = len(graph_obj.nodes)
    alpha = p.alpha
    beta = p.beta
    
    virtual_map = {} 
    original_gold = {}
    for n in range(original_num_cities):
        original_gold[n] = graph_obj.nodes[n].get('gold', 0)
        
    sorted_by_gold = sorted([(n, g) for n, g in original_gold.items() if n != 0], key=lambda x: x[1], reverse=True)
    num_to_split = int((original_num_cities - 1) * split_factor)
    nodes_to_split = set([n for n, _ in sorted_by_gold[:num_to_split]])
    
    expanded_gold = [] 
    expanded_to_original_idx = [] 
    current_id = 0
    
    for n in range(original_num_cities):
        g = original_gold[n]
        if n in nodes_to_split: g /= 2.0
        virtual_map[current_id] = n
        expanded_to_original_idx.append(n)
        expanded_gold.append(g)
        current_id += 1
    for n in range(original_num_cities):
        if n in nodes_to_split:
            g = original_gold[n] / 2.0 
            virtual_map[current_id] = n
            expanded_to_original_idx.append(n)
            expanded_gold.append(g)
            current_id += 1
            
    num_total_nodes = len(expanded_gold)
    gold_amounts = np.array(expanded_gold)
    indices = expanded_to_original_idx
    dist_matrix = base_dist[np.ix_(indices, indices)]
    res_matrix = base_res[np.ix_(indices, indices)]
    alpha_pow_beta = alpha ** beta

    def get_trip_chain_cost(sequence):
        cost = 0
        current_w = 0
        prev = 0 
        for node in sequence:
            d = dist_matrix[prev][node]
            w_term = (current_w ** beta) * alpha_pow_beta
            r = res_matrix[prev][node]
            cost += d + w_term * r
            current_w += gold_amounts[node]
            prev = node
        d = dist_matrix[prev][0]
        w_term = (current_w ** beta) * alpha_pow_beta
        r = res_matrix[prev][0]
        cost += d + w_term * r
        return cost

    def evaluate_genome(genome):
        final_path = []
        current_trip = []
        total_cost = 0
        for city in genome:
            test_trip = current_trip + [city]
            cost_merged = get_trip_chain_cost(test_trip)
            if not current_trip:
                current_trip.append(city)
                continue
            cost_current_closed = get_trip_chain_cost(current_trip)
            cost_new_single = get_trip_chain_cost([city])
            if cost_merged <= (cost_current_closed + cost_new_single):
                current_trip.append(city)
            else:
                total_cost += cost_current_closed
                for node in current_trip:
                    final_path.append((virtual_map[node], gold_amounts[node]))
                final_path.append((0, 0))
                current_trip = [city]
        if current_trip:
            total_cost += get_trip_chain_cost(current_trip)
            for node in current_trip:
                final_path.append((virtual_map[node], gold_amounts[node]))
            final_path.append((0, 0))
        return total_cost, final_path

    def generate_greedy_seed():
        unvisited = list(range(1, num_total_nodes))
        tour = []
        curr = 0
        while unvisited:
            nxt = min(unvisited, key=lambda x: dist_matrix[curr][x])
            tour.append(nxt)
            unvisited.remove(nxt)
            curr = nxt
        return tour

    def crossover_ox1(p1, p2):
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b+1] = p1[a:b+1]
        ptr = (b + 1) % size
        for gene in p2[b+1:] + p2[:b+1]:
            if gene not in child:
                child[ptr] = gene
                ptr = (ptr + 1) % size
        return child

    def mutate_inversion(genome):
        if random.random() < MUTATION_RATE:
            size = len(genome)
            a, b = sorted(random.sample(range(size), 2))
            genome[a:b+1] = genome[a:b+1][::-1]
        return genome

    cities_indices = list(range(1, num_total_nodes))
    population = []
    population.append(generate_greedy_seed())
    for _ in range(POPULATION_SIZE - 1):
        ind = cities_indices[:]
        random.shuffle(ind)
        population.append(ind)

    scored_pop = []
    for genome in population:
        cost, _ = evaluate_genome(genome)
        scored_pop.append((cost, genome))
    
    scored_pop.sort(key=lambda x: x[0])
    best_fitness = scored_pop[0][0]
    best_genome = list(scored_pop[0][1])

    start_ga = time.time()
    for gen in range(GENERATIONS):
        if gen % 10 == 0 or gen == GENERATIONS - 1:
            elapsed = time.time() - start_ga
            avg_time_per_gen = elapsed / (gen + 1) if gen > 0 else 0
            remaining = avg_time_per_gen * (GENERATIONS - gen)
            print(f"   [{label}] Gen {gen+1}/{GENERATIONS} | Best (UB): {best_fitness:,.2f} | ETA: {remaining:.1f}s")
        new_pop = [g for _, g in scored_pop[:ELITISM_SIZE]]
        while len(new_pop) < POPULATION_SIZE:
            candidates = random.sample(scored_pop, TOURNAMENT_SIZE)
            p1 = min(candidates, key=lambda x: x[0])[1]
            candidates = random.sample(scored_pop, TOURNAMENT_SIZE)
            p2 = min(candidates, key=lambda x: x[0])[1]
            new_pop.append(mutate_inversion(crossover_ox1(p1, p2)))
        population = new_pop
        scored_pop = []
        for genome in population:
            cost, _ = evaluate_genome(genome)
            scored_pop.append((cost, genome))
            if cost < best_fitness:
                best_fitness = cost
                best_genome = list(genome)
        scored_pop.sort(key=lambda x: x[0])
    _, final_path = evaluate_genome(best_genome)
    return final_path