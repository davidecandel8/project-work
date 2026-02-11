import networkx as nx
import numpy as np
from Problem import Problem
import random
import time
import os
import sys

# --- HYPERPARAMETERS ---
# Tuned for a balance between exploration and exploitation.
POPULATION_SIZE = 50       # Number of individuals in the population
GENERATIONS = 80           # Number of evolution cycles
TOURNAMENT_SIZE = 5        # Size of the tournament for parent selection
MUTATION_RATE = 0.3        # Probability of flipping a sub-sequence (Inversion Mutation)
ELITISM_SIZE = 2           # Number of best individuals kept unchanged across generations

# If True, performs a final verification using exact Dijkstra traversal (slower but precise).
EXACT_VERIFICATION = True 

def solution(p: Problem):
    """
    Main entry point for the project solution.
    
    Architectural Overview:
    1.  **Validation**: Checks input parameters for sanity (None check, bounds check).
    2.  **Data Preparation**: Caches the graph and pre-computes distance matrices using 
        fast C-optimized algorithms (Floyd-Warshall via NumPy).
    3.  **Baseline Initialization**: Calculates the "Star Topology" cost to ensure we have a valid fallback.
    4.  **Optimization Phase (GA)**: Runs a Genetic Algorithm.
        - If Beta is low (< 1.5): Runs standard Atomic GA.
        - If Beta is high (>= 1.5): Runs Split GA (virtual nodes) to optimize load management.
    5.  **Refinement Phase (Local Search)**: Applies 2-Opt local search to untangle paths.
    6.  **Verification & Reporting**: Calculates the exact physical cost, prints summary to terminal,
        and saves a detailed report to 'reports/' folder.

    Args:
        p (Problem): The problem instance provided by the evaluation environment.

    Returns:
        list[tuple]: The optimal path sequence format: [(node_id, gold_collected), ..., (0, 0)]
                     Returns None if input is invalid.
    """
    
    # --- 0. INPUT VALIDATION ---
    # Handle None input gracefully
    if p is None:
        print("[ERROR] Input problem is None.")
        return None

    # Sanity check on parameters
    try:
        # --- 1. SETUP & CACHING ---
        # Accessing p.graph creates a deep copy every time. We cache it once here.
        G = p.graph 
        if G is None or len(G.nodes) == 0:
            print("[ERROR] Graph is empty or None.")
            return None
        
        if p.alpha < 0 or p.beta < 0:
            print(f"[ERROR] Invalid parameters: Alpha={p.alpha}, Beta={p.beta}. Must be >= 0.")
            return None
            
    except Exception as e:
        print(f"[ERROR] Exception during validation: {e}")
        return None
    
    # --- 2. DYNAMIC REPORTING SETUP ---
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    
    if num_nodes < 1:
        return []

    # --- METRICS CALCULATION ---
    # The given 'Problem.py' encapsulates the generation parameters (Input Hiding).
    # The initial 'density' probability used in the constructor is not exposed as a public property
    # Therefore, we must reverse-engineer the actual density of the generated graph 

    # Formula for Undirected Graph Density: D = (2 * E) / (V * (V - 1))
    # Where:
    #   E = Number of actual edges (num_edges)
    #   V = Number of nodes (num_nodes)
    #   V * (V - 1) / 2 = Maximum possible edges in a complete undirected graph.

    # Example:
    #   Consider a graph with V=4 nodes. Max edges = (4*3)/2 = 6.
    #   If we have E=3 edges actually connected:
    #   Density = (2 * 3) / (4 * 3) = 6 / 12 = 0.5 (50% connectivity).
    
    calculated_density = (2 * num_edges) / (num_nodes * (num_nodes - 1)) if num_nodes > 1 else 0
    
    # Create reports directory
    report_dir = "reports"
    os.makedirs(report_dir, exist_ok=True)
    
    # Helper to format numbers: remove trailing zeros if integer
    def fmt(n):
        return f"{n:g}"

    # Construct filename: report_N{TotalNodes}_D{.1f}_A{g}_B{g}.txt
    # Example: report_N100_D0.2_A1_B2.txt (Density forced to 1 decimal place)
    report_filename = f"report_N{num_nodes}_D{calculated_density:.1f}_A{fmt(p.alpha)}_B{fmt(p.beta)}.txt"
    report_path = os.path.join(report_dir, report_filename)

    # --- 3. MATRIX PRE-COMPUTATION ---
    # Instead of calculating path lengths using Dijkstra (which is slow for a lot of queries
    # we pre-calculate the shortest path between ALL pairs of nodes.

    # Why 'floyd_warshall_numpy'?
    # 1. Complexity Trade-off: It costs O(V^3) once at startup, but grants O(1) lookup time 
    #    during the Genetic Algorithm evolution. This is crucial for performance.
    # 2. Low-Level Optimization: Unlike standard NetworkX functions (pure Python), 
    #    this uses NumPy's C-optimized linear algebra backend. It is significantly faster 
    #    and memory-efficient for dense matrix operations.
    #    In src/test.py we tested Floyd-Warshall (NumPy) vs Dijkstra.
    # 3. Output: Returns a 2D dense matrix where M[i, j] is the shortest distance from node i to j.

    try:
        global_dist_matrix = nx.floyd_warshall_numpy(G, weight='dist')
    except Exception as e:
        print(f"[ERROR] Matrix computation failed: {e}")
        return None
    
    #The cost function includes a heavy power operation: (alpha * dist * weight)^beta.
    # We can mathematically decompose this into: (alpha^beta) * (weight^beta) * (dist^beta).
    # Since 'dist' (distance between nodes) and 'beta' are constant throughout the 
    # entire genetic algorithm evolution, we can pre-calculate the (dist^beta) term.
    # Instead of calling the expensive pow() function millions of times inside the 
    # fitness loop, we compute it ONCE here using vectorized NumPy operations.
    # During the evolution, the cost calculation becomes a simple multiplication (O(1)).
    global_res_matrix = np.power(global_dist_matrix, p.beta)
    
    best_path = None
    best_cost_ub = float('inf')

    # --- 4. BASELINE CALCULATION (SAFETY NET) ---

    # We generate a simple "Star Topology" path (Depot -> City -> Depot) for each city.
    # This rapresents our upper bound baseline valid solution.
    # It is a naive approach different from the professor's baseline, 
    # but it ensures we have a valid path to compare against in a time-efficient manner.
    path_star = generate_star_path(G)
    cost_star_ub = calculate_path_cost_fast(p, path_star, global_dist_matrix, global_res_matrix)
    
    # Actual best path and cost are initialized with the Star Topology results
    best_path = path_star
    best_cost_ub = cost_star_ub
    
    # --- 5. ATOMIC STRATEGY (STANDARD GA) ---
    path_atomic = run_ga_solver(p, G, global_dist_matrix, global_res_matrix, split_factor=0.0)
    cost_atomic_ub = calculate_path_cost_fast(p, path_atomic, global_dist_matrix, global_res_matrix)
    
    if cost_atomic_ub < best_cost_ub:
        best_path = path_atomic
        best_cost_ub = cost_atomic_ub
    
    # --- 6. SPLIT STRATEGY (VIRTUAL NODES GA) ---
    # Active only if Beta >= 1.5.
    if p.beta >= 1.5:
        path_split = run_ga_solver(p, G, global_dist_matrix, global_res_matrix, split_factor=0.25)
        cost_split_ub = calculate_path_cost_fast(p, path_split, global_dist_matrix, global_res_matrix)
        
        if cost_split_ub < best_cost_ub:
            best_path = path_split
            best_cost_ub = cost_split_ub
            
    if best_path is None:
        best_path = path_atomic

    # --- 7. LOCAL REFINEMENT (2-OPT) ---
    refined_path = refine_path_with_2opt(p, best_path, global_dist_matrix, global_res_matrix)
    
    # --- 8. FINAL VERIFICATION & FORMATTING ---
    if not refined_path or refined_path[-1] != (0, 0):
        refined_path.append((0, 0))
        
    final_real_cost = calculate_path_real_cost_exact(p, G, refined_path)
    
    # --- 9. AUTOMATIC REPORTING ---
    print_terminal_minimal_report(p, G, final_real_cost, report_path)
    write_report_to_file(p, G, refined_path, final_real_cost, report_path, calculated_density)
    
    return refined_path

# --- HELPER FUNCTIONS ---

def generate_star_path(graph_obj):
    """
    Generates a Star Topology path (Depot -> City -> Depot).
    Args: 
        graph_obj: The graph object containing nodes and edges.
    Returns:
        list[tuple]: A path in the format [(node_id, gold_collected), ..., (0, 0)]
    """
    path = []
    nodes = list(range(1, len(graph_obj.nodes)))
    for node in nodes:
        gold = graph_obj.nodes[node].get('gold', 0)
        path.append((node, gold))
        path.append((0, 0))
    return path

def calculate_path_cost_fast(p, path, dist_matrix, res_matrix):
    """
    Calculates the cost Upper Bound using pre-computed matrices (O(1) lookup).
    
    This function represents the 'Fitness Evaluation' step, which is the performance 
    bottleneck of the Genetic Algorithm. To maximize speed, it avoids expensive 
    graph traversals (Dijkstra) and reduces heavy math operations by leveraging 
    pre-calculated matrices.

    Args:
        p (Problem): The problem instance containing global constants (alpha, beta).
        path (list[tuple]): The candidate solution (genome). A list of tuples (node_id, gold_amount).
        dist_matrix (np.ndarray): A 2D dense matrix where [i][j] contains the shortest 
                                  distance (Floyd-Warshall) between node i and j.
        res_matrix (np.ndarray): A 2D dense matrix where [i][j] contains the pre-computed 
                                 geometric factor: (distance[i][j] ** beta).

    Returns:
        float: The scalar total cost (fitness) of the provided path.
    """
    cost = 0
    current_w = 0
    prev_node = 0
    
    # Pre-calculate the alpha component of the formula.
    # Since alpha and beta are constant for the problem instance, 
    # we compute (alpha^beta) once here instead of inside the loop.
    alpha_pow_beta = p.alpha ** p.beta
    
    for node, gold in path:
        d = dist_matrix[prev_node][node] # O(1) lookup from Floyd-Warshall matrix
        r = res_matrix[prev_node][node] # O(1) lookup from pre-computed geometric matrix
        w_term = (current_w ** p.beta) * alpha_pow_beta
        cost += d + w_term * r
        
        if node == 0: 
            current_w = 0
        else: 
            current_w += gold
        prev_node = node
    return cost

def calculate_path_real_cost_exact(p, graph_obj, path):
    """
    Calculates the EXACT physical cost by reconstructing the path edge-by-edge.
    """
    cost = 0
    current_w = 0
    prev_node = 0
    
    for node, gold in path:
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
    """
    Applies 2-Opt Local Search to untangle crossing paths.
    """
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
    """
    Core 2-Opt mechanism: tries to reverse segments to minimize cost.
    """
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

def run_ga_solver(p: Problem, graph_obj, base_dist, base_res, split_factor=0.0):
    """
    Runs the Genetic Algorithm.
    """
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

    # --- Evolution Loop ---
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

    for gen in range(GENERATIONS):
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

def write_report_to_file(p, G, path, my_cost, filename, density):
    """
    Generates the detailed text report required for the assignment.
    Includes Performance Comparison, City Coverage, and Trip Log.
    """
    # 1. Calculate Baseline for comparison using the OFFICIAL method
    prof_cost = p.baseline()
    
    improvement = prof_cost - my_cost
    improvement_pct = (improvement / prof_cost * 100) if prof_cost > 0 else 0
    outcome = "IMPROVEMENT" if improvement > 0 else "NO IMPROVEMENT"
    
    with open(filename, "w", encoding='utf-8') as f:
        # Header
        f.write("COMPUTATIONAL INTELLIGENCE PROJECT REPORT\n")
        f.write("=========================================\n")
        f.write(f"Student: s347245\n")
        f.write(f"Parameters: Alpha={p.alpha:.2f}, Beta={p.beta:.2f}, Density={density:.2f}, Cities={len(G.nodes)}\n")
        f.write("=========================================\n\n")
        
        # PERFORMANCE COMPARISON
        f.write("PERFORMANCE COMPARISON\n")
        f.write("-" * 25 + "\n")
        f.write(f" - Baseline Cost:       {prof_cost:,.1f}\n")
        f.write(f" - Algorithm Cost:      {my_cost:,.1f}\n")
        f.write(f" - Absolute Improvement: {improvement:,.1f}\n")
        f.write(f" - % Improvement:       {improvement_pct:.2f}%\n")
        f.write(f" - Result:              {outcome}\n\n")
        
        # CITY COVERAGE DETAIL
        f.write("CITY COVERAGE DETAIL:\n")
        f.write(f"{'City ID':<10} | {'Available':<12} | {'Visits':<8}\n")
        f.write("-" * 35 + "\n")
        
        city_stats = {}
        for n in G.nodes():
            if n == 0: continue
            city_stats[n] = {'available': G.nodes[n].get('gold', 0), 'visits': 0}

        for node, _ in path:
            if node != 0:
                city_stats[node]['visits'] += 1
                
        for cid in sorted(city_stats.keys()):
            avail = city_stats[cid]['available']
            visits = city_stats[cid]['visits']
            f.write(f"{cid:<10} | {avail:<12.1f} | {visits:<8}\n")
        f.write("\n")

        # TRIP LOG
        f.write("TRIP LOG:\n")
        f.write(f"{'Trip #':<8} | {'Load':<12} | {'Path'}\n")
        f.write("-" * 60 + "\n")
        
        current_trip = []
        trip_count = 1
        
        for node, gold in path:
            if node == 0:
                if current_trip:
                    trip_load = sum(g for _, g in current_trip)
                    path_str = "0 -> " + " -> ".join([str(n) for n, _ in current_trip]) + " -> 0"
                    f.write(f"Trip #{trip_count:<3} | Load: {trip_load:<6.1f} | Path: {path_str}\n")
                    current_trip = []
                    trip_count += 1
            else:
                current_trip.append((node, gold))

def print_terminal_minimal_report(p, G, my_cost, report_path):
    """
    Prints a minimalist confirmation to terminal.
    Shows only that execution finished and the final score.
    """
    # Use official baseline for the terminal output too, for consistency.
    prof_cost = p.baseline()
    improvement_pct = ((prof_cost - my_cost) / prof_cost * 100) if prof_cost > 0 else 0
    
    print("\n" + "="*60)
    print(f" [s347245] OPTIMIZATION COMPLETE")
    print("-" * 60)
    print(f" Final Cost:      {my_cost:,.1f}")
    print(f" Improvement:     {improvement_pct:.2f}% vs Baseline")
    print(f" Full Report:     {report_path}")
    print("="*60 + "\n")

# --- MAIN BLOCK (PROTECTED) ---
if __name__ == "__main__":
    # This block executes ONLY if you run this file directly.
    # It is ignored if imported by the professor's script (import s347245).
    print("Running Test Suite for s347245...")
    
    # Comprehensive Test Cases (N, Density, Alpha, Beta)
    test_cases = [
        (100, 0.2, 1, 1),
        (100, 0.2, 2, 1),
        (100, 0.2, 1, 2),
        (100, 1, 1, 1),
        (100, 1, 2, 1),
        (100, 1, 1, 2),
        (1000, 0.2, 1, 1),
        (1000, 0.2, 2, 1),
        (1000, 0.2, 1, 2),
        (1000, 1, 1, 1),
        (1000, 1, 2, 1),
        (1000, 1, 1, 2)
    ]
    
    for config in test_cases:
        n, d, a, b = config
        print(f"\n>>> Running Test: Cities={n}, Density={d}, Alpha={a}, Beta={b}")
        p = Problem(n, density=d, alpha=a, beta=b, seed=42)
        solution(p)
    
    print("\nAll tests completed.")