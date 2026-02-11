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
    5.  **Refinement Phase (Local Search)**: Applies Swap local search to optimize load placement.
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
        # MATRIX 1: LINEAR DISTANCE
        # Calculates the shortest path based on standard edge weights.
        # Used for the first term of the cost function (linear distance).
        global_dist_matrix = nx.floyd_warshall_numpy(G, weight='dist')

        # MATRIX 2: NON-LINEAR DISTANCE
        # The cost function includes a heavy power operation: (alpha * dist * weight)^beta.
        # We can mathematically decompose this into: (alpha^beta) * (weight^beta) * (dist^beta).
        # Since 'dist' (distance between nodes) and 'beta' are constant throughout the 
        # entire genetic algorithm evolution, we can pre-calculate the (dist^beta) term.
        # Instead of calling the expensive pow() function millions of times inside the 
        # fitness loop, we compute it ONCE here using vectorized NumPy operations.
        # During the evolution, the cost calculation becomes a simple multiplication (O(1)).
        # The cost function second term depends on (dist^beta).
        # Since (a + b)^beta != a^beta + b^beta for beta > 1, we cannot simply power the sum.
        # We must sum the powers.
        
        # We create a virtual graph where edge weights represent the non-linear cost contribution (d^beta).
        # This allows us to run Floyd-Warshall again to get the correct non-linear path costs.
        G_beta = G.copy()
        for u, v, data in G_beta.edges(data=True):
            data['dist_beta'] = data['dist'] ** p.beta
            
        # Running Floyd-Warshall on this graph gives us the path that minimizes sum(d^beta).
        # entry [i][j] now contains the correct sum of powered edges, solving the overestimation bug.
        global_res_matrix = nx.floyd_warshall_numpy(G_beta, weight='dist_beta')

    except Exception as e:
        print(f"[ERROR] Matrix computation failed: {e}")
        return None
    
    best_path = None
    best_cost_ub = float('inf')

    # --- 4. BASELINE CALCULATION ---

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
    # In high-beta scenarios, the non-linear cost penalty for carrying large loads is severe.
    # By creating virtual nodes for the heaviest cities, we allow the GA to optimize load management 
    # by treating partial pickups as separate entities.
    if p.beta >= 1.5:
        path_split = run_ga_solver(p, G, global_dist_matrix, global_res_matrix, split_factor=0.25)
        cost_split_ub = calculate_path_cost_fast(p, path_split, global_dist_matrix, global_res_matrix)
        
        if cost_split_ub < best_cost_ub:
            best_path = path_split
            best_cost_ub = cost_split_ub
            
    if best_path is None:
        best_path = path_atomic

    # --- 7. LOCAL REFINEMENT (SWAP SEARCH) ---
    # The GA provides us with a strong candidate solution, but it may contain suboptimal segments.
    # We apply a Swap local search to each individual trip (between depot returns) to untangle any crossing paths and reduce costs.
    refined_path = refine_path_with_local_search(p, best_path, global_dist_matrix, global_res_matrix)
    
    # --- 8. FINAL VERIFICATION & FORMATTING ---
    # If the refined path does not end with a return to the depot, we append it to ensure validity.
    if not refined_path or refined_path[-1] != (0, 0):
        refined_path.append((0, 0))
        
    # Calculate the exact physical cost of the refined path by reconstructing the actual edges traversed in the original graph.
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
    
    DIFFERENCE VS 'FAST' CALCULATION:
    1. Granularity:
       - The 'Fast' function treats the path between two cities as a single "macro-link" 
         and applies the cost formula once to the total distance.
       - This 'Exact' function uses Dijkstra to find the actual sequence of physical 
         road segments (edges) connecting the two cities.
         
    2. Mathematical Rigor (Non-Linearity):
       - Due to the non-linear exponent beta, (d1 + d2)^beta != d1^beta + d2^beta.
       - The 'Fast' function is an approximation (Upper Bound if beta > 1).
       - This function is the "Ground Truth": it sums the cost of every small edge individually.
       
    3. Performance:
       - This function is computationally expensive (calls nx.shortest_path repeatedly).
       - It should ONLY be used for final validation, NOT inside the Genetic Algorithm loop.

    Args:
        p (Problem): The problem instance containing global constants (alpha, beta).
        graph_obj: The original graph object from the problem instance.
        path (list[tuple]): The candidate solution (genome) to evaluate, in the format [(node_id, gold_collected), ..., (0, 0)].
    Returns:
        float: The exact total cost of the provided path.
    """
    cost = 0
    current_w = 0
    prev_node = 0
    
    for node, gold in path:
        # We need to find the actual road segments between 'prev_node' and 'node'
        real_path_nodes = nx.shortest_path(graph_obj, prev_node, node, weight='dist')
        # Iterate through every physical edge in the shortest path
        # Example: To go from A to B, we might pass through x -> y -> z.
        for j in range(len(real_path_nodes) - 1):
            u = real_path_nodes[j]
            v = real_path_nodes[j+1]
            d = graph_obj[u][v]['dist']
            cost += d + (p.alpha * d * current_w) ** p.beta
            
        if node == 0: current_w = 0
        else: current_w += gold
        prev_node = node
    return cost

def refine_path_with_local_search(p, path, dist_matrix, res_matrix):
    """
    Post-Processing Phase: Applies Swap Local Search to each sub-trip individually.
    The full path consists of multiple trips separated by the Depot (0).
    We cannot swap nodes between different trips without re-evaluating the global load constraints.
    Therefore, we isolate each trip (Depot -> Cities -> Depot) and optimize its internal geometry.

    Args: 
        p (Problem): The problem instance containing global constants (alpha, beta).
        path (list[tuple]): The candidate solution (genome) to refine.
        dist_matrix (np.ndarray): Pre-computed distance matrix for O(1) lookups.
        res_matrix (np.ndarray): Pre-computed matrix for O(1) lookups.
    Returns:
        list[tuple]: The refined path after applying Swap Local Search, in the format [(node_id, gold_collected), ..., (0, 0)]
    """
    final_path_structure = []
    current_trip = []
    
    for node_info in path:
        node, gold = node_info
        # Check if we hit a Depot return
        if node == 0:
            if current_trip:
                # Optimize the specific sequence of cities within this single trip.
                optimized_trip = swap_local_search(p, current_trip, dist_matrix, res_matrix)
                final_path_structure.extend(optimized_trip)
                current_trip = []
            # Explicitly add the depot return to the final structure
            final_path_structure.append((0, 0))
        else:
            # Build the current trip buffer
            current_trip.append(node_info)

    # If the input path did NOT end with (0,0), the last trip is still sitting in 'current_trip'.
    # We must process and flush it to avoid silently deleting valid cities from the solution.
    if current_trip:
        optimized_trip = swap_local_search(p, current_trip, dist_matrix, res_matrix)
        final_path_structure.extend(optimized_trip)
        final_path_structure.append((0, 0))

    return final_path_structure

def swap_local_search(p, sequence, dist_matrix, res_matrix):
    """
    Implements a Local Search based on the Swap operator to optimize a specific route segment.
    It iteratively explores the neighborhood of the current sequence by swapping pairs of nodes.
    Since the cost function depends on the accumulated weight, a complete recalculation of the path cost 
    is performed for each swap to ensure validity.

    Args:
        p (Problem): The problem instance containing global constants (alpha, beta).
        sequence (list[tuple]): The current sub-trip sequence of nodes [(node_id, gold), ...].
        dist_matrix (np.ndarray): Pre-computed distance matrix for O(1) lookups.
        res_matrix (np.ndarray): Pre-computed geometric matrix for O(1) lookups.

    Returns:
        list[tuple]: The optimized sequence of nodes after applying local search.
    """
    best_seq = sequence
    alpha_pow_beta = p.alpha ** p.beta
    
    def fast_seq_cost(seq):
        '''
        Calculates the cost of a given sequence of nodes using pre-computed matrices.
        This is a specialized version of the cost function that operates on a simple list of nodes (without gold amounts)
        and assumes the path starts and ends at the depot (0). It uses the pre-computed distance and geometric matrices for O(1) lookups.
        Args:
            seq (list[int]): A list of node indices representing the order of visits in the trip.   
        Returns:
            float: The total cost of the trip.
        '''
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
    max_iter = 50 # safety cap to prevent infinite loops in pathological cases
    iter_count = 0
    
    # keep optimizing as long as we find improvements and we haven't hit the iteration cap
    while improved and iter_count < max_iter:
        improved = False
        iter_count += 1
        n = len(best_seq)

        # Try all possible pairs of indices (i, j) to SWAP them
        for i in range(n - 1):
            for j in range(i + 1, n):
                # Construct new sequence using SWAP
                # Swap node at i with node at j
                new_seq = best_seq[:] # Create a copy
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]
                
                new_cost = fast_seq_cost(new_seq)

                if new_cost < best_cost:
                    best_seq = new_seq
                    best_cost = new_cost
                    improved = True
                    break # restart the search from the new configuration immediately
            if improved: break
    return best_seq

def run_ga_solver(p: Problem, graph_obj, base_dist, base_res, split_factor=0.0):
    """
    Runs the Genetic Algorithm. In this implementation, we have two modes:
    1. Atomic GA (split_factor=0.0): Standard GA where each city is represented as a single node.
    2. Split GA (split_factor > 0): We create virtual nodes for the top 'split_factor' percentage of cities 
       with the highest gold amounts. This allows the GA to optimize load management by treating 
       partial pickups as separate entities.
    Args:
        p (Problem): The problem instance containing global constants (alpha, beta).
        graph_obj: The original graph object from the problem instance.
        base_dist: The pre-computed distance matrix (Floyd-Warshall).
        base_res: The pre-computed geometric matrix (distance^beta).
        split_factor: A float between 0 and 1 indicating the percentage of cities to split into virtual nodes.

    Returns:
        list[tuple]: The best path found by the GA in the format [(node_id, gold_collected), ..., (0, 0)]
        """
    original_num_cities = len(graph_obj.nodes)
    alpha = p.alpha
    beta = p.beta
    
    virtual_map = {} 
    original_gold = {}
    
    # For each original city, we check if it is in the top 'split_factor' percentage of gold amounts.
    for n in range(original_num_cities):
        original_gold[n] = graph_obj.nodes[n].get('gold', 0)
        
    # We sort cities by gold amount without the city 0 (depot) in descending order. So the first cities in the sorted list are the richest ones.
    sorted_by_gold = sorted([(n, g) for n, g in original_gold.items() if n != 0], key=lambda x: x[1], reverse=True)

    # Determine 'K': the number of cities to split based on the 'split_factor' hyperparameter.
    # Example: If N=100 and split_factor=0.2, we select the top 20 heaviest cities.
    num_to_split = int((original_num_cities - 1) * split_factor)

    # Extract the Node IDs of the top 'K' heaviest cities.
    nodes_to_split = set([n for n, _ in sorted_by_gold[:num_to_split]])
    
    expanded_gold = [] 
    expanded_to_original_idx = [] 
    current_id = 0

    # Goal: Create an 'expanded' set of nodes where heavy cities are duplicated.
    # This allows the Genetic Algorithm to visit a heavy city twice (picking up half load each time),
    # significantly reducing the non-linear cost penalty (w^beta).

    # Pass 1: Base Layer Creation
    # Iterate through ALL original cities to create the primary instance of each node.
    
    for n in range(original_num_cities):
        g = original_gold[n] 
        # If the node is flagged for splitting, we only assign HALF of its gold 
        # to this primary instance. The other half will be assigned to the 'clone' later.
        if n in nodes_to_split: g /= 2.0
        virtual_map[current_id] = n
        expanded_to_original_idx.append(n)
        expanded_gold.append(g)
        current_id += 1
    
    # Iterate again to create the SECOND instance (the clone) for the split cities.
    for n in range(original_num_cities):
        if n in nodes_to_split:
            g = original_gold[n] / 2.0 
            virtual_map[current_id] = n
            expanded_to_original_idx.append(n)
            expanded_gold.append(g)
            current_id += 1
            
    num_total_nodes = len(expanded_gold) # update total number of nodes

    # gold_amounts[i] gives the gold amount for the virtual node i. For non-split cities, this is the original gold. 
    # For split cities, this is half of the original gold for both the primary and clone nodes.
    gold_amounts = np.array(expanded_gold) 
    
    indices = expanded_to_original_idx # update indices
    dist_matrix = base_dist[np.ix_(indices, indices)] # update distance matrix
    res_matrix = base_res[np.ix_(indices, indices)] # update matrix
    alpha_pow_beta = alpha ** beta

    def get_trip_chain_cost(sequence):
        '''
        Calculates the cost of a trip chain (a sequence of virtual nodes) using the pre-computed matrices.
        Args:
            sequence (list[int]): A list of virtual node indices representing the trip chain.
        Returns:
            float: The total cost of the trip chain.
        '''
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
        '''
        Evaluates the total cost of a given genome (a sequence of virtual nodes) by reconstructing the path and 
        calculating the cost using the pre-computed matrices. Implements a 'Greedy Split' strategy to minimize non-linear costs.

        Args:
            genome (list[int]): A list of virtual node indices representing the order of visits in the GA solution.
        Returns:
            tuple: (total_cost, final_path) where total_cost is the scalar cost of the genome and final_path is the 
                   reconstructed path in the format [(original_node_id, gold_collected), ..., (0, 0)].
        '''
        final_path = []
        current_trip = []
        total_cost = 0
        
        # Iterate through sequence of cities suggested by the Genetic Algorithm
        for city in genome:
            # Create a temporary trip extending the current one with the new city and calculate its cost.
            test_trip = current_trip + [city]
            cost_merged = get_trip_chain_cost(test_trip)
            
            # if current_trip is empty, we start a new trip with the current city, we must add it without comparison
            if not current_trip:
                current_trip.append(city)
                continue
            
            # Calculate cost of closing the current trip NOW (return to depot) and starting a new trip with the new city.
            cost_current_closed = get_trip_chain_cost(current_trip)
            cost_new_single = get_trip_chain_cost([city])
            
            if cost_merged <= (cost_current_closed + cost_new_single): # cheaper to continue
                current_trip.append(city)
            else:
                total_cost += cost_current_closed
                for node in current_trip: # cheaper to close current trip and start new one
                    final_path.append((virtual_map[node], gold_amounts[node]))
                final_path.append((0, 0)) # return to depot: -> (0, 0)
                current_trip = [city]

        # After processing all cities, if there is an open trip, we must close it by returning to the depot.    
        if current_trip:
            total_cost += get_trip_chain_cost(current_trip)
            for node in current_trip:
                final_path.append((virtual_map[node], gold_amounts[node]))
            final_path.append((0, 0))
            
        return total_cost, final_path

    def generate_greedy_genome():
        '''
        Generates a greedy solution for the GA by always visiting the nearest unvisited city next.
        This provides a strong starting point for the GA to evolve from, especially in low-beta scenarios 
        where distance is more critical.

        Returns:
            list[int]: A list of virtual node indices representing the greedy path.
        '''
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
        '''
        Implements Order Crossover (OX1) for TSP-like problems. It creates a child by taking a random segment 
        from parent 1 and filling the remaining genes in the order they appear in parent 2.
        
        Args:
            p1 (list[int]): Parent 1 genome (sequence of virtual node indices).
            p2 (list[int]): Parent 2 genome (sequence of virtual node indices).
        Returns:
            list[int]: The child genome resulting from the crossover.
        '''
        size = len(p1)
        a, b = sorted(random.sample(range(size), 2))
        child = [-1] * size
        child[a:b+1] = p1[a:b+1]
        genes_in_child = set(child[a:b+1])
        ptr = (b + 1) % size
        for gene in p2[b+1:] + p2[:b+1]:
            if gene not in genes_in_child:
                child[ptr] = gene
                genes_in_child.add(gene) # Keep set updated just in case (though genes are unique)
                ptr = (ptr + 1) % size
        return child

    def mutate_inversion(genome):
        '''
        Implements Inversion Mutation by selecting a random segment of the genome and reversing it.
        This mutation is effective for TSP-like problems as it can significantly alter the path structure,
        allowing the algorithm to escape local minima.
        Args:
            genome (list[int]): The genome to mutate (sequence of virtual node indices).
        Returns:
            list[int]: The mutated genome.
        '''
        if random.random() < MUTATION_RATE:
            size = len(genome)
            a, b = sorted(random.sample(range(size), 2))
            genome[a:b+1] = genome[a:b+1][::-1]
        return genome

    # --- Evolution Loop ---
    cities_indices = list(range(1, num_total_nodes))
    population = []
    population.append(generate_greedy_genome()) # start with a greedy solution to give the GA a strong initial candidate 

    # Randomly generate the rest of the initial population by shuffling the city indices.
    for _ in range(POPULATION_SIZE - 1):
        ind = cities_indices[:]
        random.shuffle(ind)
        population.append(ind)

    # Evaluate initial population and sort by fitness (cost) - Generation 0
    scored_pop = []
    for genome in population:
        cost, _ = evaluate_genome(genome) 
        scored_pop.append((cost, genome))
    
    # Sort by Cost Ascending (Lower is better).
    # scored_pop[0] becomes the current best solution (Elite)
    scored_pop.sort(key=lambda x: x[0])

    # Track the Global Best solution found so far across all generations.
    best_fitness = scored_pop[0][0]
    best_genome = list(scored_pop[0][1])

    # Iterate through the evolutionary process
    for gen in range(GENERATIONS):
        # --- ELITISM ---
        # Directly copy the top N best individuals to the next generation.
        # This guarantees that the best solution found is never lost due to destructive crossover/mutation.
        new_pop = [list(g) for _, g in scored_pop[:ELITISM_SIZE]]

        # --- OFFSPRING GENERATION ---
        # Fill the rest of the new population until we reach POPULATION_SIZE.
        while len(new_pop) < POPULATION_SIZE:
            # Tournament Selection for Parent 1:
            # Pick 'TOURNAMENT_SIZE' random individuals and select the one with lowest cost.
            candidates = random.sample(scored_pop, TOURNAMENT_SIZE)
            p1 = min(candidates, key=lambda x: x[0])[1]

            # Tournament Selection for Parent 2:
            candidates = random.sample(scored_pop, TOURNAMENT_SIZE)
            p2 = min(candidates, key=lambda x: x[0])[1]

            # Crossover (OX1): Combine parents' traits preserving order.
            # Mutation (Inversion): Apply small random changes to escape local minima.
            child = mutate_inversion(crossover_ox1(p1, p2))
            # Add child to the new population
            new_pop.append(child)
        
        # Replace the old population with the new generation
        population = new_pop

        # --- RE-EVALUATION ---
        # Calculate fitness for the new generation.
        scored_pop = []
        for genome in population:
            cost, _ = evaluate_genome(genome)
            scored_pop.append((cost, genome))

            # Update Global Best if we found a new record low cost.
            if cost < best_fitness:
                best_fitness = cost
                best_genome = list(genome)
        
        # Sort again to prepare for the next generation's Elitism and Selection.
        scored_pop.sort(key=lambda x: x[0])

    # Take the absolute best genome found and reconstruct the full path.
    # We ignore the cost return (_) here because we only need the path.
    _, final_path = evaluate_genome(best_genome)
    return final_path

def write_report_to_file(p, G, path, my_cost, filename, density):
    """
    Generates the detailed text report required for the assignment.
    Includes Performance Comparison, City Coverage, and Trip Log.
    """
    # Calculate Baseline for comparison using the OFFICIAL method
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

# --- MAIN BLOCK ---
if __name__ == "__main__":
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