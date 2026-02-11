import networkx as nx
import numpy as np
import time
import random
from itertools import combinations

def generate_synthetic_graph(num_nodes, density, seed=42):
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    rng = np.random.default_rng(seed)
    
    possible_edges = list(combinations(range(num_nodes), 2))
    
    edge_count = 0
    for u, v in possible_edges:
        if rng.random() < density:
            w = rng.random() * 99 + 1
            G.add_edge(u, v, weight=w)
            edge_count += 1
            
    for i in range(num_nodes - 1):
        if not G.has_edge(i, i+1):
            G.add_edge(i, i+1, weight=rng.random() * 99 + 1)
            
    return G

def run_benchmark():
    print(f"{'NODES':<10} | {'DENSITY':<10} | {'FW (NumPy) [s]':<15} | {'Dijkstra [s]':<15} | {'SPEEDUP':<10}")
    print("-" * 75)

    scenarios = [
        (50, 0.5),
        (100, 0.2),    
        (100, 0.8),    
        (200, 0.5),    
        (300, 0.5),    
        (1000, 0.1),   
        (1000, 1),   
    ]

    for n, d in scenarios:
        G = generate_synthetic_graph(n, d)
        
        start_fw = time.time()
        _ = nx.floyd_warshall_numpy(G, weight='weight')
        end_fw = time.time()
        time_fw = end_fw - start_fw

        start_dijk = time.time()
        for node in G.nodes():
            _ = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        end_dijk = time.time()
        time_dijk = end_dijk - start_dijk

        speedup = time_dijk / time_fw if time_fw > 0 else 0.0

        print(f"{n:<10} | {d:<10} | {time_fw:<15.5f} | {time_dijk:<15.5f} | {speedup:<10.1f}x")

if __name__ == "__main__":
    print("Start Benchmark: Floyd-Warshall (NumPy) vs Dijkstra (Python)...")
    print("Note: FW is O(V^3), repeated Dijkstra is O(V * E * logV).")    
    run_benchmark()
    
    print("\nBenchmark completed.")