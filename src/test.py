import networkx as nx
import numpy as np
import time
import random
from itertools import combinations

def generate_synthetic_graph(num_nodes, density, seed=42):
    """
    Genera un grafo casuale ponderato per il test.
    Simula la struttura di Problem.py senza dipendenze esterne.
    """
    G = nx.Graph()
    G.add_nodes_from(range(num_nodes))
    
    rng = np.random.default_rng(seed)
    
    # Aggiungiamo archi casuali basati sulla densità
    # Usiamo un approccio vettoriale semplice per velocità di generazione
    possible_edges = list(combinations(range(num_nodes), 2))
    
    # Per grafi grandi, selezioniamo randomicamente senza creare la lista completa (che occuperebbe troppa RAM)
    # Qui per semplicità di benchmark su N < 1000 usiamo un loop controllato
    edge_count = 0
    for u, v in possible_edges:
        if rng.random() < density:
            # Peso casuale tra 1.0 e 100.0 (simula distanza)
            w = rng.random() * 99 + 1
            G.add_edge(u, v, weight=w)
            edge_count += 1
            
    # Assicuriamo che il grafo sia connesso (o quasi) aggiungendo un percorso sequenziale
    # (Come fa il tuo Problem.py con 'c2 == c1 + 1')
    for i in range(num_nodes - 1):
        if not G.has_edge(i, i+1):
            G.add_edge(i, i+1, weight=rng.random() * 99 + 1)
            
    return G

def run_benchmark():
    print(f"{'NODES':<10} | {'DENSITY':<10} | {'FW (NumPy) [s]':<15} | {'Dijkstra [s]':<15} | {'SPEEDUP':<10}")
    print("-" * 75)

    # Scenari di Test: (Nodi, Densità)
    # Nota: Densità 1.0 = Grafo Completo (worst case per Dijkstra)
    # Nota: Densità 0.1 = Grafo Sparso
    scenarios = [
        (50, 0.5),
        (100, 0.2),    # Scenario tipico del tuo progetto
        (100, 0.8),    # Grafo denso
        (200, 0.5),    # Aumento carico
        (300, 0.5),    # Il divario dovrebbe diventare enorme qui
        (1000, 0.1),   # Grafo molto grande e sparso
        (1000, 1),   # Grafo molto grande e denso
    ]

    for n, d in scenarios:
        # 1. Preparazione Grafo
        G = generate_synthetic_graph(n, d)
        
        # 2. Test Floyd-Warshall (NumPy / C-Optimized)
        start_fw = time.time()
        _ = nx.floyd_warshall_numpy(G, weight='weight')
        end_fw = time.time()
        time_fw = end_fw - start_fw

        # 3. Test Dijkstra (Python Loop)
        # Dobbiamo eseguirlo N volte (una per ogni nodo sorgente) per ottenere la matrice completa
        start_dijk = time.time()
        for node in G.nodes():
            _ = nx.single_source_dijkstra_path_length(G, node, weight='weight')
        end_dijk = time.time()
        time_dijk = end_dijk - start_dijk

        # 4. Calcolo Speedup
        speedup = time_dijk / time_fw if time_fw > 0 else 0.0

        print(f"{n:<10} | {d:<10} | {time_fw:<15.5f} | {time_dijk:<15.5f} | {speedup:<10.1f}x")

if __name__ == "__main__":
    print("Avvio Benchmark: Floyd-Warshall (NumPy) vs Repeated Dijkstra (Python)...")
    print("Nota: FW è O(V^3), Dijkstra ripetuto è O(V * E * logV).")
    print("Tuttavia, l'overhead di Python penalizza drasticamente Dijkstra.\n")
    
    run_benchmark()
    
    print("\nBenchmark completato.")