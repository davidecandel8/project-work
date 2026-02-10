import logging
from itertools import combinations

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

from icecream import ic

class Problem:
    _graph: nx.Graph
    _alpha: float
    _beta: float

    def __init__(
        self,
        num_cities: int,
        *,
        alpha: float = 1.0,
        beta: float = 1.0,
        density: float = 0.5,
        seed: int = 42,
    ):
        rng = np.random.default_rng(seed) # random generator with seed
        self._alpha = alpha # cost coefficient
        self._beta = beta # cost exponent
        cities = rng.random(size=(num_cities, 2)) # random city positions
        cities[0, 0] = cities[0, 1] = 0.5 # depot at center

        self._graph = nx.Graph() # create empty graph
        self._graph.add_node(0, pos=(cities[0, 0], cities[0, 1]), gold=0) 
        for c in range(1, num_cities):
            self._graph.add_node(c, pos=(cities[c, 0], cities[c, 1]), gold=(1 + 999 * rng.random()))

        tmp = cities[:, np.newaxis, :] - cities[np.newaxis, :, :] # pairwise differences: x1-x2, y1-y2
        d = np.sqrt(np.sum(np.square(tmp), axis=-1)) # pairwise euclidean distances
        for c1, c2 in combinations(range(num_cities), 2): 
            if rng.random() < density or c2 == c1 + 1:
                self._graph.add_edge(c1, c2, dist=d[c1, c2])

        assert nx.is_connected(self._graph) # ensure graph is connected

    @property # getter for graph
    def graph(self) -> nx.Graph:
        return nx.Graph(self._graph)

    @property # getter for alpha
    def alpha(self):
        return self._alpha

    @property # getter for beta
    def beta(self):
        return self._beta

    def cost(self, path, weight): # cost of traveling path with load weight
        dist = nx.path_weight(self._graph, path, weight='dist')
        return dist + (self._alpha * dist * weight) ** self._beta

    def baseline(self): # baseline solution cost
        total_cost = 0
        for dest, path in nx.single_source_dijkstra_path(
            self._graph, source=0, weight='dist'
        ).items(): # shortest paths from depot to all nodes
            cost = 0
            for c1, c2 in zip(path, path[1:]): # for each edge in path
                cost += self.cost([c1, c2], 0) # cost with zero load
                cost += self.cost([c1, c2], self._graph.nodes[dest]['gold']) # cost with full load
            logging.debug(
                f"dummy_solution: go to {dest} ({' > '.join(str(n) for n in path)} ({cost})"
            )
            total_cost += cost
        return total_cost

    def plot(self):
        plt.figure(figsize=(10, 10))
        pos = nx.get_node_attributes(self._graph, 'pos')
        size = [100] + [self._graph.nodes[n]['gold'] for n in range(1, len(self._graph))]
        color = ['red'] + ['lightblue'] * (len(self._graph) - 1)
        return nx.draw(self._graph, pos, with_labels=True, node_color=color, node_size=size)
