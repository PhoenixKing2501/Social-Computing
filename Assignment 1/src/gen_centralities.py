import os
import sys
from collections import defaultdict, deque
from copy import deepcopy
from math import hypot
from typing import Deque, Dict, Set, Tuple


class Graph():
    def __init__(self):
        self.nodes: Set[int] = set()
        self.graph: Dict[int, Set[int]] = defaultdict(set[int])
        self.neg_edges: Dict[int, Set[int]] = defaultdict(set[int])
    # END __init__

    def addEdge(self, u: int, v: int, w: int = 1):
        self.nodes.add(u)
        self.nodes.add(v)
        self.graph[u].add(v)
        self.graph[v].add(u)
        if w < 0:
            self.neg_edges[u].add(v)
            self.neg_edges[v].add(u)
        # END if w < 0
    # END addEdge

    def getNodes(self) -> Set[int]:
        return self.nodes

    def getEdges(self) -> Dict[int, Set[int]]:
        return self.graph

    def getNegEdges(self) -> Dict[int, Set[int]]:
        return self.neg_edges

    def isNegEdge(self, u: int, v: int) -> bool:
        return v in self.neg_edges[u]

    def getDegree(self, u: int) -> int:
        return len(self.graph[u])

    def getNeighbors(self, u) -> Set[int]:
        return self.graph[u]

    def isNeighbor(self, u: int, v: int) -> bool:
        return v in self.graph[u]
# END class Graph


def loadGraph(path: str) -> Graph:
    graph = Graph()
    with open(path, 'r') as fb:
        for line in fb:
            if line[0] == '#':
                continue

            edge = line.strip().split()
            u, v, w = map(int, edge if len(edge) == 3 else edge + [1])
            graph.addEdge(u, v, w)
        # END for line in fb
    # END with open(path, 'r')

    return graph
# END loadGraph


# Following the algorithm in wikipedia [https://en.wikipedia.org/wiki/Brandes%27_algorithm]
def performBrandes(
    graph: Graph
) -> Tuple[Dict[int, Dict[int, int]],
           Dict[int, float]]:

    shortest_path_all: Dict[int, Dict[int, int]] = {}
    betweenness_centrality: Dict[int, float] = defaultdict(float)

    nodes = graph.getNodes()

    for s in nodes:
        dependency: Dict[int, float] = defaultdict(float)
        previous: Dict[int, Set[int]] = defaultdict(set[int])
        num_shortest_paths: Dict[int, int] = defaultdict(int)
        shortest_path: Dict[int, int] = {}

        # Initialization
        num_shortest_paths[s] = 1
        shortest_path[s] = 0

        queue: Deque[int] = deque([s])
        stack: Deque[int] = deque()

        # Single-source shortest paths
        while queue:
            v = queue.popleft()
            stack.append(v)

            for w in graph.getNeighbors(v):
                if w not in shortest_path:
                    shortest_path[w] = shortest_path[v] + 1
                    queue.append(w)
                # END if w not in shortest_path

                if shortest_path[w] == shortest_path[v] + 1:
                    num_shortest_paths[w] += num_shortest_paths[v]
                    previous[w].add(v)
                # END if shortest_path[w] == shortest_path[v] + 1
            # END for w in graph.getNeighbors(v)
        # END while queue

        # Accumulation
        while stack:
            w = stack.pop()

            for v in previous[w]:
                dependency[v] += ((num_shortest_paths[v] /
                                   num_shortest_paths[w]) *
                                  (1 + dependency[w]))
            # END for v in previous[w]

            if w != s:
                betweenness_centrality[w] += dependency[w]
            # END if w != s
        # END while stack

        shortest_path_all[s] = shortest_path
    # END for s in graph.getNodes()

    for node in nodes:
        if node not in betweenness_centrality:
            betweenness_centrality[node] = 0.0
        else:
            betweenness_centrality[node] /= 2
        # END if node not in betweenness_centrality
    # END for node in graph.getNodes()

    return shortest_path_all, betweenness_centrality
# END performBrandes


def getAllClosenessCentrality(
    graph: Graph,
    shortest_path: Dict[int, Dict[int, int]],
    normalize: bool = True
) -> Dict[int, float]:

    closeness_centrality: Dict[int, float] = {}

    for node in graph.getNodes():
        if len(shortest_path[node]) <= 1:
            closeness_centrality[node] = 0.0
            continue
        # END if len(shortest_path[node]) <= 1

        sum_shortest_path = sum(shortest_path[node].values())
        farness_centrality = (sum_shortest_path /
                              (len(shortest_path[node]) - 1))

        # Multiply farness centrality by the number of nodes in the graph
        # divided by the number of nodes reachable from the node
        # to normalize the value
        # This is done as how it is in SNAP (I've checked the source code)
        if normalize:
            farness_centrality *= ((len(graph.getNodes()) - 1) /
                                   (len(shortest_path[node]) - 1))
        # END if normalize

        closeness_centrality[node] = 1.0 / farness_centrality
    # END for node in graph.getNodes()

    return closeness_centrality
# END getAllClosenessCentrality


def checkConvergence(
    prev: Dict[int, float],
    curr: Dict[int, float],
    epsilon: float
) -> bool:
    return hypot(*(prev[node] - curr[node]
                   for node in prev)) < epsilon
# END checkConvergence


def getAllPageRank(
    graph: Graph,
    damping_factor: float = 0.8,
    epsilon: float = 1e-6,
    max_iter: int = 1_000
) -> Dict[int, float]:

    if len(graph.getNegEdges()) == 0:
        preference: Dict[int, float] = {node: 1 / len(graph.getNodes())
                                        for node in graph.getNodes()}
    else:  # Only for epinions dataset there are negative edges
        preference: Dict[int, float] = {node: sum(-1 if graph.isNegEdge(node, v) else 1
                                                  for v in graph.getNeighbors(node))
                                        for node in graph.getNodes()}
        pref_sum = sum(preference.values())
        preference = {node: value / pref_sum
                      for node, value in preference.items()}
    # END if len(graph.getNegEdges()) == 0

    page_rank = deepcopy(preference)

    for i in range(max_iter):
        page_rank_prev = deepcopy(page_rank)

        for node in graph.getNodes():
            page_rank[node] = (1 - damping_factor) * preference[node] + \
                damping_factor * sum(page_rank_prev[v] / len(graph.getNeighbors(v))
                                     for v in graph.getNeighbors(node))
        # END for node in graph.getNodes()

        # Check for convergence
        if checkConvergence(page_rank_prev, page_rank, epsilon):
            # print(f"Converged after {i} iterations", file=sys.stderr)
            break
    # END for _ in range(max_iter)

    return page_rank
# END getAllPageRank


def getInfluencerNodes(name: str) -> int:
    LIMIT = 200
    with open(f'centralities/betweenness_{name}.txt', 'r') as fb:
        betweenness = set([int(line.split()[0].strip())
                           for line in fb.readlines()][:LIMIT])

    with open(f'centralities/closeness_{name}.txt', 'r') as fb:
        closeness = set([int(line.split()[0].strip())
                         for line in fb.readlines()][:LIMIT])

    with open(f'centralities/pagerank_{name}.txt', 'r') as fb:
        pagerank = set([int(line.split()[0].strip())
                        for line in fb.readlines()][:LIMIT])

    return len(betweenness & closeness & pagerank)
# END getInfluencerNodes


def saveCentrality(centrality: Dict[int, float], name: str, cntType: str):
    with open(f'centralities/{cntType}_{name}.txt', 'w') as fb:
        for node, value in sorted(centrality.items(),
                                  key=lambda x: x[1],
                                  reverse=True):
            fb.write(f'{node}\t{value:.4f}\n')
        # END for node, value in sorted(centrality.items(),
    # END with open(f'centralities/{name}_{cntType}.txt', 'w')
# END saveCentrality


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_structure.py <path_to_graph>")
        sys.exit(1)
    # END if len(sys.argv) != 2

    name = os.path.basename(sys.argv[1]).split('.')[0]

    # Load the graph
    graph: Graph = loadGraph(sys.argv[1])

    # Make `centralities` directory
    if not os.path.exists('centralities'):
        os.makedirs('centralities')

    # Perform Brandes algorithm to get shortest path and betweenness centrality
    shortest_path, betweenness_centrality = performBrandes(graph)

    # Save betweenness centrality
    saveCentrality(betweenness_centrality, name, 'betweenness')

    del betweenness_centrality

    # Get all closeness centrality
    closeness_centrality = getAllClosenessCentrality(graph, shortest_path)

    # Save closeness centrality
    saveCentrality(closeness_centrality, name, 'closeness')

    del closeness_centrality

    # Get all page rank
    DAMPING_FACTOR = 0.8
    page_rank = getAllPageRank(graph, DAMPING_FACTOR)

    # Save page rank
    saveCentrality(page_rank, name, 'pagerank')

    del page_rank

    print(f"Number of influencer nodes: {getInfluencerNodes(name)}")
# END main


if __name__ == "__main__":
    main()
