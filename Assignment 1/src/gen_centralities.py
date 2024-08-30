import os
import sys
from collections import defaultdict, deque
from typing import Deque, Dict, Set, Tuple


class Graph(object):
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
    # END addEdge

    def getNodes(self) -> Set[int]:
        return self.nodes

    def getEdges(self) -> Dict[int, Set[int]]:
        return self.graph

    def getNegEdges(self) -> Dict[int, Set[int]]:
        return self.neg_edges

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
    n_nodes = len(nodes)

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
                dependency[v] += (num_shortest_paths[v] /
                                  num_shortest_paths[w]) * (1 + dependency[w])
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
    shortest_path: Dict[int, Dict[int, int]]
) -> Dict[int, float]:

    closeness_centrality: Dict[int, float] = {}

    for node in graph.getNodes():
        sum_shortest_path = sum(shortest_path[node].values())
        closeness_centrality[node] = \
            (len(graph.getNodes()) - 1) / sum_shortest_path
    # END for node in graph.getNodes()

    return closeness_centrality
# END getAllClosenessCentrality


def getAllPageRank(graph):
    pass
# END getAllPageRank


def saveCentrality(centrality: Dict, name: str, type: str):
    pass
# END saveCentrality


def getAllClosenessCentrNodes(graph):
    closenessCentr = {nodeID: graph.GetClosenessCentr(nodeID)
                      for nodeID in (node.GetId()
                                     for node in graph.Nodes())}
    return closenessCentr
# END getAllClosenessCentrNodes


def getAllBetweennessCentrNodes(graph):
    betweennessCentr, _ = graph.GetBetweennessCentr()
    # betweennessCentr = [(nodeID, centr)
    #                     for nodeID, centr in betweennessCentr.items()]
    return dict(betweennessCentr)
# END getAllBetweennessCentrNodes


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_structure.py <path_to_graph>")
        sys.exit(1)
    # END if len(sys.argv) != 2

    name = os.path.basename(sys.argv[1]).split('.')[0]

    # Load the graph
    graph: Graph = loadGraph(sys.argv[1])

    # Make `centralities` directory
    # if not os.path.exists('centralities'):
    #     os.makedirs('centralities')

    # Perform Brandes algorithm to get shortest path and betweenness centrality
    shortest_path, betweenness_centrality = performBrandes(graph)

    # Get all closeness centrality
    closeness_centrality = getAllClosenessCentrality(graph, shortest_path)

# END main


if __name__ == "__main__":
    main()
