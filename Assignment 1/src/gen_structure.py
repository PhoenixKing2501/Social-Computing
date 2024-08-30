import heapq
import os
import sys
from collections import defaultdict
from typing import Dict

import matplotlib.pyplot as plt
import snap

SEED = 42
Rnd = snap.TRnd(SEED)  # type: ignore
Rnd.Randomize()


def makeDir(directory: str):
    if not os.path.exists(directory):
        os.makedirs(directory)
# END makeDir


def allNodesMaxDeg(graph):
    degrees = [(item.GetVal2(), item.GetVal1())
               for item in graph.GetNodeInDegV()]
    deg = max(degrees)[0]
    maxNodes = (item[1]
                for item in degrees
                if item[0] == deg)
    return (str(node) for node in maxNodes)
# END allNodesMaxDeg


def plotDegDist(graph, name):
    DegToCntV = graph.GetDegCnt()
    x = [item.GetVal1() for item in DegToCntV]
    y = [item.GetVal2() for item in DegToCntV]

    # add lines joining the points
    plt.plot(x, y, 'r-')
    plt.scatter(x, y, color='b', marker='x')
    plt.xlabel('Log10(Degree)')
    plt.ylabel('Log10(Count)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Degree Distribution')
    plt.savefig(f'plots/deg_dist_{name}.png')
    plt.close()
# END plotDegDist


def generateShortestPathDist(graph):

    hop_count: Dict[int, int] = defaultdict(int)

    for NI in graph.Nodes():
        _, NodeVec = graph.GetNodesAtHops(NI.GetId(), False)
        for item in NodeVec:
            hop_count[item.GetVal1()] += item.GetVal2()
    # END for NI in graph.Nodes()

    return {k: v//2 for k, v in hop_count.items()}
# END generateShortestPathDist


def plotShortestPathDist(graph, name):
    hop_count = generateShortestPathDist(graph)

    x = list(hop_count.keys())
    y = list(hop_count.values())

    # add lines joining the points
    plt.plot(x, y, 'r-')
    plt.scatter(x, y, color='b', marker='x')
    plt.xlabel('Shortest Path Length')
    plt.ylabel('Log10(Count)')
    plt.yscale('log')
    plt.title('Shortest Path Distribution')
    plt.savefig(f'plots/shortest_path_{name}.png')
    plt.close()
# END plotShortestPathDist


def plotConCompSzDist(graph, name):
    CntV = graph.GetSccSzCnt()
    x = [item.GetVal1() for item in CntV]
    y = [item.GetVal2() for item in CntV]

    # add lines joining the points
    plt.plot(x, y, 'r-')
    plt.scatter(x, y, color='b', marker='x')
    plt.xlabel('Log10(Connected Component Size)')
    plt.ylabel('Log10(Count)')
    plt.xscale('log')
    plt.yscale('log')
    plt.title('Connected Component Size Distribution')
    plt.savefig(f'plots/connected_comp_{name}.png')
    plt.close()
# END plotConCompSzDist


def plotCCDist(graph, name):
    NIdCCfH = graph.GetNodeClustCfAll()

    x = NIdCCfH.values()

    # plot histogram
    plt.hist(x, bins=100, color='b', edgecolor='black', log=True)
    # add lines joining the midpoints of the bars
    plt.xlabel('Clustering Coefficient')
    plt.ylabel('Log10(Count)')
    plt.title('Clustering Coefficient Distribution')
    plt.savefig(f'plots/clustering_coeff_{name}.png')
    plt.close()
# END plotCCDist


def getTopKDegNodes(graph, k: int):
    degrees = ((item.GetVal1(), item.GetVal2())
               for item in graph.GetNodeInDegV())
    degrees = heapq.nlargest(k, degrees,
                             key=lambda x: x[1])
    return (str(item[0]) for item in degrees)
# END getTopKDegNodes


def getTopKClosenessCentrNodes(graph, k: int):
    closenessCentr = ((nodeID, graph.GetClosenessCentr(nodeID))
                      for nodeID in (node.GetId()
                                     for node in graph.Nodes()))
    closenessCentr = heapq.nlargest(k, closenessCentr,
                                    key=lambda x: x[1])
    return (str(item[0]) for item in closenessCentr)
# END getTopKClosenessCentrNodes


def getTopKBetweennessCentrNodes(graph, k: int):
    betweennessCentr, _ = graph.GetBetweennessCentr()
    betweennessCentr = [(nodeID, centr) for nodeID, centr in betweennessCentr.items()]
    betweennessCentr = heapq.nlargest(k, betweennessCentr,
                                      key=lambda x: x[1])
    return (str(item[0]) for item in betweennessCentr)
# END getTopKBetweennessCentrNodes


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_structure.py <path_to_graph>")
        sys.exit(1)
    # END if len(sys.argv) != 2

    name = os.path.basename(sys.argv[1]).split('.')[0]

    # Load the graph
    graph = snap.LoadEdgeList(snap.TUNGraph, sys.argv[1], 0, 1)  # type: ignore

    # Make `plots` directory
    if not os.path.exists('plots'):
        os.makedirs('plots')

    # Graph stats:
    # 1.a. Number of nodes
    print(f"Number of nodes: {graph.GetNodes()}")

    # 1.b. Number of edges
    print(f"Number of edges: {graph.GetEdges()}")

    # 2.a. Number of nodes which have degree = 7
    print(f"Number of nodes with degree=7: {graph.CntDegNodes(7)}")

    # 2.b. Node id(s) for the node with the highest degree.
    print(f"Node id(s) with highest degree: "
          f"{', '.join(allNodesMaxDeg(graph))}")

    # 2.c. Plot of the Degree distribution
    plotDegDist(graph, name)

    NTestNode = 1000
    diameters = graph.GetBfsEffDiamAll(NTestNode, False)
    # 3.a. Approximate full diameter (maximum shortest path length)
    #      starting from 1000 random test nodes.
    print(f"Approximate full diameter: {diameters[0]:.4f}")

    # 3.b. Approximate effective diameter computed starting from 1000 random test nodes.
    print(f"Approximate effective diameter: {diameters[2]}")

    # 3.c. Plot of the distribution of the shortest path lengths in the network.
    plotShortestPathDist(graph, name)

    # 4.a. Fraction of nodes in the largest connected component
    print(f"Fraction of nodes in largest connected component: "
          f"{graph.GetMxSccSz():.4f}")

    # 4.b. Number of edge bridges
    print(f"Number of edge bridges: {len(graph.GetEdgeBridges())}")

    # 4.c. Number of articulation points
    print(f"Number of articulation points: {len(graph.GetArtPoints())}")

    # 4.d. Plot of the distribution of sizes of connected components
    plotConCompSzDist(graph, name)

    clustInfo, _ = graph.GetClustCfAll()
    # 5.a. Average clustering coefficient of the network
    print(f"Average clustering coefficient: {clustInfo[0]:.4f}")

    # 5.b. Number of triads
    print(f"Number of triads: {clustInfo[1]}")

    # 5.c. Clustering coefficient of a randomly selected node.
    #      Also report the selected node id.
    node = graph.GetRndNId()
    print(f"Clustering coefficient of random node {node}: "
          f"{graph.GetNodeClustCf(node):.4f}")

    # 5.d. Number of triads a randomly selected node participates in.
    #      Also report the selected node id.
    print(f"Number of triads random node {node} participates in: "
          f"{graph.GetNodeTriads(node)}")

    # 5.e. Plot of the distribution of clustering coefficient
    plotCCDist(graph, name)

    # 6.a. Degree centrality
    print(f"Top 5 nodes by degree centrality: "
          f"{' '.join(getTopKDegNodes(graph, 5))}")

    # 6.b. Closeness centrality
    print(f"Top 5 nodes by closeness centrality: "
          f"{' '.join(getTopKClosenessCentrNodes(graph, 5))}")

    # 6.c. Betweenness centrality
    print(f"Top 5 nodes by betweenness centrality: "
          f"{' '.join(getTopKBetweennessCentrNodes(graph, 5))}")
# END main


if __name__ == "__main__":
    main()
