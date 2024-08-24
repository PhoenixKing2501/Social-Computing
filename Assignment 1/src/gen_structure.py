import snap
import sys

SEED = 42
Rnd = snap.TRnd(SEED)
Rnd.Randomize()


def main():
    if len(sys.argv) != 2:
        print("Usage: python gen_structure.py <path_to_graph>")
        sys.exit(1)
    # END if len(sys.argv) != 2

    # Load the graph
    graph = snap.LoadEdgeList(snap.TUNGraph, sys.argv[1], 0, 1)

    # Graph stats:
    print(f"Number of nodes: {graph.GetNodes()}")
    print(f"Number of edges: {graph.GetEdges()}")
    print(f"Number of nodes with degree=7: {graph.CntDegNodes(7)}")
    print(f"Node id(s) with highest degree: {graph.GetMxDegNId()}")
# END main


if __name__ == "__main__":
    main()
