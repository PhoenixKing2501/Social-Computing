import snap

ALPHA = 0.75
MAX_ITER = 100


def main():
    graph = snap.TNGraph.New()
    for i in range(1, 7):
        graph.AddNode(i)

    # A, B, C, D, E, F
    # 1, 2, 3, 4, 5, 6

    graph.AddEdge(1, 2)  # A -> B
    graph.AddEdge(2, 4)  # B -> D
    graph.AddEdge(2, 6)  # B -> F
    graph.AddEdge(3, 1)  # C -> A
    graph.AddEdge(3, 4)  # C -> D
    graph.AddEdge(4, 5)  # D -> E
    graph.AddEdge(5, 2)  # E -> B
    graph.AddEdge(5, 6)  # E -> F

    # PageRank
    PRankH = graph.GetPageRank(0.75, 1e-10, MAX_ITER)

    print("PageRank:")
    for item in PRankH:
        print(f"Node {item}: {PRankH[item]}")

    # for i in range(1, 7):
    #     graph.AddEdge(6, i)  # F -> A, B, C, D, E, F

    # PageRank
    PRankH = graph.GetPageRank(0.75, 1e-10, MAX_ITER)

    print("\nPageRank with phantom edges from F:")
    for item in PRankH:
        print(f"Node {item}: {PRankH[item]}")


if __name__ == "__main__":
    main()
