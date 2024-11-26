import numpy as np
from pprint import pp

# .       A  B  C  D  E  F
GRAPH = [[0, 1, 0, 0, 0, 0,],  # A
         [0, 0, 0, 1, 0, 1,],  # B
         [1, 0, 0, 1, 0, 0,],  # C
         [0, 0, 0, 0, 1, 0,],  # D
         [0, 1, 0, 0, 0, 1,],  # E
         [0, 0, 0, 0, 0, 0,]]  # F

ALPHA = 0.75
MAX_ITER = 100


def Page_Rank_Eigen():
    # Page Rank Algorithm

    # Step 1: Create a matrix M
    M = np.array(GRAPH)

    print("Matrix M:", M, sep="\n", end="\n\n")

    # Step 2: Add 1 to each element of zero out-degree nodes
    for i in range(len(M)):
        if np.sum(M[i]) == 0:
            M[i] = np.ones(len(M))

    print("Matrix M after adding 1 to zero out-degree nodes:",
          M, sep="\n", end="\n\n")

    # Step 3.1: Calculate the out-degree of each node
    out_degree = np.sum(M, axis=1)

    # Step 3.2: Calculate probability matrix P
    P = M / out_degree[:, None]

    print("Probability matrix P:", P, sep="\n", end="\n\n")

    # Step 3.3: Calculate the teleportation matrix
    N = len(P)
    E = np.ones((N, N)) / N

    print("Teleportation matrix E:", E, sep="\n", end="\n\n")

    print("alpha:", ALPHA, end="\n\n")

    # Step 4: Calculate the transition matrix A
    A = ALPHA * P + (1 - ALPHA) * E

    print("A = alpha * P + (1 - alpha) * E\n")

    print("Transition matrix A:", A, sep="\n", end="\n\n")

    print("Sum of each row in A:", np.sum(A, axis=1), end="\n\n")

    # Step 5: Calculate the PageRank vector
    # Transpose A to get the left eigenvectors
    eigenvalues, eigenvectors = np.linalg.eig(A.T)

    for i in range(len(eigenvalues)):
        print(f"Eigenvalue {i+1}: {eigenvalues[i]}")
        print(f"Eigenvector {i+1}:", eigenvectors[:, i], end="\n\n")

    # Get the eigenvector corresponding to the largest eigenvalue
    idx = np.argmax(eigenvalues.real)
    pagerank = eigenvectors[:, idx].real

    # Make the ranks +ve
    print("PageRank vector:", pagerank)
    print("Sum:", np.sum(pagerank))
    print("Norm:", np.linalg.norm(pagerank))

    pagerank /= np.sum(pagerank)

    print("\nPageRank vector:")
    for i, rank in enumerate(pagerank):
        print(f"Node {chr(65+i)}: {rank:.8f}")

    # Prove that the PageRank vector is correct
    print("\nPageRank * A:", pagerank @ A, end="\n\n")


def Page_Rank_Power_Iteration():
    # Page Rank Algorithm using Power Iteration

    # Step 1: Create a matrix M
    M = np.array(GRAPH)

    in_degree = np.sum(M, axis=0)
    page_rank = in_degree / np.sum(in_degree)

    print("Matrix M:", M, sep="\n", end="\n\n")

    # Uncomment the following code to add 1 to each element of zero out-degree nodes
    # vvvvvvvvvvvvvvvvvvvvvvvv
    # for i in range(len(M)):
    #     if np.sum(M[i]) == 0:
    #         M[i] = np.ones(len(M))

    # print("Matrix M after adding 1 to zero out-degree nodes:",
    #       M, sep="\n", end="\n\n")
    # ^^^^^^^^^^^^^^^^^^^^^^^^^^^^
    # Upto here

    N = len(M)

    out_degree = np.sum(M, axis=1)

    # page_rank = np.ones(N) / N

    for i in range(MAX_ITER):
        page_rank_new = np.zeros_like(page_rank)
        for u in range(N):
            t = 0
            for v in range(N):
                if M[v, u] == 1:
                    t += page_rank[v] / out_degree[v]
            page_rank_new[u] = (1 - ALPHA) / N + ALPHA * t

        # Normalize the PageRank vector
        page_rank = page_rank_new / np.sum(page_rank_new)

        print(f"Iteration {i+1}: {page_rank}")
        print(f"Sum: {np.sum(page_rank)}\n")

    print("Final PageRank vector:")
    for i, rank in enumerate(page_rank):
        print(f"Node {chr(65+i)}: {rank:.8f}")

    print("\nSum:", np.sum(page_rank))
    print("Norm:", np.linalg.norm(page_rank))


# My method
def Page_Rank_Power_Iteration_2():
    # Page Rank Algorithm using Power Iteration (Alternative)

    # Step 1: Create a matrix M
    M = np.array(GRAPH)

    in_degree = np.sum(M, axis=0)
    page_rank = in_degree / np.sum(in_degree)

    print("Matrix M:", M, sep="\n", end="\n\n")

    for i in range(len(M)):
        if np.sum(M[i]) == 0:
            M[i] = np.ones(len(M))

    print("Matrix M after adding 1 to zero out-degree nodes:",
          M, sep="\n", end="\n\n")

    N = len(M)

    out_degree = np.sum(M, axis=1)

    # page_rank = np.ones(N) / N

    for i in range(MAX_ITER):
        contribution = np.array([page_rank[v] / out_degree[v]
                                 for v in range(N)])

        page_rank = np.zeros_like(page_rank)

        for u in range(N):
            for v in range(N):
                if M[v, u] == 1:
                    page_rank[u] += contribution[v]

        page_rank = (1 - ALPHA) / N + ALPHA * page_rank

        # Normalize the PageRank vector
        page_rank /= np.sum(page_rank)

        print(f"Iteration {i+1}: {page_rank}")
        print(f"Sum: {np.sum(page_rank)}\n")

    print("Final PageRank vector:")
    for i, rank in enumerate(page_rank):
        print(f"Node {chr(65+i)}: {rank:.8f}")

    print("\nSum:", np.sum(page_rank))
    print("Norm:", np.linalg.norm(page_rank))


def main():
    print("Page Rank Algorithm using Eigenvalues and Eigenvectors:\n")
    Page_Rank_Eigen()
    print()

    print("Page Rank Algorithm using Power Iteration:\n")
    Page_Rank_Power_Iteration()
    print()

    print("Page Rank Algorithm using Power Iteration (Alternative):\n")
    Page_Rank_Power_Iteration_2()
    print()


if __name__ == "__main__":
    main()
