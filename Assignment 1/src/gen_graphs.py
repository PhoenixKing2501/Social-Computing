import gzip
import os
import random
import shutil

SEED = 42

random.seed(SEED)


def gunzip_file(gz_file: str) -> None:
    with gzip.open(gz_file, 'rb') as f_in:
        with open(gz_file[:-3], 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
# END gunzip_file


def make_facebook_graph() -> None:
    with open('subgraphs/facebook.elist', 'w') as f:
        with open('data/facebook_combined.txt', 'r') as fb:
            for line in fb:
                if line[0] == '#':
                    continue
                u, v = map(int, line.strip().split())

                # Remove all nodes that have IDs
                # divisible by 4
                if (u % 4 != 0) and (v % 4 != 0):
                    f.write(f'{u} {v}\n')
            # END for
        # END with
    # END with
# END make_facebook_graph


def make_epinions_graph() -> None:
    with open('subgraphs/epinions.elist', 'w') as f:
        with open('data/soc-sign-epinions.txt', 'r') as ep:
            for line in ep:
                if line[0] == '#':
                    continue
                u, v, w = map(int, line.strip().split())

                # Keep only the nodes whose IDs
                # are divisible by 5, along with
                # their signed weights
                if (u % 5 == 0) and (v % 5 == 0):
                    f.write(f'{u} {v} {w}\n')
            # END for
        # END with
    # END with
# END make_epinions_graph


def make_random_graph() -> None:
    NODES = 1000
    EDGES = 50000

    all_edges = set()

    for _ in range(EDGES):
        u, v = 0, 0

        while u == v or (u, v) in all_edges:
            u = random.randint(1, NODES)
            v = random.randint(1, NODES)
            if u > v:
                u, v = v, u
        # END while

        all_edges.add((u, v))
    # END for

    # sort all_edges with u < v and by u
    all_edges = sorted(all_edges)

    with open('networks/random.elist', 'w') as f:
        for u, v in all_edges:
            f.write(f'{u-1} {v-1}\n')
    # END with
# END make_random_graph


def make_smallworld_graph() -> None:
    NODES = 1000
    DEGREE = 50
    REWIRE_PROB = 0.6

    all_edges = set()

    # Create a ring lattice
    for i in range(NODES):
        for j in range(1, DEGREE // 2 + 1):
            v = (i + j) % NODES
            all_edges.add((i, v))
        # END for
    # END for

    # Rewire the edges
    for i in range(NODES):
        for j in range(1, DEGREE // 2 + 1):
            if random.random() < REWIRE_PROB:
                while True:
                    v = random.randint(0, NODES - 1)
                    if (v != i and
                        (i, v) not in all_edges and
                            (v, i) not in all_edges):
                        break
                # END while

                all_edges.remove((i, (i + j) % NODES))
                all_edges.add((i, v))
            # END if
        # END for
    # END for

    # sort all_edges with u < v and by u
    all_edges = map(lambda x: (min(x), max(x)), all_edges)
    all_edges = sorted(all_edges)

    with open('networks/smallworld.elist', 'w') as f:
        for u, v in all_edges:
            f.write(f'{u} {v}\n')
    # END with
# END make_smallworld_graph


def main():
    # Extract data
    for file in os.listdir('data'):
        if file.endswith('.gz'):
            gunzip_file(f'data/{file}')

    # Make `subgraphs` directory
    if not os.path.exists('subgraphs'):
        os.makedirs('subgraphs')

    # Make `networks` directory
    if not os.path.exists('networks'):
        os.makedirs('networks')

    # Generate graphs
    # 1. facebook_combined.txt -> facebook.elist
    make_facebook_graph()

    # 2. soc-sign-epinions.txt -> epinions.elist
    make_epinions_graph()

    # 3. random.elist [#Nodes: 1000, #Edges: 50000]
    make_random_graph()

    # 4. smallworld.elist [#Nodes: 1000, Node Degree: 50, Rewire Probability = 0.6]
    make_smallworld_graph()
# END main


if __name__ == "__main__":
    main()