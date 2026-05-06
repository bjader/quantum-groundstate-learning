import logging
from typing import Tuple, List

import networkx as nx
import numpy as np


def read_edges_txt(path: str) -> Tuple[List[Tuple[int, int]], int]:
    """Read undirected 0-indexed edges, return unique edges as 0-indexed and N sites"""
    edges_set = set()
    n = 0
    with open(path, "r") as f:
        for line in f:
            s = line.strip()
            parts = s.split()
            if len(parts) < 2:
                continue
            i, j = int(parts[0]), int(parts[1])
            if i == j:
                logging.warning("Skipping self-edge %d %d", i, j)
                continue
            a, b = (i, j) if i < j else (j, i)
            n = max(n, a, b)
            edges_set.add((a, b))
    edges = sorted(list(edges_set))
    return edges, n + 1


def get_coloured_edges(n, edges, colouring="optimal", layers=1, max_blocks=np.inf):
    """
    Given a graph, return a list where all parallel edges are grouped together.
    :param n: Number of nodes
    :param edges: Tuple of edges
    :param colouring:
    :param layers:
    :param max_blocks:
    :return:
    """
    ansatz_edges = []
    for _ in range(layers):
        if colouring == "naive":
            for (i, j) in edges:
                if len(ansatz_edges) < max_blocks:
                    ansatz_edges.append((i, j))
        elif colouring == "optimal":
            G = nx.Graph()
            G.add_nodes_from(range(n))
            G.add_edges_from(edges)
            L = nx.line_graph(G)
            edge_to_color = nx.coloring.greedy_color(L, strategy='smallest_last')

            color_classes = {}
            for (u, v), col in edge_to_color.items():
                color_classes.setdefault(col, []).append((u, v))

            for color in sorted(color_classes):
                for (i, j) in color_classes[color]:
                    ansatz_edges.append((i, j))

    return ansatz_edges
