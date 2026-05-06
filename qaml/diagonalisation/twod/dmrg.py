import logging
from collections import deque
from typing import Tuple, List

import networkx as nx
import numpy as np
from tenpy import MPS, Chain, SpinHalfSite, CouplingMPOModel
from tenpy.algorithms import dmrg


class DMRGResult:
    """
    Class to store the results of the DMRG algorithm.
    """

    def __init__(self, e, psi, engine, bfs_order, e1=None):
        self.e = e
        self.e1 = e1
        self.psi = psi
        self.engine = engine
        self.bfs_order = bfs_order


def degree_stats(edges_0: List[Tuple[int, int]], N: int) -> np.ndarray:
    deg = np.zeros(N, dtype=int)
    for i, j in edges_0:
        deg[i] += 1
        deg[j] += 1
    return deg


def bfs_order(edges_0: List[Tuple[int, int]], N: int, start: int | None = None) -> List[int]:
    """Return a BFS order of nodes [0..N-1]. If disconnected, continues to remaining components.
    If start is None, pick a node of maximum degree.
    """
    adj: List[List[int]] = [[] for _ in range(N)]
    for i, j in edges_0:
        adj[i].append(j)
        adj[j].append(i)
    if start is None:
        deg = degree_stats(edges_0, N)
        start = int(np.argmax(deg))
    visited = np.zeros(N, dtype=bool)
    order: List[int] = []
    q: deque[int] = deque()
    q.append(start)
    visited[start] = True
    while q:
        u = q.popleft()
        order.append(u)
        for v in adj[u]:
            if not visited[v]:
                visited[v] = True
                q.append(v)
    for u in range(N):
        if not visited[u]:
            visited[u] = True
            order.append(u)
            for v in adj[u]:
                if not visited[v]:
                    visited[v] = True
                    order.append(v)
    return order


def reindex_edges(edges_0: List[Tuple[int, int]], order: List[int]) -> List[Tuple[int, int]]:
    inv = {old: new for new, old in enumerate(order)}
    return [(inv[i], inv[j]) if inv[i] < inv[j] else (inv[j], inv[i]) for i, j in edges_0]

def inverse_order_single_site_obs(n, order, obs):
    inv = np.empty(n, dtype=int)
    for new, old in enumerate(order):
        inv[new] = old
    obs_reordered = np.empty(n, dtype=float)
    for new in range(n):
        obs_reordered[inv[new]] = obs[new]
    return obs_reordered

def inverse_order_bitstring(order, b):
    n = len(b)
    inv = np.empty(n, dtype=int)
    for new, old in enumerate(order):
        inv[new] = old
    obs_reordered = np.empty(n, dtype=str)
    for new in range(n):
        obs_reordered[inv[new]] = b[new]
    return "".join(obs_reordered)


def inverse_pair_indices(n, order, pairs):
    inv = np.empty(n, dtype=int)
    for new, old in enumerate(order):
        inv[new] = old
    reordered_pairs = []
    for i, j in pairs:
        a, b = inv[i], inv[j]
        reordered_pairs.append([a, b])
    return reordered_pairs

def inverse_order_two_site_list(
    n: int,
    order,
    pairs,
    values,
    sort_pairs,
) -> Tuple[List[List[int]], np.ndarray]:
    """Reindex pair correlations from chain order to original labels."""
    inv = np.empty(n, dtype=int)
    for new, old in enumerate(order):
        inv[new] = old

    out_pairs: List[List[int]] = []
    out_vals = np.asarray(values, dtype=float)
    for (i, j) in pairs:
        a, b = int(inv[i]), int(inv[j])
        if sort_pairs and a > b:
            a, b = b, a
        out_pairs.append([a, b])
    return out_pairs, out_vals

class XXZGeneralGraph(CouplingMPOModel):
    """XXZ on an arbitrary graph, represented on a 1D Chain lattice.

    model_params must contain:
      - N: number of sites
      - edges: list of pairs (i, j) with 0 <= i < j < N in chain order
    optional:
      - Jxy, Delta, hz
      - conserve: 'Sz' or None
      - bc_MPS: 'finite' or 'infinite' (finite is default)
    """

    def __init__(self, model_params):
        super().__init__(model_params)

    def init_sites(self, model_params):
        conserve = model_params.get("conserve", "Sz")
        if conserve not in ("Sz", None):
            raise ValueError("conserve must be 'Sz' or None for SpinHalfSite")
        site = SpinHalfSite(conserve=conserve)
        return site

    def init_lattice(self, model_params):
        n = int(model_params["n"])  # number of sites
        bc_MPS = model_params.get("bc_MPS", "finite")
        bc = model_params.get("bc", "open")
        site = self.init_sites(model_params)
        return Chain(L=n, site=site, bc=bc, bc_MPS=bc_MPS)

    def init_terms(self, model_params):
        jxy = float(model_params.get("Jxy", 1.0))
        delta = float(model_params.get("Delta", 1.0))
        hz = float(model_params.get("hz", 0.0))
        edges = model_params["edges"]  # list of (i, j) 0-indexed in chain order
        # XY terms via S+ S- + S- S+
        for i, j in edges:
            self.add_coupling_term(0.5 * jxy, i, j, "Sp", "Sm")
            self.add_coupling_term(0.5 * jxy, i, j, "Sm", "Sp")
            self.add_coupling_term(delta, i, j, "Sz", "Sz")
        if abs(hz) > 0.0:
            for i in range(self.lat.N_sites):
                self.add_onsite_term(hz, i, "Sz")


def run_heisenberg_2d(n, edges: List[Tuple], jxy: float = 1.0, jz: float = 1.0, hz: float = 0.0,
                      conserve: str | None = None,
                      chi: int = 160, trunc_cut: float = 1e-8, max_sweeps: int = 10,
                      start_node: int | None = None,
                      seed: int = 1, mixer: bool = True, neel: bool = True, calculate_e1: bool = True,):
    """
    DMRG for the spin-1/2 XXZ Heisenberg model on a Heavy-Hex coupling graph using TeNPy.

    Hamiltonian
      H = sum_{(i,j) in E} [ Jxy (Sx_i Sx_j + Sy_i Sy_j) + Delta Sz_i Sz_j ] + hz * sum_i Sz_i

    Approach
    - Keep a 1D MPS and represent long-range couplings with a CouplingMPOModel.
    - Order the graph by BFS to reduce MPO bond dimension.
    """

    if n == 0:
        raise SystemExit("No edges found or file empty.")
    deg = degree_stats(edges, n)
    logging.info("Graph stats: n=%d, |E|=%d, deg_min=%d, deg_max=%d, deg==2: %d, deg==3: %d",
                 n, len(edges), deg.min(), deg.max(), int((deg == 2).sum()), int((deg == 3).sum()))

    order = bfs_order(edges, n, start=start_node)
    edges_reord = reindex_edges(edges, order)

    model_params = dict(n=n, edges=edges_reord, Jxy=jxy, Delta=jz, hz=hz,
                        conserve=conserve, bc_MPS="finite", bc="open")
    model = XXZGeneralGraph(model_params)

    # initial product state
    sites = model.lat.mps_sites()
    if neel:
        g = nx.Graph()
        g.add_edges_from(edges_reord)
        color = nx.algorithms.bipartite.color(g)
        to_flip = [q for q, c in color.items() if c == 0]
        product_state = ["down" if i in to_flip else "up" for i in range(n)]
    else:
        rng = np.random.default_rng(seed)
        product_state = rng.choice(["up", "down"], size=n).tolist()
    psi_init = MPS.from_product_state(sites, product_state, bc=model.lat.bc_MPS)

    dmrg_params = dict(
        mixer=mixer,  # helps escape local minima at small chi
        trunc_params=dict(chi_max=chi, svd_min=1e-10, trunc_cut=trunc_cut),
        max_sweeps=max_sweeps,
        combine=True,
    )

    logging.info("Starting DMRG with chi %s, sweeps=%d", chi, max_sweeps)
    dmrg_engine = dmrg.TwoSiteDMRGEngine(psi_init, model, dmrg_params)
    e, psi_gs = dmrg_engine.run()
    e1 = None
    if calculate_e1:
        diff = 0
        orthogonal_to = [psi_gs]
        while diff < 1e-10: # In case the ground state is degenerate
            psi_init = MPS.from_product_state(sites, product_state, bc=model.lat.bc_MPS)
            e1_engine = dmrg.TwoSiteDMRGEngine(psi_init, model, dmrg_params, orthogonal_to=orthogonal_to)
            e1, psi1 = e1_engine.run()
            diff = e1 - e
            orthogonal_to.append(psi1)

    return DMRGResult(e, psi_gs, dmrg_engine, order, e1)
