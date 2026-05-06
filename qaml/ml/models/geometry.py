import numpy as np
import torch
from torch import nn


# from netket.graph import Graph

# Wrapped mapping from parameters to lattice in class
# Slightly adapted version of code by lllewis

# object we use in the deep learning model
class LocalLayer(nn.Module):
    def __init__(self, parameter_map) -> None:
        super().__init__()
        self.parameter_map = parameter_map
        self._cached_pm = None
        self._cached_device = None

    def forward(self, x):
        device = x.device
        if self._cached_device != device:
            self._cached_pm = [p.to(device) for p in self.parameter_map]
            self._cached_device = device
        return [x[..., params] for params in self._cached_pm]


# pauli_qubits: list of iterable of qubits each pauli acts on
class GridMap:
    def __init__(self, shape, pauli_qubits=None, delta1=0, mode="local") -> None:
        shape = tuple(shape)
        self.shape = shape
        self.d = len(shape)
        self.mode = mode
        self.n = torch.prod(torch.tensor(shape))
        self.delta1 = delta1
        # indices of qubits on n-d grid structure (coordinate-to-qubit index)
        self.indices = torch.arange(self.n).reshape(shape)
        # list of all coordinates (qubit-index-to-coordinate)
        self.coordinates = torch.from_numpy(np.array([index for index in np.ndindex(shape)]))
        self.edges = self.get_edges()
        self.m = len(self.edges)

        # lexographic order of edges
        tuple_edges = np.array([(edge[0].item(), -(edge[1].item())) for edge in self.edges],
                               dtype=np.dtype([('x', int), ('y', int)]))
        sorted_idx = np.argsort(tuple_edges, order=('x', 'y'))

        if mode == "local":
            self.edges = self.edges[sorted_idx]

        self.pauli_qubits = self.edges if pauli_qubits is None else pauli_qubits
        self.parameter_map = [self.get_local_parameters(qubits) for qubits in self.pauli_qubits]
        # print(self.parameter_map)

    def get_layer(self):
        return LocalLayer(self.parameter_map)

    # potentially generalize to different models
    def get_edges(self):
        if self.mode == "nonlocal":
            return torch.tensor([[ind2, ind1] for ind1 in range(self.n) for ind2 in range(ind1)])
        # maybe improve this, but one-time calc
        edges = []
        for ind1 in range(self.n):
            for ind2 in range(ind1):
                # adjacent, so l1 distance 1
                if self.distance(ind1, ind2) == 1:
                    edges.append(
                        (ind2, ind1))  # index 1 can be larger than index 2, want lexographic order

        return torch.tensor(edges)

    # Given two qubits q1, q2 (1-indexed integers) in length x width grid
    # Output l1 distance between q1 and q2 in grid
    def distance(self, q1, q2):
        return torch.abs(self.coordinates[q1] - self.coordinates[q2]).sum()

    def get_nearby_qubits(self, q, delta1):
        return torch.argwhere(
            (self.coordinates - self.coordinates[q].repeat(self.n, 1)).float().norm(
                dim=1) <= delta1)

    # qubits: iterable of integers
    # returns indices of all parameters (parameters are edges) for a tensor with qubit indices
    def get_local_parameters(self, qubits):
        local_qubits = []
        # print("qubits: ", qubits)
        for qubit in qubits:
            nearby_qubits = self.get_nearby_qubits(qubit, self.delta1)
            local_qubits.append(nearby_qubits)
        local_qubits = torch.cat(local_qubits).flatten().unique()
        # print(local_qubits)

        local_parameters = []
        for qubit in local_qubits:
            # print(qubit)
            for j in range(2):
                # get edges connected to that qubit
                loc_param = torch.argwhere(self.edges[:, j] == torch.ones((self.m,)) * qubit)
                # print(f"{j}: {loc_param}")
                # print(f"{j} len(loc_param): {len(loc_param)}")
                if len(loc_param) > 0:
                    local_parameters.append(loc_param.flatten())
        # print("local_parameters: ", local_parameters)
        return torch.cat(local_parameters).flatten().unique()


"""    def get_graph(self, all_unique_colors=True):
        edges = self.edges.tolist()
        if all_unique_colors:
            edges = [(edge[0], edge[1], i) for i, edge in enumerate(edges)]
        graph = Graph(edges)

        return graph"""


### LEGACY ###

def lllewis234_get_local():
    length = 4
    width = 5
    grid = 0
    # generate all edges in grid in same order as Xfull
    all_edges = []
    for i in range(0, length):
        for j in range(1, width + 1):
            if i != length - 1:
                all_edges.append((width * i + j, width * (i + 1) + j))
            if j != width:
                all_edges.append((width * i + j, width * i + j + 1))
    print(all_edges)

    def calc_distance(q1, q2):
        # Given two qubits q1, q2 (1-indexed integers) in length x width grid
        # Output l1 distance between q1 and q2 in grid

        pos1 = np.array(np.where(grid == q1)).T[0]
        pos2 = np.array(np.where(grid == q2)).T[0]

        return np.abs(pos1[0] - pos2[0]) + np.abs(pos1[1] - pos2[1])

    def get_nearby_qubit_pairs(d):
        # Given distance d > 0
        # Output all pairs of qubits that are within distance d of each other

        if d == 1:
            return all_edges

        qubit_pairs = []
        for q1 in range(1, length * width + 1):
            for q2 in range(1, length * width + 1):
                dist = calc_distance(q1, q2)
                pair = tuple(sorted((q1, q2)))
                if dist == d and pair not in qubit_pairs:
                    qubit_pairs.append(pair)

        return qubit_pairs

    # Finding local patches of a given radius

    def get_local_region_qubits(q, delta1):
        # Given a qubit q (1-indexed integer) in length x width grid and radius delta1
        # delta1 = -1 if all qubits are in local region
        # Output list of qubits (1-indexed integers) within a radius of delta1 of q

        if delta1 == 0:
            return [q]
        elif delta1 == -1:
            return list(range(1, length * width + 1))

        local_qubits = []
        for q2 in range(1, length * width + 1):
            dist = calc_distance(q, q2)

            if dist <= delta1:
                local_qubits.append(q2)

        return local_qubits

    def get_local_region_edges(q1, q2, delta1):
        # Given two qubits q1, q2 (1-indexed integers) in length x width grid and radius delta1
        # delta1 = -1 if all qubits are in local region
        # Output list of tuples of qubits (1-indexed integers) corresponding to edges in local region of radius delta1

        if delta1 == 0:
            return [(q1, q2)]
        elif delta1 == -1:
            return all_edges

        local_qubits = list(
            set(get_local_region_qubits(q1, delta1) + get_local_region_qubits(q2, delta1)))

        local_edges = []
        for edge in all_edges:
            (q1, q2) = edge
            if q1 in local_qubits and q2 in local_qubits:
                local_edges.append(edge)

        return local_edges

    def get_local_region_params(q1, q2, delta1, data, i):
        # Given two qubits q1, q2 (1-indexed integers) in length x width grid, radius delta1, and input data (i.e., Xfull)
        # delta1 = -1 if all qubits are considered nearby
        # Output data but only for parameters corresponding to edges within radius delta1

        edges = get_local_region_edges(q1, q2, delta1)

        indices = [all_edges.index(edge) for edge in edges]

        return np.array([data[i][j] for j in sorted(indices)])


# Heavy-Hex Lattice Support using NetworkX
class HeavyHexGridMap:
    """
    GridMap implementation for heavy-hex lattice topology using NetworkX.

    This class constructs a heavy-hex lattice and computes all pairwise shortest
    path distances between qubits using NetworkX graph algorithms. It maintains
    compatibility with CombinedFullDNN's edge-based input architecture.

    Parameters:
    -----------
    distance : int
        Heavy-hex lattice distance parameter (e.g., 5, 7)
    pauli_qubits : list, optional
        List of qubit pairs for Pauli operators. Defaults to all edges.
    delta1 : int or float
        Radius for local parameter neighborhoods.
        - If int: graph distance (number of hops)
        - If float: Euclidean distance in 2D coordinate space
    mode : str
        "local" for nearest-neighbor edges, "nonlocal" for all pairs
    use_qiskit : bool
        If True, use Qiskit's heavy-hex generation. If False, use custom edges.
    custom_edges : list of tuples, optional
        Custom edge list [(i,j), ...] for arbitrary topologies
    custom_coordinates : dict, optional
        Custom coordinate mapping {qubit_idx: (x, y)}
    """
    def __init__(self, distance=None, pauli_qubits=None, delta1=0, mode="local",
                 use_qiskit=True, custom_edges=None, custom_coordinates=None):
        import networkx as nx

        self.mode = mode
        self.delta1 = delta1
        self.d = 2  # Heavy-hex is 2D

        # Generate or load heavy-hex topology first
        if use_qiskit and distance is not None:
            self._init_from_qiskit(distance)
        elif custom_edges is not None:
            self._init_from_custom(custom_edges, custom_coordinates)
        else:
            raise ValueError("Must provide either distance (with use_qiskit=True) or custom_edges")

        # Build NetworkX graph BEFORE any coordinate fallback can use it
        self.G = nx.Graph()
        self.G.add_nodes_from(range(self.n))
        self.G.add_edges_from(self.edges.tolist())

        # Load coordinates if they were not already set by _init_from_qiskit/_init_from_custom
        if not hasattr(self, "coordinates") or self.coordinates is None:
            self._load_heavy_hex_coordinates(distance)

        # Compute all-pairs shortest path distances
        self._compute_distance_matrix()

        # Sort edges lexicographically
        tuple_edges = np.array(
            [(edge[0].item(), -(edge[1].item())) for edge in self.edges],
            dtype=np.dtype([("x", int), ("y", int)])
        )
        sorted_idx = np.argsort(tuple_edges, order=("x", "y"))

        if mode == "local":
            self.edges = self.edges[sorted_idx]

        # Set up parameter mapping for neural network
        self.pauli_qubits = self.edges if pauli_qubits is None else pauli_qubits
        self.parameter_map = [self.get_local_parameters(qubits) for qubits in self.pauli_qubits]

    def _init_from_qiskit(self, distance):
        """Initialize heavy-hex lattice using Qiskit"""
        try:
            from qiskit.transpiler import CouplingMap
        except ImportError:
            raise ImportError(
                "Qiskit required for heavy-hex generation. Install with: pip install qiskit"
            )

        cmap = CouplingMap.from_heavy_hex(distance)
        self.n = cmap.size()

        edges = sorted({tuple(sorted((i, j))) for i, j in cmap.get_edges()})
        self.edges = torch.tensor(edges, dtype=torch.long)
        self.m = len(self.edges)

        # Do NOT call _load_heavy_hex_coordinates here.
        # Coordinates are loaded after self.G is created in __init__.

    def _load_heavy_hex_coordinates(self, distance):
        """Load pre-computed 2D coordinates for heavy-hex lattice"""
        try:
            from graph import heavy_hex_2d_coords

            if distance == 5:
                coords_dict = heavy_hex_2d_coords.d5_2d_coords
            elif distance == 7:
                coords_dict = heavy_hex_2d_coords.d7_2d_coords(
                    nodes_1_indexed=True,
                    edges_tuples=True
                )
            else:
                # Fallback to NetworkX layout
                import networkx as nx
                pos = nx.spring_layout(self.G, seed=42)
                self.coordinates = torch.tensor(
                    [pos[i] for i in range(self.n)],
                    dtype=torch.float32
                )
                return

            self.coordinates = torch.zeros((self.n, 2), dtype=torch.float32)
            for idx_1based, (x, y) in coords_dict.items():
                idx_0based = idx_1based - 1
                if 0 <= idx_0based < self.n:
                    self.coordinates[idx_0based] = torch.tensor([x, y], dtype=torch.float32)

        except (ImportError, AttributeError):
            # Fallback to NetworkX layout
            import networkx as nx
            pos = nx.spring_layout(self.G, seed=42)
            self.coordinates = torch.tensor(
                [pos[i] for i in range(self.n)],
                dtype=torch.float32
            )

    def _compute_distance_matrix(self):
        """
        Compute all-pairs shortest path distances using NetworkX.
        Stores both graph distances (hops) and Euclidean distances.
        """
        import networkx as nx

        # Compute graph distances (shortest path lengths)
        self.graph_distances = torch.full((self.n, self.n), float('inf'), dtype=torch.float32)

        # Use NetworkX to compute all shortest paths
        all_paths = dict(nx.all_pairs_shortest_path_length(self.G))

        for i in range(self.n):
            if i in all_paths:
                for j, dist in all_paths[i].items():
                    self.graph_distances[i, j] = dist

        # Compute Euclidean distances from coordinates
        if hasattr(self, 'coordinates'):
            self.euclidean_distances = torch.cdist(self.coordinates, self.coordinates, p=2)
        else:
            self.euclidean_distances = self.graph_distances.clone()

    def distance(self, q1, q2):
        """
        Compute distance between two qubits.
        Uses graph distance (shortest path) if delta1 is int,
        Euclidean distance if delta1 is float.
        """
        if isinstance(self.delta1, int):
            # Use graph distance (number of hops)
            return self.graph_distances[q1, q2].item()
        else:
            # Use Euclidean distance
            return self.euclidean_distances[q1, q2].item()

    def get_nearby_qubits(self, q, delta1):
        """
        Get all qubits within delta1 distance of qubit q.

        Parameters:
        -----------
        q : int
            Qubit index
        delta1 : int or float
            Distance threshold

        Returns:
        --------
        torch.Tensor : Indices of nearby qubits
        """
        if delta1 == 0:
            return torch.tensor([q], dtype=torch.long)

        if isinstance(delta1, int):
            # Use graph distance
            distances = self.graph_distances[q]
        else:
            # Use Euclidean distance
            distances = self.euclidean_distances[q]

        nearby = torch.argwhere(distances <= delta1).flatten()
        return nearby

    def get_local_parameters(self, qubits):
        """
        Get indices of all edge parameters (edges) within delta1 distance
        of the given qubits.

        This method identifies which edges should be included in the local
        receptive field for a neural network predicting observables on the
        given qubits.

        Parameters:
        -----------
        qubits : iterable
            Qubit indices to find local parameters for

        Returns:
        --------
        torch.Tensor : Indices of edges (parameters) in the local region
        """
        local_qubits = []

        # Find all qubits within delta1 of any input qubit
        for qubit in qubits:
            nearby_qubits = self.get_nearby_qubits(qubit, self.delta1)
            local_qubits.append(nearby_qubits)

        local_qubits = torch.cat(local_qubits).flatten().unique()

        # Find all edges connected to these local qubits
        local_parameters = []
        for qubit in local_qubits:
            for j in range(2):  # Check both positions in edge tuple
                # Get edges where this qubit appears in position j
                loc_param = torch.argwhere(self.edges[:, j] == qubit)
                if len(loc_param) > 0:
                    local_parameters.append(loc_param.flatten())

        if len(local_parameters) == 0:
            return torch.tensor([], dtype=torch.long)

        return torch.cat(local_parameters).flatten().unique()

    def get_layer(self):
        """Return LocalLayer for use in neural network"""
        return LocalLayer(self.parameter_map)

    def get_edges(self):
        """Return edge list (for compatibility)"""
        return self.edges

    def get_edge_features(self, include_distances=True):
        """
        Generate edge features incorporating graph distances.

        This method creates feature vectors for each edge that include:
        - Edge type information (which qubits are connected)
        - Graph distance between the qubits
        - Optionally: Euclidean distance

        Parameters:
        -----------
        include_distances : bool
            Whether to include distance information in features

        Returns:
        --------
        dict : Dictionary with edge features and metadata
        """
        edge_features = {
            'edges': self.edges,
            'n_edges': self.m,
            'n_qubits': self.n,
        }

        if include_distances:
            # Add graph distances for each edge
            edge_graph_dists = torch.zeros(self.m, dtype=torch.float32)
            edge_euclidean_dists = torch.zeros(self.m, dtype=torch.float32)

            for idx, (i, j) in enumerate(self.edges):
                edge_graph_dists[idx] = self.graph_distances[i, j]
                edge_euclidean_dists[idx] = self.euclidean_distances[i, j]

            edge_features['graph_distances'] = edge_graph_dists
            edge_features['euclidean_distances'] = edge_euclidean_dists

        return edge_features

    def get_distance_matrix(self, distance_type='graph'):
        """
        Get the full distance matrix.

        Parameters:
        -----------
        distance_type : str
            'graph' for shortest path distances, 'euclidean' for spatial distances

        Returns:
        --------
        torch.Tensor : n x n distance matrix
        """
        if distance_type == 'graph':
            return self.graph_distances
        elif distance_type == 'euclidean':
            return self.euclidean_distances
        else:
            raise ValueError(f"Unknown distance_type: {distance_type}")

    def visualize_topology(self, save_path=None):
        """
        Visualize the heavy-hex topology with NetworkX.

        Parameters:
        -----------
        save_path : str, optional
            Path to save the visualization
        """
        try:
            import matplotlib.pyplot as plt
            import networkx as nx

            plt.figure(figsize=(12, 10))

            # Convert coordinates to dict for NetworkX
            pos = {i: self.coordinates[i].numpy() for i in range(self.n)}

            # Draw the graph
            nx.draw(self.G, pos,
                    node_color='lightblue',
                    node_size=500,
                    with_labels=True,
                    font_size=8,
                    edge_color='gray',
                    width=2)

            plt.title(f"Heavy-Hex Lattice ({self.n} qubits, {self.m} edges)")
            plt.axis('equal')

            if save_path:
                plt.savefig(save_path, dpi=300, bbox_inches='tight')
            else:
                plt.show()

        except ImportError:
            print("Matplotlib required for visualization. Install with: pip install matplotlib")
