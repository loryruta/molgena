import torch
import torch_geometric
from typing import *
from utils.tensor_utils import *

if TYPE_CHECKING:
    from model.encode_mol import EncodeMol


class TensorGraph:
    """ A graph, or combination of multiple graphs (batch), tensorized for efficient parallel computation.
    Data is organized similarly to torch_geometric. For batching:
    https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
    """

    node_features: torch.FloatTensor
    edge_features: Optional[torch.FloatTensor] = None
    edges: Optional[torch.LongTensor] = None
    batch_indices: Optional[torch.LongTensor] = None
    node_hiddens: Optional[torch.FloatTensor] = None
    edge_hiddens: Optional[torch.FloatTensor] = None

    def num_nodes(self):
        return len(self.node_features)

    def is_empty(self):
        return self.num_nodes() == 0

    def has_edges(self):
        return (self.edges is not None) and (self.edges.numel() > 0)

    def num_edges(self):
        return 0 if self.edge_features is None else len(self.edge_features)

    def is_batched(self):
        return self.batch_indices is not None

    def make_batched(self):
        if self.is_batched():
            raise Exception("TensorGraph is already batched")
        self.batch_indices = cast(torch.LongTensor, torch.zeros((self.num_nodes(),), dtype=torch.long))
        return self

    def batch_size(self) -> int:
        """ Counts the number of unique batch indices.
        If batch indices isn't provided, returns 1 if the graph isn't empty. """
        if self.batch_indices is None:
            return 1 if self.num_nodes() > 0 else 0
        return len(torch.unique_consecutive(self.batch_indices))

    def batch_locations(self, batch_size: Optional[int] = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """ Gets the indices, number of nodes and node offsets, of every graph within the batch.
        If `batch_size` is given counts and offsets will always be `batch_size`. Empty batch elements are zero padded.
        """

        # TODO unit test!

        if self.batch_indices is None:
            raise Exception("batch_indices not set")

        indices, counts = torch.unique_consecutive(self.batch_indices, return_counts=True)
        offsets = exclusive_prefix_sum(counts)

        if batch_size is not None:
            padded_indices = torch.empty((batch_size,), dtype=torch.int64).fill_(-1)
            padded_indices[indices] = indices

            padded_counts = torch.zeros((batch_size,), dtype=torch.int64)
            padded_counts.scatter_(0, indices, counts)

            padded_offsets = torch.zeros((batch_size,), dtype=torch.int64)
            padded_offsets.scatter_(0, indices, offsets)

            return padded_indices, padded_counts, padded_offsets

        return indices, counts, offsets

    def check_tightly_packed_batch(self):
        """ Checks that batch elements are tightly packed (sequential with no missing batch index). """
        assert self.is_batched()
        values = torch.unique_consecutive(self.batch_indices)
        return min(values) == 0 and (len(values) - 1 == max(values))

    def _validate_bidirectional_edges(self):
        """ Validates that both directions of the same connection, are put in consecutive order.
        This is done to ease retrieving the opposite direction of an edge; e.g. in EncodeMolMPN.
        """
        assert self.edges.shape[1] % 2 == 0
        from_nodes, to_nodes = self.edges

        indices = torch.arange(0, self.num_edges())
        swap_indices = indices + ((indices + 1) % 2) * 2 - 1
        assert (cast(torch.BoolTensor, from_nodes == to_nodes[swap_indices])).all()

    def validate(self):
        """ Validates that TensorGraph's values are correct. """
        assert (self.node_features is not None) and self.node_features.shape[0] == self.num_nodes()
        if self.edges is not None:
            assert self.edge_features is not None
            assert self.edges.shape == (2, self.edge_features.shape[0])
            # For our use-cases, edges are always bidirectional (i.e. in mol_graph and mgraphs)
            self._validate_bidirectional_edges()
        assert (self.batch_indices is None) or self.batch_indices.shape[0] == self.num_nodes()
        assert (self.node_hiddens is None) or self.node_hiddens.shape[0] == self.num_nodes()
        assert (self.edge_hiddens is None) or self.edge_hiddens.shape[0] == self.num_edges()

    def create_hiddens(self, node_hidden_dim: int, edge_hidden_dim: int) -> None:
        self.node_hiddens = cast(torch.FloatTensor, torch.zeros((self.num_nodes(), node_hidden_dim,)))
        self.edge_hiddens = cast(torch.FloatTensor, torch.zeros((self.num_edges(), edge_hidden_dim,)))

    def to_torch_geometric(self) -> torch_geometric.data.Data:
        return torch_geometric.data.Data(
            x=self.node_features,
            edge_index=self.edges,
            edge_attr=self.edge_features
        )  # TODO batch_indices

    def __str__(self):
        return (f"TensorGraph[\n"
                f"  node_features={self.node_features.shape}\n"
                f"  edge_features={self.edge_features.shape}\n"
                f"  edges={self.edges.shape}\n"
                f"  batch_indices={None if self.batch_indices is None else self.batch_indices.shape}\n"
                f"  node_hiddens={None if self.node_hiddens is None else self.node_hiddens.shape}\n"
                f"  edge_hiddens={None if self.edge_hiddens is None else self.edge_hiddens.shape}\n"
                f"]")


def batch_tensor_graphs(graphs: List[TensorGraph]):
    """ Given a list of TensorGraph, batch them together producing one TensorGraph. """

    assert len(graphs) > 0

    node_features = []
    edge_features = []
    edges = []
    batch_indices = []
    node_hiddens = []
    edge_hiddens = []

    node_offset = 0

    batched_graph: TensorGraph = TensorGraph()

    has_node_hiddens = graphs[0].node_hiddens is not None
    has_edge_hiddens = graphs[0].node_hiddens is not None

    for batch_idx, graph in enumerate(graphs):
        num_nodes = graph.num_nodes()

        node_features.append(graph.node_features)
        if graph.edge_features is not None:
            edge_features.append(graph.edge_features)
            edges.append(graph.edges + node_offset)
        batch_indices.extend([batch_idx] * num_nodes)

        # Graphs should all have hidden vectors or none of them
        assert has_node_hiddens == (graph.node_hiddens is not None)
        assert has_edge_hiddens == (graph.edge_hiddens is not None)

        if has_node_hiddens:
            node_hiddens.append(graph.node_hiddens)

        if has_edge_hiddens:
            edge_hiddens.append(graph.edge_hiddens)

        node_offset += num_nodes
        batch_idx += 1

    batched_graph.node_features = torch.cat(node_features)
    if len(edge_features) > 0:
        batched_graph.edge_features = torch.cat(edge_features)
        batched_graph.edges = torch.cat(edges, dim=1)
    batched_graph.batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    batched_graph.node_hiddens = torch.cat(node_hiddens) if has_node_hiddens else None
    batched_graph.edge_hiddens = torch.cat(node_hiddens) if has_node_hiddens else None
    return batched_graph


def find_node_orbit(graph: TensorGraph, node_idx: int, detector: 'EncodeMol') -> List[int]:
    """ Given an unbatched TensorGraph, and a node, finds the orbit containing such node (including it).
    The orbit detection works by computing node_hidden(s) with an MPN, and then checking for equal node_hidden(s).

    :param graph:
        A *unbatched* TensorGraph.
    :param node_idx:
        The node for which we want to find isomorphic nodes.
    :param detector:
        The MPN used to detect the node's orbit (e.g. EncodeMol).
    :return:
        List of node indices lying in the same orbit (including the node itself).
    """

    assert not graph.is_batched()

    batched_graph = batch_tensor_graphs([graph])  # TODO API-wise is preferable to work on graph

    with torch.no_grad():
        detector(batched_graph, 1)

    rounded_node_hiddens = torch.round(batched_graph.node_hiddens, decimals=7)
    isomorphic_nodes = \
        torch.nonzero(torch.all(rounded_node_hiddens == rounded_node_hiddens[node_idx], dim=1)).squeeze(-1).tolist()
    assert len(isomorphic_nodes) > 0  # Self should be always included
    return isomorphic_nodes


def compute_node_orbit_mask_with_precomputed_node_hiddens(
        graph: TensorGraph,
        node_indices: List[int]) -> torch.BoolTensor:
    """ Given a graph and a list of node_indices, computes node orbits.
    The graph must be batched with pre-computed node hiddens, while node_indices has one node per batch item.
    The node, is the node we want to compute the orbit for (including itself).
    Returns a mask over all nodes, where nodes on the same orbit of the given node are set to True.
    """

    assert graph.batch_indices is not None
    assert graph.node_hiddens is not None

    # Check that batch indices are sequential (none missing), and size is equal to the node_indices list length
    assert graph.batch_size() == len(node_indices)
    assert max(graph.batch_indices) == len(node_indices) - 1

    node_hiddens = torch.round(graph.node_hiddens, decimals=7)  # So we can use ==
    ref_node_hiddens = node_hiddens[node_indices, :]
    ref_node_hiddens = torch.index_select(ref_node_hiddens, 0, graph.batch_indices)
    return cast(torch.BoolTensor, torch.all(node_hiddens == ref_node_hiddens, dim=1))
