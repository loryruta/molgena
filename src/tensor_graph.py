import torch
import torch_geometric
from typing import *
from utils.tensor_utils import *


class TensorGraph:
    """ A graph, or combination of multiple graphs (batch), tensorized for efficient parallel computation.
    Data is organized similarly to torch_geometric. For batching:
    https://pytorch-geometric.readthedocs.io/en/latest/advanced/batching.html
    """

    node_features: torch.FloatTensor
    edge_features: torch.FloatTensor
    edges: torch.LongTensor
    batch_indices: Optional[torch.LongTensor] = None
    node_hiddens: Optional[torch.FloatTensor] = None
    edge_hiddens: Optional[torch.FloatTensor] = None

    def num_nodes(self):
        return len(self.node_features)

    def num_edges(self):
        return len(self.edge_features)

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

    def validate(self):
        assert self.node_features is not None and len(self.node_features) == self.num_nodes()
        assert self.edge_features is not None and len(self.edge_features) == self.num_edges()
        assert self.edges is not None and len(self.edges) == self.num_edges()
        assert self.node_hiddens is None or len(self.node_hiddens) == self.num_nodes()
        assert self.edge_hiddens is None or len(self.edge_hiddens) == self.num_edges()

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

    node_features, edge_features, edges, batch_indices, node_hiddens, edge_hiddens = [], [], [], [], [], []

    node_offset = 0

    batched_graph: TensorGraph = TensorGraph()

    has_node_hiddens = graphs[0].node_hiddens is not None
    has_edge_hiddens = graphs[0].node_hiddens is not None

    for batch_idx, graph in enumerate(graphs):
        num_nodes = graph.num_nodes()

        node_features.append(graph.node_features)
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
    batched_graph.edge_features = torch.cat(edge_features)
    batched_graph.edges = torch.cat(edges, dim=1)
    batched_graph.batch_indices = torch.tensor(batch_indices, dtype=torch.long)
    batched_graph.node_hiddens = torch.cat(node_hiddens) if has_node_hiddens else None
    batched_graph.edge_hiddens = torch.cat(node_hiddens) if has_node_hiddens else None
    return batched_graph
