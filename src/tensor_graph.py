import torch
import torch_geometric
from typing import *


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

    def validate(self):
        assert self.node_features is not None and len(self.node_features) == self.num_nodes()
        assert self.edge_features is not None and len(self.edge_features) == self.num_edges()
        assert self.edges is not None and len(self.edges) == self.num_edges()
        assert self.node_hiddens is None or len(self.node_hiddens) == self.num_nodes()
        assert self.edge_hiddens is None or len(self.edge_hiddens) == self.num_edges()

    def to_torch_geometric(self) -> torch_geometric.data.Data:
        return torch_geometric.data.Data(
            x=self.node_features,
            edge_index=self.edges,
            edge_attr=self.edge_features
        )  # TODO batch_indices


def batch_tensor_graphs(*graphs: TensorGraph):
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
