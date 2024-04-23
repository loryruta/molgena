from common import *
import math
from typing import *
import torch
from torch import nn
from tensor_graph import TensorGraph
from utils.tensor_utils import *


class EncodeMolMPN(nn.Module):
    """ Performs message passing in a loopy belief propagation fashion.
     As described in JT-VAE paragraph 2.2 Graph Encoder:
     https://arxiv.org/abs/1802.04364
     """

    num_steps: int
    node_features_dim: int
    edge_features_dim: int
    node_hidden_dim: int
    edge_hidden_dim: int

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self.num_steps = params['num_steps']
        self.node_features_dim = params['node_features_dim']
        self.edge_features_dim = params['edge_features_dim']
        self.node_hidden_dim = params['node_hidden_dim']
        self.edge_hidden_dim = params['edge_hidden_dim']

        self.W1 = create_mlp(self.node_features_dim, self.edge_hidden_dim, params["W1"])
        self.W2 = create_mlp(self.edge_features_dim, self.edge_hidden_dim, params["W2"])
        self.W3 = create_mlp(self.edge_hidden_dim, self.edge_hidden_dim, params["W3"])
        self.U1 = create_mlp(self.node_features_dim, self.node_hidden_dim, params["U1"])
        self.U2 = create_mlp(self.edge_hidden_dim, self.node_hidden_dim, params["U2"])

    def forward(self, mol_graph: TensorGraph) -> None:
        # Reference:
        # https://arxiv.org/abs/1603.05629 (Discriminative Embeddings of Latent Variable Models for Structured Data)
        # https://arxiv.org/abs/1802.04364 (Junction Tree Variational Autoencoder for Molecular Graph Generation)

        assert mol_graph.node_hiddens is not None
        assert mol_graph.edge_hiddens is not None

        has_edges = mol_graph.edges.numel() > 0

        from_nodes: torch.Tensor
        to_nodes: torch.Tensor

        if has_edges:
            from_nodes, to_nodes = mol_graph.edges[:]

            W1_x_u = self.W1(mol_graph.node_features)  # (|V|, EH,)
            W1_x_u = torch.index_select(W1_x_u, 0, from_nodes)  # (|E|, EH,)

            W2_x_uv = self.W2(mol_graph.edge_features)  # (|E|, EH,)

            # Update edges
            for t in range(self.num_steps):
                # Sum edge_hiddens at every from node
                wu_hidden_sum = torch.zeros((mol_graph.num_nodes(), self.edge_hidden_dim))
                wu_hidden_sum = torch.index_add(wu_hidden_sum, 0, from_nodes, mol_graph.edge_hiddens)  # (|V|, EH,)
                wu_hidden_sum = torch.index_select(wu_hidden_sum, 0, from_nodes)  # (|E|, EH,)

                # Assumption: every edge has a backlink which is in its previous/next position in edge list
                backlink_indices = torch.arange(0, mol_graph.num_edges())
                backlink_indices = backlink_indices + ((backlink_indices + 1) % 2) * 2 - 1
                wu_hidden_sum -= torch.index_select(mol_graph.edge_hiddens, 0, backlink_indices)

                W3_wu_eh_sum = self.W3(wu_hidden_sum)  # (|E|, EH,)

                uv_hidden = torch.relu(W1_x_u + W2_x_uv + W3_wu_eh_sum)  # (|E|, EH,)
                mol_graph.edge_hiddens = uv_hidden

        # Update nodes
        U1_x_u = self.U1(mol_graph.node_features)  # (|V|, NH,)
        U2_vu_hidden = self.U2(mol_graph.edge_hiddens)  # (|E|, NH,)

        U2_vu_hidden_sum = torch.zeros((mol_graph.num_nodes(), self.node_hidden_dim,))  # (|V|, NH,)
        if has_edges:
            to_nodes = mol_graph.edges[1]
            U2_vu_hidden_sum = U2_vu_hidden_sum.index_add(0, to_nodes, U2_vu_hidden)  # (|V|, EH,)

        u_hidden = torch.relu(U1_x_u + U2_vu_hidden_sum)
        mol_graph.node_hiddens = u_hidden
