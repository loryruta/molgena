from common import *
import torch
from torch import nn
from tensor_graph import TensorGraph


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

    def __init__(self, **kwargs):
        super().__init__()

        self.num_steps = kwargs['num_steps']
        self.node_features_dim = kwargs['node_features_dim']
        self.edge_features_dim = kwargs['edge_features_dim']
        self.node_hidden_dim = kwargs['node_hidden_dim']
        self.edge_hidden_dim = kwargs['edge_hidden_dim']

        self.W1 = nn.Parameter(torch.empty((self.edge_hidden_dim, self.node_features_dim)))
        self.W2 = nn.Parameter(torch.empty((self.edge_hidden_dim, self.edge_features_dim)))
        self.W3 = nn.Parameter(torch.empty((self.edge_hidden_dim, self.edge_hidden_dim)))
        self.U1 = nn.Parameter(torch.empty((self.node_hidden_dim, self.node_features_dim)))
        self.U2 = nn.Parameter(torch.empty((self.node_hidden_dim, self.edge_hidden_dim)))
        # TODO initialize parameters

    def _update_edges(self, mol_graph: TensorGraph) -> None:
        from_nodes, to_nodes = mol_graph.edges[:]

        W1_x_u = self.W1.matmul(mol_graph.node_features.t()).t()  # (|V|, EH,)
        W1_x_u = torch.index_select(W1_x_u, 0, from_nodes)  # (|E|, EH,)

        W2_x_uv = self.W2.matmul(mol_graph.edge_features.t()).t()  # (|E|, EH,)

        wu_hidden_sum = torch.zeros((mol_graph.num_nodes(), self.edge_hidden_dim,))
        wu_hidden_sum.scatter_reduce(0, to_nodes.unsqueeze(1), mol_graph.edge_hiddens, reduce='sum')  # (|V|, EH,)
        wu_hidden_sum = torch.index_select(wu_hidden_sum, 0, from_nodes)  # (|E|, EH,)
        W3_wu_eh_sum = self.W3.matmul(wu_hidden_sum.t()).t()  # (|E|, EH,)

        uv_hidden = torch.relu(W1_x_u + W2_x_uv + W3_wu_eh_sum)  # (|E|, EH,)
        mol_graph.edge_hiddens = uv_hidden

    def _update_nodes(self, mol_graph: TensorGraph) -> None:
        from_nodes, to_nodes = mol_graph.edges[:]

        U1_x_u = self.U1.matmul(mol_graph.node_features.t()).t()  # (|V|, NH,)
        U2_vu_hidden = self.U2.matmul(mol_graph.edge_hiddens.t()).t()  # (|E|, NH,)

        U2_vu_hidden_sum = torch.zeros((mol_graph.num_nodes(), self.node_hidden_dim,))  # (|V|, NH,)
        U2_vu_hidden_sum.scatter_reduce(0, to_nodes.unsqueeze(1), U2_vu_hidden, reduce='sum')  # (|V|, EH,)

        u_hidden = torch.relu(U1_x_u + U2_vu_hidden_sum)
        mol_graph.node_hiddens = u_hidden

    def forward(self, mol_graph: TensorGraph) -> None:
        for _ in range(self.num_steps):
            self._update_edges(mol_graph)
        self._update_nodes(mol_graph)
