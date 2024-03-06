from common import *
import torch
from torch import nn
from time import time
from typing import *
import pandas as pd


class EncodeMolMPN(nn.Module):
    """ Performs message passing in a loopy belief propagation fashion.
     As described in JT-VAE paragraph 2.2 Graph Encoder:
     https://arxiv.org/abs/1802.04364
     """

    def __init__(self, **kwargs):
        super().__init__()

        self._num_steps = kwargs['num_steps']
        self._node_features_dim = kwargs['node_features_dim']
        self._edge_features_dim = kwargs['edge_features_dim']
        self._node_hidden_dim = kwargs['node_hidden_dim']
        self._edge_hidden_dim = kwargs['edge_hidden_dim']

        self.W1 = nn.Parameter(torch.empty((self._edge_hidden_dim, self._node_features_dim)))
        self.W2 = nn.Parameter(torch.empty((self._edge_hidden_dim, self._edge_features_dim)))
        self.W3 = nn.Parameter(torch.empty((self._edge_hidden_dim, self._edge_hidden_dim)))
        self.U1 = nn.Parameter(torch.empty((self._node_hidden_dim, self._node_features_dim)))
        self.U2 = nn.Parameter(torch.empty((self._node_hidden_dim, self._edge_hidden_dim)))

    def _update_edges(self, graphs):
        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = graphs

        num_nodes = node_features.shape[0]

        from_nodes, to_nodes = edges[:]

        W1_x_u = self.W1.matmul(node_features.t()).t()  # (|V|, EH,)
        W1_x_u = torch.index_select(W1_x_u, 0, from_nodes)  # (|E|, EH,)

        W2_x_uv = self.W2.matmul(edge_features.t()).t()  # (|E|, EH,)

        tmp = torch.zeros((num_nodes, self._edge_hidden_dim,))
        wu_hidden_sum = torch.scatter_reduce(tmp, 0, to_nodes.unsqueeze(1), edge_hidden, reduce='sum')  # (|V|, EH,)
        wu_hidden_sum = torch.index_select(wu_hidden_sum, 0, from_nodes)  # (|E|, EH,)
        W3_wu_eh_sum = self.W3.matmul(wu_hidden_sum.t()).t()  # (|E|, EH,)

        uv_hidden = torch.relu(W1_x_u + W2_x_uv + W3_wu_eh_sum)  # (|E|, EH,)
        return node_features, edge_features, edges, node_hidden, uv_hidden, batch_indices

    def _update_nodes(self, graphs):
        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = graphs

        num_nodes = node_features.shape[0]

        from_nodes, to_nodes = edges[:]

        U1_x_u = self.U1.matmul(node_features.t()).t()  # (|V|, NH,)
        U2_vu_hidden = self.U2.matmul(edge_hidden.t()).t()  # (|E|, NH,)

        tmp = torch.zeros((num_nodes, self._edge_hidden_dim,))
        U2_vu_hidden_sum = torch.scatter_reduce(tmp, 0, to_nodes.unsqueeze(1), U2_vu_hidden, reduce='sum')  # (|V|, EH,)

        u_hidden = torch.relu(U1_x_u + U2_vu_hidden_sum)
        return node_features, edge_features, edges, u_hidden, edge_hidden, batch_indices

    def forward(self, graphs):
        for _ in range(self._num_steps):
            self._update_edges(graphs)
        self._update_nodes(graphs)
        return graphs


class EncodeMol(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self._node_hidden_dim = kwargs['node_hidden_dim']

        self._encode_mol_mpn = EncodeMolMPN(**kwargs)

    def forward(self, graphs):
        self._encode_mol_mpn(graphs)

        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = graphs

        num_batches = (torch.max(batch_indices) + 1).item()  # TODO could be cached
        node_hidden_dim = self._node_hidden_dim

        mol_repr = torch.zeros((num_batches, node_hidden_dim,))
        mol_repr = torch.scatter_reduce(mol_repr, 0, batch_indices.unsqueeze(1), node_hidden, reduce='mean')  # (B, NH,)
        return mol_repr


def _main():
    BATCH_SIZE = 1024
    NH = 32
    EH = 32

    df = pd.read_csv(DATASET_PATH)
    smiles_list = list(df['smiles'].sample(n=BATCH_SIZE))

    graphs = _batch_smiles(smiles_list, node_hidden_dim=NH, edge_hidden_dim=EH)

    encode_mol = EncodeMol(node_features_dim=5, edge_features_dim=1, node_hidden_dim=NH,
                                      edge_hidden_dim=EH, num_steps=100)

    num_params = sum(param.numel() for param in encode_mol.parameters())
    print(f"Model params: {num_params}")

    start_at = time()
    mol_repr = encode_mol(graphs)
    assert mol_repr.shape == (len(smiles_list), EH,)
    elapsed_time = time() - start_at

    print(f"Mol repr: {mol_repr.shape}, Elapsed time: {elapsed_time:.3f}s")


if __name__ == '__main__':
    _main()
