from common import *
import torch
from torch import nn
from mol_graph import create_mol_graph
from time import time
from typing import *
import pandas as pd


class EncodeMolLayer(torch.nn.Module):
    """ Performs message passing in a loopy belief propagation fashion.
     As described in JT-VAE paragraph 2.2 Graph Encoder.
     """

    def __init__(self, **config):
        super().__init__()

        self._config = config

        self.T = config['num_steps']
        self.W1 = nn.Parameter(torch.empty((config['edge_hidden_dim'], config['node_features_dim'])))
        self.W2 = nn.Parameter(torch.empty((config['edge_hidden_dim'], config['edge_features_dim'])))
        self.W3 = nn.Parameter(torch.empty((config['edge_hidden_dim'], config['edge_hidden_dim'])))
        self.U1 = nn.Parameter(torch.empty((config['node_hidden_dim'], config['node_features_dim'])))
        self.U2 = nn.Parameter(torch.empty((config['node_hidden_dim'], config['edge_hidden_dim'])))

    def _update_edges(self, graphs):
        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = graphs

        num_nodes = node_features.shape[0]
        edge_hidden_dim = self._config['edge_hidden_dim']

        from_nodes, to_nodes = edges[:]

        W1_x_u = self.W1.matmul(node_features.t()).t()  # (|V|, EH,)
        W1_x_u = torch.index_select(W1_x_u, 0, from_nodes)  # (|E|, EH,)

        W2_x_uv = self.W2.matmul(edge_features.t()).t()  # (|E|, EH,)

        tmp = torch.zeros((num_nodes, edge_hidden_dim,))
        wu_hidden_sum = torch.scatter_reduce(tmp, 0, to_nodes.unsqueeze(1), edge_hidden, reduce='sum')  # (|V|, EH,)
        wu_hidden_sum = torch.index_select(wu_hidden_sum, 0, from_nodes)  # (|E|, EH,)
        W3_wu_eh_sum = self.W3.matmul(wu_hidden_sum.t()).t()  # (|E|, EH,)

        uv_hidden = torch.relu(W1_x_u + W2_x_uv + W3_wu_eh_sum)  # (|E|, EH,)
        return node_features, edge_features, edges, node_hidden, uv_hidden, batch_indices

    def _update_nodes(self, graphs):
        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = graphs

        num_nodes = node_features.shape[0]
        edge_hidden_dim = self._config['edge_hidden_dim']

        from_nodes, to_nodes = edges[:]

        U1_x_u = self.U1.matmul(node_features.t()).t()  # (|V|, NH,)
        U2_vu_hidden = self.U2.matmul(edge_hidden.t()).t()  # (|E|, NH,)

        tmp = torch.zeros((num_nodes, edge_hidden_dim,))
        U2_vu_hidden_sum = torch.scatter_reduce(tmp, 0, to_nodes.unsqueeze(1), U2_vu_hidden, reduce='sum')  # (|V|, EH,)

        u_hidden = torch.relu(U1_x_u + U2_vu_hidden_sum)
        return node_features, edge_features, edges, u_hidden, edge_hidden, batch_indices

    def _propagate(self, graphs):
        for _ in range(self.T):
            self._update_edges(graphs)
        self._update_nodes(graphs)
        return graphs

    def forward(self, graphs):  # TODO convert graphs to a dataclass ?
        self._propagate(graphs)

        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = graphs

        num_batches = (torch.max(batch_indices) + 1).item()  # TODO could be cached
        node_hidden_dim = self._config['node_hidden_dim']

        mol_repr = torch.zeros((num_batches, node_hidden_dim,))
        mol_repr = torch.scatter_reduce(mol_repr, 0, batch_indices.unsqueeze(1), node_hidden, reduce='mean')  # (B, NH,)
        return mol_repr


def _batch_smiles(smiles_list: List[str], node_hidden_dim: int, edge_hidden_dim: int):
    node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = [], [], [], [], [], []

    batch_idx = 0
    node_offset = 0
    for i, smiles in enumerate(smiles_list):
        mol_graph = create_mol_graph(smiles)  # (node_features, edge_features, edges,)

        num_nodes = mol_graph[0].shape[0]
        num_edges = mol_graph[1].shape[0]

        node_features.append(mol_graph[0])
        edge_features.append(mol_graph[1])
        edges.append(mol_graph[2] + node_offset)

        node_hidden.append(torch.zeros(size=(num_nodes, node_hidden_dim,)))
        edge_hidden.append(torch.zeros(size=(num_edges, edge_hidden_dim,)))

        batch_indices.extend([batch_idx] * num_nodes)

        node_offset += num_nodes
        batch_idx += 1

    node_features = torch.cat(node_features)
    edge_features = torch.cat(edge_features)
    edges = torch.cat(edges, dim=1)
    node_hidden = torch.cat(node_hidden)
    edge_hidden = torch.cat(edge_hidden)
    batch_indices = torch.tensor(batch_indices, dtype=torch.long)

    return node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices


def _main():
    BATCH_SIZE = 1024
    NH = 32
    EH = 32

    df = pd.read_csv(DATASET_PATH)
    smiles_list = list(df['smiles'].sample(n=BATCH_SIZE))

    graphs = _batch_smiles(smiles_list, node_hidden_dim=NH, edge_hidden_dim=EH)

    encode_mol_layer = EncodeMolLayer(node_features_dim=5, edge_features_dim=1, node_hidden_dim=NH,
                                      edge_hidden_dim=EH, num_steps=100)

    start_at = time()
    mol_repr = encode_mol_layer(graphs)
    assert mol_repr.shape == (len(smiles_list), EH,)
    elapsed_time = time() - start_at

    print(f"Mol repr: {mol_repr.shape}, Elapsed time: {elapsed_time}s")


if __name__ == '__main__':
    _main()
