from common import *
import torch
from torch import nn
from typing import *
import networkx as nx
from mol_graph import create_mol_graph, MolGraph
from time import time


# You really want to implement a custom MessagePassing layer by yourself?
# Not a good idea: it's very hard to optimize and to do batch training. Use torch_geometric


class GNNLayer(nn.Module):
    """ Performs message passing in a loopy belief propagation fashion.
     Described in JT-VAE paragraph 2.2 Graph Encoder.

     Expects that input graph have nodes and edges already initialized with:
       - features: a torch.Tensor representing the features of the nodes
       - hidden: a torch.Tensor representing the hidden vector
     """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self._params = params

        self.W1 = nn.Parameter(torch.empty((params['edge_hidden_dim'], params['node_features_dim'])))
        self.W2 = nn.Parameter(torch.empty((params['edge_hidden_dim'], params['edge_features_dim'])))
        self.W3 = nn.Parameter(torch.empty((params['edge_hidden_dim'], params['edge_hidden_dim'])))
        self.U1 = nn.Parameter(torch.empty((params['node_hidden_dim'], params['node_features_dim'])))
        self.U2 = nn.Parameter(torch.empty((params['node_hidden_dim'], params['edge_hidden_dim'])))

    def forward(self, g: nx.Graph) -> nx.Graph:
        # Update edges
        for i, j in g.edges:
            jk_hidden_sum = torch.stack([g.edges[j, k]['hidden'] for j, k in g.edges(j)]).sum(dim=0)

            new_val = torch.relu(
                self.W1.matmul(g.nodes[i]['features']) +
                self.W2.matmul(g.edges[i, j]['features']) +
                self.W3.matmul(jk_hidden_sum))
            g.edges[i, j]['hidden'] = new_val

        # Update nodes
        for i, i_features in g.nodes(data='features'):
            ij_hidden_sum = torch.stack([
                self.U2.matmul(g.edges[i, j]['hidden'])
                for i, j in g.edges(i)
            ]).sum(dim=0)

            new_val = self.U1.matmul(i_features) + ij_hidden_sum
            g.nodes[i]['hidden'] = new_val

        return g


def _main():
    smiles = "C[C@@H](NC(=O)Nc1ccn(-c2ncccc2Cl)n1)[C@@H]1CCCO1"

    mol_graph, _ = create_mol_graph(smiles, node_hidden_dim=32, edge_hidden_dim=32)

    gnn_layer = GNNLayer({
        'edge_hidden_dim': 32,
        'node_hidden_dim': 32,
        'node_features_dim': MolGraph.ATOM_FEATURES_DIM,
        'edge_features_dim': MolGraph.BOND_FEATURES_DIM
    })
    gnn_layer.to('cuda')

    for i in mol_graph.nodes:
        mol_graph.nodes[i]['hidden'].zero_()

    for i, j in mol_graph.edges:
        mol_graph.edges[i, j]['hidden'].zero_()

    print(f"Parsed SMILES \"{smiles}\"; Nodes: {len(mol_graph.nodes)}, Edges: {len(mol_graph.edges)}")

    num_iterations = 1
    start_at = time()
    for _ in range(num_iterations):
        gnn_layer(mol_graph)
    elapsed_dt = time() - start_at
    print(f"Run {num_iterations} iterations on GNN layer; Elapsed: {elapsed_dt:.3f}s")


if __name__ == '__main__':
    _main()
