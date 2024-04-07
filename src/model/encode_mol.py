from common import *
import torch
from torch import nn
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN


class EncodeMol(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()

        self._node_hidden_dim = kwargs['node_hidden_dim']

        self._encode_mol_mpn = EncodeMolMPN(**kwargs)

    def forward(self, mol_graph: TensorGraph):
        self._encode_mol_mpn(mol_graph)

        batch_indices = mol_graph.batch_indices

        mol_repr = torch.zeros((mol_graph.batch_size(), self._node_hidden_dim,))
        mol_repr.scatter_reduce(0, batch_indices.unsqueeze(1), mol_graph.node_hiddens, reduce='mean')  # (B, NH,)
        return mol_repr
