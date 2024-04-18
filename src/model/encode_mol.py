from common import *
from typing import *
import torch
from torch import nn
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN


class EncodeMol(nn.Module):
    """ Encodes the input batched molecular graph, to a batch of embedding vectors. """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self._node_hidden_dim = params['node_hidden_dim']

        self._encode_mol_mpn = EncodeMolMPN(params)

    def forward(self, mol_graph: TensorGraph, batch_size: int):
        """
        :param batch_size:
            The batch size to determine the size of the output; i.e. (batch_size, node_hidden_dim).
            To support empty molecular graphs, it has to be externally supplied. Empty graphs will have their
            representation equal to zero.
        """

        self._encode_mol_mpn(mol_graph)

        batch_indices = mol_graph.batch_indices
        assert batch_indices.min().item() >= 0
        assert batch_indices.max().item() < batch_size

        mol_repr = torch.zeros((batch_size, self._node_hidden_dim,))
        mol_repr = mol_repr.index_reduce(0, batch_indices, mol_graph.node_hiddens, reduce='mean')  # (B, NH,)
        return mol_repr
