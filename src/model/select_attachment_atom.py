from common import *
import math
from typing import *
from torch import nn
import torch.nn.functional as F
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN
from utils.tensor_utils import *


class SelectAttachmentAtom(nn.Module):
    """ Given two clusters, that are attached together, we know from our construction rules that they're attached by
    only by a single bond.

    This module draws the atoms for attachment for cluster1 and cluster2 and finally classifies the attachment type.
    """

    def __init__(self, params: Dict[str, Any]):  # TODO use params
        super().__init__()

        self._node_hidden_dim = params["molgraph_node_hidden_dim"]
        self._mol_repr_dim = params["mol_repr_dim"]

        self._mlp = create_mlp(
            self._node_hidden_dim + self._mol_repr_dim + self._mol_repr_dim, 1, params["hidden_layers"])
        self._mlp.append(nn.Sigmoid())

    def forward(self,
                cluster1_node_hiddens: torch.FloatTensor,
                cluster1_batch_indices: torch.LongTensor,
                cluster2_molreprs: torch.FloatTensor,
                target_molreprs: torch.FloatTensor):
        """
        :param cluster1_node_hiddens:
            The node_hiddens for cluster1.
            May take into account the molecular structure "outside" cluster1 (e.g. in case cluster1 is the pmol).
        :param cluster1_batch_indices:
            A list of indices indicating the batch position of each cluster1_node_hiddens.
        :param cluster2_molreprs:
            The molrepr for cluster2 (the cluster to form the attachment with).
        :param target_molreprs:
            The molrepr for the target molecule (i.e. the output of EncodeMol).
        """
        assert cluster1_batch_indices.ndim == 1
        num_nodes = cluster1_node_hiddens.shape[0]
        assert cluster1_node_hiddens.shape == (num_nodes, self._node_hidden_dim)
        num_attachments = torch.unique(cluster1_batch_indices).numel()  # Batch size
        assert cluster2_molreprs.shape == (num_attachments, self._mol_repr_dim)
        assert target_molreprs.shape == (num_attachments, self._mol_repr_dim)

        mlp_input = torch.cat([
            cluster1_node_hiddens,
            torch.index_select(cluster2_molreprs, 0, cluster1_batch_indices),
            torch.index_select(target_molreprs, 0, cluster1_batch_indices)
        ], dim=1)
        mlp_output = self._mlp(mlp_input).squeeze(dim=-1)
        assert mlp_output.shape == (num_nodes,)
        return mlp_output
