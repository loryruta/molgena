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
                cluster1_molgraphs: TensorGraph,
                cluster2_mol_reprs: torch.FloatTensor,
                attachment_tmol_reprs: torch.FloatTensor):
        """
        :param cluster1_molgraphs:
            The molgraph for cluster1.
        :param cluster2_mol_reprs:
            The molrepr for cluster2 (the cluster to form the attachment with).
        :param attachment_tmol_reprs:
            The molrepr for the target molecule (i.e. the output of EncodeMol).
        """
        assert cluster1_molgraphs.check_tightly_packed_batch()
        assert cluster1_molgraphs.node_hiddens is not None, "cluster1_molgraphs must have pre-computed node_hiddens"
        num_attachments = cluster1_molgraphs.batch_size()
        assert cluster2_mol_reprs.shape == (num_attachments, self._mol_repr_dim)
        assert attachment_tmol_reprs.shape == (num_attachments, self._mol_repr_dim)

        mlp_input = torch.cat([
            cluster1_molgraphs.node_hiddens,
            torch.index_select(cluster2_mol_reprs, 0, cluster1_molgraphs.batch_indices),
            torch.index_select(attachment_tmol_reprs, 0, cluster1_molgraphs.batch_indices)
        ], dim=1)
        mlp_output = self._mlp(mlp_input).squeeze()
        assert mlp_output.shape == (cluster1_molgraphs.num_nodes(),)
        return mlp_output
