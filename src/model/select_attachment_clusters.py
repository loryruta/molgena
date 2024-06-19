from common import *
import math
from typing import *
from torch import nn
import torch.nn.functional as F
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN
from utils.tensor_utils import *


class SelectAttachmentClusters(nn.Module):  # TODO SelectAttachmentCluster (singular, also filename)
    """ When the next Motif is selected, we need to know to which clusters of the partial molecule's mgraph it has to be
    attached to. After a dataset inspection, it turned out the backlink's cluster is _always_ one, which makes the task
    a lot easier considering the partial molecule mgraph could showcase different automorphisms.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self._node_hidden_dim = params["mgraph_node_hidden_dim"]
        self._motif_mrepr_dim = params["motif_mrepr_dim"]

        self._mlp = create_mlp(
            self._node_hidden_dim + self._motif_mrepr_dim, 1, params["hidden_layers"])
        self._mlp.append(nn.Sigmoid())

    def forward(self,
                pmol_mgraphs: TensorGraph,
                next_motif_mreprs: torch.Tensor) -> torch.FloatTensor:
        batch_size = next_motif_mreprs.shape[0]  # Empty/full molecules are excluded!
        assert pmol_mgraphs.check_tightly_packed_batch()
        assert pmol_mgraphs.batch_size() == batch_size
        assert pmol_mgraphs.node_hiddens is not None, "pmol_mgraphs must have pre-computed node_hiddens"

        mlp_input = torch.cat([
            pmol_mgraphs.node_hiddens,
            torch.index_select(next_motif_mreprs, 0, pmol_mgraphs.batch_indices),
            # TODO target molecule is important for choosing how to attach the motif!
        ], dim=1)
        mlp_output = self._mlp(mlp_input).squeeze(dim=-1)
        assert mlp_output.shape == (pmol_mgraphs.num_nodes(),)
        return mlp_output
