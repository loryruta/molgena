from common import *
from utils.tensor_utils import *


class SelectAttachmentBondType(nn.Module):
    """ Given two clusters, that are attached together, we know from our construction rules that they're attached by
    only a single bond. Given pairs of node hiddens from cluster1 and cluster2, this module classifies the bond type.
    """

    def __init__(self, params: Dict[str, Any]):  # TODO use params
        super().__init__()

        self._molgraph_node_hidden_dim = params["molgraph_node_hidden_dim"]
        self._mol_repr_dim = params["mol_repr_dim"]

        self._mlp = create_mlp(
            self._molgraph_node_hidden_dim + self._molgraph_node_hidden_dim + self._mol_repr_dim,
            4,  # TODO What's the cardinality of bond types?
            params["hidden_layers"])
        # Output logits must be _unnormalized_ to be fed to F.cross_entropy (pytorch)

    def forward(self,
                cluster1_node_hiddens: torch.FloatTensor,
                cluster2_node_hiddens: torch.FloatTensor,
                target_molreprs: torch.FloatTensor) -> torch.FloatTensor:
        num_attachments = target_molreprs.shape[0]
        assert cluster1_node_hiddens.shape == (num_attachments, self._molgraph_node_hidden_dim)
        assert cluster2_node_hiddens.shape == (num_attachments, self._molgraph_node_hidden_dim)
        assert target_molreprs.shape == (num_attachments, self._mol_repr_dim)

        mlp_input = torch.cat([
            cluster1_node_hiddens,
            cluster2_node_hiddens,
            target_molreprs
        ], dim=1)
        mlp_output = self._mlp(mlp_input)
        assert mlp_output.shape == (num_attachments, 4)
        return mlp_output
