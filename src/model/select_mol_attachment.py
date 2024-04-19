from common import *
from typing import *
from torch import nn
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN


class SelectMolAttachment(nn.Module):
    """ Given two input molecules, say A and B, return the list of atoms of B considered better for bond creation.
    Practical use example: given the input molecule and the motif molecule, select motif atoms to form bonds.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self._mol_a_repr_dim = params['mol_a_repr_dim']
        self._mol_b_node_hidden_dim = params['mol_b_mpn']['node_hidden_dim']
        self._mol_b_edge_hidden_dim = params['mol_b_mpn']['edge_hidden_dim']

        self._mol_b_mpn = EncodeMolMPN(params['mol_b_mpn'])

        # MLP classifying whether to pick or not B atoms for attachment
        self._pick_atom_mlp = nn.Sequential(
            nn.Linear(self._mol_a_repr_dim + self._mol_b_node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, mol_a_reprs: torch.Tensor, mol_b_graphs: TensorGraph) -> torch.FloatTensor:
        """
        :param mol_a_reprs:
            `A` molecular representations stacked into a single tensor (B, MA)
        :param mol_b_graphs:
            `B` molecular graphs batched together (see batch_tensorized_graphs)

        :return:
            A (BA, 1) a float tensor over molecule B atoms.
            Values higher than a threshold (hyperparameter) are considered to be selected as candidates.
        """

        # Run message passing on B
        mol_b_graphs.create_hiddens(self._mol_b_node_hidden_dim, self._mol_b_edge_hidden_dim)
        self._mol_b_mpn(mol_b_graphs)

        # Concat A molecular representations with B's node hiddens and run classification
        b_atom_mol_a_repr = torch.index_select(mol_a_reprs, 0, mol_b_graphs.batch_indices)
        mlp_input = torch.cat([mol_b_graphs.node_hiddens, b_atom_mol_a_repr], dim=1)
        mlp_output = self._pick_atom_mlp(mlp_input).squeeze()
        return mlp_output
