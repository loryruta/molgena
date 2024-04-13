from common import *
from torch import nn
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN
from model.encode_mol import EncodeMol


class SelectMolAttachment(nn.Module):
    """ Given two input molecules, say A and B, return the list of atoms of B considered better for bond creation.
    Practical use example: given the input molecule and the motif molecule, select motif atoms to form bonds.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self._num_mpn_steps = kwargs['num_mpn_steps']
        self._mol_a_repr_dim = kwargs['mol_a_repr_dim']
        self._mol_b_node_features_dim = kwargs['mol_b_node_features_dim']
        self._mol_b_edge_features_dim = kwargs['mol_b_edge_features_dim']
        self._mol_b_node_hidden_dim = kwargs['mol_b_node_hidden_dim']
        self._mol_b_edge_hidden_dim = kwargs['mol_b_edge_hidden_dim']

        self._encode_mol_mpn = EncodeMolMPN(
            num_steps=self._num_mpn_steps,
            node_features_dim=self._mol_b_node_features_dim,
            edge_features_dim=self._mol_b_edge_features_dim,
            node_hidden_dim=self._mol_b_node_hidden_dim,
            edge_hidden_dim=self._mol_b_edge_hidden_dim
        )

        # MLP telling whether to pick or not B atom for attachment
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

        # B = Batch size
        # AM = `A` molecule representation dim
        # BA = `B` molecule total atom count
        # BAH = `B` molecule atom hidden size

        b_atom_hiddens = mol_b_graphs.node_hiddens
        b_batch_indices = mol_b_graphs.batch_indices

        # Run message passing on B molecular graphs to build up hidden vectors (same of EncodeMol)
        self._encode_mol_mpn(mol_b_graphs)

        # Pair B atoms with the A molecular representation, B atom and molecular representation is concatenated to
        # obtain MLP input, and then run inference
        b_atom_mol_a_repr = torch.index_select(mol_a_reprs, 0, b_batch_indices)  # (BA, AM)
        mlp_input = torch.cat([b_atom_hiddens, b_atom_mol_a_repr], dim=1)  # (BA, BAH+AM)
        mlp_output = self._pick_atom_mlp(mlp_input).squeeze()  # (BA,)
        # selected_atoms = (mlp_output >= self._select_motif_atom_threshold)  # (BA, 1)

        # The return value is a 1-dim float tensor, representing a distribution, over all batched_mol_b atoms.
        # Tells the probability for the i-th atom to be selected as a candidate.
        # Trained with cross-entropy, on inference select the i-th atom as a candidate only if its value is higher than
        # a threshold! TODO review comment
        return mlp_output

