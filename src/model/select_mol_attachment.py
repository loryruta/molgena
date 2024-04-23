from common import *
from typing import *
from torch import nn
from tensor_graph import TensorGraph
from model.encode_mol_mpn import EncodeMolMPN
from utils.tensor_utils import *


class SelectMolAttachment(nn.Module):
    """ Given two input molecules, say A and B, return the list of atoms of B considered better for bond creation.
    Practical use example: given the input molecule and the motif molecule, select motif atoms to form bonds.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self._mol_a_repr_dim = params['mol_a_repr_dim']
        self._mol_b_node_hidden_dim = params['mol_b_mpn']['node_hidden_dim']
        self._mol_b_edge_hidden_dim = params['mol_b_mpn']['edge_hidden_dim']
        self._rnn_iterations = params['rnn_iterations']
        self._rnn_hidden_size = params['rnn_hidden_size']

        self._mol_b_mpn = EncodeMolMPN(params['mol_b_mpn'])

        self._max_atoms_per_batch = 50

        # MLP classifying whether to pick or not B atoms for attachment
        self._pick_atom_mlp = create_mlp(
            self._mol_a_repr_dim + self._rnn_hidden_size,
            1,
            params['pick_atom_mlp']['hidden_layers'],
            non_linearity_func=nn.ReLU()
        )
        self._pick_atom_mlp.append(nn.Sigmoid())
        self._rnn = nn.GRUCell(self._mol_b_node_hidden_dim, self._rnn_hidden_size)

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

        batch_size = mol_a_reprs.shape[0]
        node_hidden_dim = self._mol_b_mpn.node_hidden_dim

        batch_indices, batch_counts, batch_offsets = mol_b_graphs.batch_locations(batch_size)
        max_batch_count = batch_counts.max().item()
        if max_batch_count > self._max_atoms_per_batch:
            raise Exception(
                f"A graph of `mol_graphs` has more nodes than `_max_atoms_per_batch` "
                f"({max_batch_count} > {self._max_atoms_per_batch})")

        # Run message passing on B
        mol_b_graphs.create_hiddens(node_hidden_dim, None)
        self._mol_b_mpn(mol_b_graphs)

        rnn_hiddens = torch.zeros((batch_size, self._rnn_hidden_size))
        output = torch.zeros((mol_b_graphs.num_nodes(),))

        # Sequentially visit all atoms per batch element for _rnn_iterations,
        # at the final iteration, use RNN hidden states to classify whether an atom is a candidate or not
        for t in range(self._rnn_iterations + 1):
            for i in range(self._max_atoms_per_batch):
                active_mask = i < batch_counts
                if not active_mask.any():
                    break

                active_indices = batch_indices[active_mask]
                active_offsets = batch_offsets[active_mask] + i

                cur_node_hiddens = mol_b_graphs.node_hiddens[active_offsets]  # (B, NH)
                cur_rnn_hiddens = rnn_hiddens[active_indices]

                rnn_hiddens[active_indices] = self._rnn(cur_node_hiddens, cur_rnn_hiddens)

                # Only at the end, perform classification and save results for the i-th atom
                if t == self._rnn_iterations:
                    mol_a_repr = torch.index_select(mol_a_reprs, 0, active_indices)
                    ith_input = torch.cat([
                        mol_a_repr,
                        rnn_hiddens[active_indices]
                    ], dim=1)
                    ith_output = self._pick_atom_mlp(ith_input).squeeze()
                    output[active_offsets] = ith_output

        return cast(torch.FloatTensor, output)
