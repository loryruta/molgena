from common import *
import torch
from torch import nn


def exclusive_prefix_sum(x):
    cum_sum = torch.cumsum(x, dim=0)
    return torch.cat([torch.tensor([0]), cum_sum[:-1]])


class SelectMolBond(nn.Module):
    """ Given the selected motif representation,
    tells which atoms of the input molecule should form a bond with it. """

    def __init__(self, **kwargs):
        super().__init__()

        self._max_atoms_per_motif = kwargs['max_atoms_per_motif']

        self._pick_bond_mlp = nn.Sequential(
            nn.Linear(),  # TODO
            nn.ReLU(),
        )

    def forward(self, mol_graphs, selected_motif_graphs, selected_attachments):
        # mol_graphs - batched graph data
        # selected_motif_graphs - batched graph data
        # selected_attachments - (B, MN) tensor

        _, _, _, mol_atom_hiddens, _, mol_batch_indices = mol_graphs
        _, _, _, motif_atom_hiddens, _, motif_batch_indices = selected_motif_graphs

        _, motif_batch_lengths = torch.unique_consecutive(motif_batch_indices, return_counts=True)
        mol_batch_uniques, mol_batch_lengths = torch.unique_consecutive(mol_batch_indices, return_counts=True)

        num_mol_nodes = mol_atom_hiddens.shape[0]

        selected_mol_bonds = torch.zeros((num_mol_nodes, self._max_atoms_per_motif, 4))  # (MRV, MN, 4)

        for selected_atom_idx in range(self._max_atoms_per_motif):  # Iterate over all possible motif atoms (~16)
            selected_motif_atom_mask = selected_attachments[:, selected_atom_idx]  # (B)

            # If the i-th atom is selected for the batch, get the i-th hidden vector
            selected_atom_indices = (
                    exclusive_prefix_sum(motif_batch_lengths) + selected_atom_idx
            )[selected_motif_atom_mask]  # (<B)
            selected_motif_atom_hiddens = torch.index_select(motif_atom_hiddens, 0, selected_atom_indices)  # (<B, MNH)

            # Difficult: only consider input molecules that selected a motif for which the i-th atom is selected.
            # Of those, get atom hiddens
            selected_mol_atom_mask = mol_batch_indices == mol_batch_uniques[selected_motif_atom_mask]
            selected_mol_atom_hiddens = mol_atom_hiddens[selected_mol_atom_mask]  # (<MRV, MRH)

            # Assign an index to the just retrieved molecule atoms, so to map to the respective motif atom
            mol_atom_to_motif_atom_indices = torch.zeros((len(selected_mol_atom_hiddens),), dtype=torch.int64)  # (<MRV)
            set_1 = torch.cumsum(mol_batch_lengths[selected_motif_atom_mask], dim=0) - 1
            mol_atom_to_motif_atom_indices[set_1] = 1
            mol_atom_to_motif_atom_indices = exclusive_prefix_sum(mol_atom_to_motif_atom_indices)

            # Concat molecule atoms with motif atom (replicated) and run inference on the MLP to get bonds. The MLP
            # output is a 4-dim vector indicating the type of bond with the respective molecule atom, so: SINGLE,
            # DOUBLE, TRIPLE, NONE
            tmp = torch.index_select(selected_motif_atom_hiddens, 0, mol_atom_to_motif_atom_indices)  # (<MRV, MNH)
            mlp_input = torch.cat([tmp, selected_mol_atom_hiddens], dim=1)
            mlp_output = self._pick_bond_mlp(mlp_input)  # (<MRV, 4)

            # TODO go to sleep.....
            selected_mol_bonds[:, selected_atom_idx, :] = \
                torch.index_select(mlp_output, 0, mol_batch_indices[selected_mol_atom_mask])

        return selected_mol_bonds
