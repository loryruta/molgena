from common import *
import torch
from torch import nn
import pandas as pd
from typing import *
from time import time
from tensor_graph import TensorGraph
from encode_mol import *


class SelectMolAttachment(nn.Module):
    """ Given two input molecules, say A and B, return the list of atoms of B considered better for bond creation.
    Practical use example: given the input molecule and the motif molecule, select motif atoms to form bonds.
    """

    def __init__(self, **kwargs):
        super().__init__()
        self._mol_a_dim = kwargs['mol_a_dim']
        self._mol_b_node_features_dim = kwargs['mol_b_node_features_dim']
        self._mol_b_edge_features_dim = kwargs['mol_b_edge_features_dim']
        self._mol_b_node_hidden_dim = kwargs['mol_b_node_hidden_dim']
        self._mol_b_edge_hidden_dim = kwargs['mol_b_edge_hidden_dim']

        # If MLP output is higher than this threshold, B atom is selected (part of the attachment)
        self._select_motif_atom_threshold = kwargs['select_motif_atom_threshold']

        self._mol_mpn = EncodeMolMPN(
            num_steps=8,
            node_features_dim=self._mol_b_node_features_dim,
            edge_features_dim=self._mol_b_edge_features_dim,
            node_hidden_dim=self._mol_b_node_hidden_dim,
            edge_hidden_dim=self._mol_b_edge_hidden_dim
        )

        # MLP telling whether to pick or not B atom for attachment
        self._pick_atom_mlp = nn.Sequential(
            nn.Linear(self._mol_a_dim + self._mol_b_node_hidden_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, mol_a_reprs: torch.Tensor, mol_b_graphs: TensorGraph):
        """
        :param mol_a_reprs:
            `A` molecular representations stacked into a single tensor (B, MA)
        :param mol_b_graphs:
            `B` molecular graphs batched together (see batch_tensorized_graphs)
        """

        # B = Batch size
        # AM = `A` molecule representation dim
        # BA = `B` molecule total atom count
        # BAH = `B` molecule atom hidden size

        _, _, _, b_atom_hidden, _, b_batch_indices = mol_b_graphs

        # Run message passing on B molecular graphs to build up hidden vectors (same of EncodeMol)
        self._mol_mpn(mol_b_graphs)

        # Pair B atoms with the A molecular representation, B atom and molecular representation is concatenated to
        # obtain MLP input, and then run inference
        b_atom_mol_a_repr = torch.index_select(mol_a_reprs, 0, b_batch_indices)  # (BA, AM)
        mlp_input = torch.cat([b_atom_hidden, b_atom_mol_a_repr], dim=1)  # (BA, BAH+AM)
        mlp_output = self._pick_atom_mlp(mlp_input)  # (BA, 1)
        selected_atoms = (mlp_output >= self._select_motif_atom_threshold)  # (BA, 1)

        # The return value is a 1-dim bool tensor on all batched_mol_b atoms that tells if the i-th atom should form a
        # bond to the corresponding molecule A (of the same batch)

        return selected_atoms


def _main():
    from motif_graph import MotifVocab
    from mol_graph import create_mol_graph

    batch_size = 1024
    mol_repr_dim = 256

    print("Loading motif vocabulary...")
    motif_vocab = MotifVocab.load()
    num_motifs = len(motif_vocab)
    print(f"Motif vocabulary loaded; Num motifs: {num_motifs}")

    print("Creating tensorized motif graphs...")
    motif_graphs = pd.DataFrame([create_mol_graph(smiles) for smiles in list(motif_vocab)])
    print("Done")

    select_motif_attachment = SelectMolAttachment(mol_repr_dim=mol_repr_dim, motif_node_features_dim=5,
                                                  motif_edge_features_dim=1, motif_node_hidden_dim=32,
                                                  motif_edge_hidden_dim=32, max_atoms_per_motif=16,
                                                  select_motif_atom_threshold=0.6
                                                  )

    num_params = sum(param.numel() for param in select_motif_attachment.parameters())
    print(f"Model params: {num_params}")

    # Validate max_atoms_per_motif is "high enough"
    select_motif_attachment.validate_max_atoms_per_motif(motif_graphs)

    # Inference test
    mol_reprs = torch.randn((batch_size, mol_repr_dim,))
    selected_motifs = torch.randint(0, len(motif_vocab), (batch_size,))

    start_at = time()
    selected_motif_attachments = select_motif_attachment(mol_reprs, selected_motifs, motif_graphs)
    elapsed_time = time() - start_at
    print(f"Inference was successful; "
          f"Output shape: {selected_motif_attachments.shape}, "
          f"Elapsed time: {elapsed_time:.3f}s")


if __name__ == '__main__':
    _main()
