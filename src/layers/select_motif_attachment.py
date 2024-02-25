from common import *
import torch
from torch import nn
import pandas as pd
from typing import *
from time import time
from encode_mol_layer import batch_tensorized_graphs, EncodeMolMPN


class SelectMotifAttachment(nn.Module):
    def __init__(self, **kwargs):
        super().__init__()
        self._mol_repr_dim = kwargs['mol_repr_dim']  # 256
        self._motif_node_features_dim = kwargs['motif_node_features_dim']  # 5
        self._motif_edge_features_dim = kwargs['motif_edge_features_dim']  # 1
        self._motif_node_hidden_dim = kwargs['motif_node_hidden_dim']  # 32
        self._motif_edge_hidden_dim = kwargs['motif_edge_hidden_dim']  # 32
        self._max_atoms_per_motif = kwargs['max_atoms_per_motif']  # 16

        self._encode_mol_mpn = EncodeMolMPN(
            num_steps=8,
            node_features_dim=self._motif_node_features_dim,
            edge_features_dim=self._motif_edge_features_dim,
            node_hidden_dim=self._motif_node_hidden_dim,
            edge_hidden_dim=self._motif_edge_hidden_dim
        )

        # MLP telling whether to pick or not a Motif's atom
        self._pick_atom_mlp = nn.Sequential(
            nn.Linear(self._motif_node_hidden_dim + self._mol_repr_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def validate_max_atoms_per_motif(self, motif_graphs: pd.DataFrame):
        """ Validates whether the supplied max_atoms_per_motif work with the given Motif vocabulary. """

        selected_motif_graphs = batch_tensorized_graphs(motif_graphs)
        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = selected_motif_graphs

        _, num_atoms = torch.unique_consecutive(batch_indices, return_counts=True)
        max_atoms = max(num_atoms)

        if self._max_atoms_per_motif < max_atoms:
            raise Exception(f"Configured max_atoms_per_node {self._max_atoms_per_motif} isn't compatible with Motif "
                            f"vocabulary (should be >={max_atoms})")
        else:
            print(f"Supplied max_atoms_per_node ({self._max_atoms_per_motif}) compatible with Motif vocabulary "
                  f"(>={max_atoms})")

    def forward(self, mol_reprs, selected_motifs, motif_graphs: pd.DataFrame):
        """
        :param mol_reprs:
            Batched input molecule representation. Shape (B, MR,)
        :param selected_motifs:
            Batched selected motif. Shape (B,)
        :param motif_graphs:
            A dataframe of tensorized motif graphs. One entry per Motif in the Motif vocabulary.
        """

        # B   = Batch size
        # MR  = Input molecule representation dim
        # MV  = Motif graphs total nodes
        # MN  = Motif max nodes (= atoms)
        # MNH = Motif node hidden dim

        selected_motif_graphs = batch_tensorized_graphs(motif_graphs.iloc[selected_motifs.cpu()])

        node_features, edge_features, edges, node_hidden, edge_hidden, batch_indices = selected_motif_graphs
        num_nodes = node_features.shape[0]
        batch_size = mol_reprs.shape[0]

        # Run message passing on selected motifs (use EncodeMolMPN)
        self._encode_mol_mpn(selected_motif_graphs)

        # Pair motif atoms with _their_ input molecule representation, concat node hidden and mol repr to obtain MLP
        # input, and run inference
        node_mol_repr = torch.index_select(mol_reprs, 0, batch_indices)  # (MV, MR)
        mlp_input = torch.cat([node_hidden, node_mol_repr, ], dim=1)  # (MV, NH+MR)
        mlp_output = self._pick_atom_mlp(mlp_input)  # (MV, 1)

        # The output is a tensor (B, MN, MNH) that is: for every batched element, for every selected Motif atom, we
        # consider a non-zero hidden vector IF such atom was taken. Note that MN (max motif atoms, e.g. ~16) could be
        # larger than actual number of atoms. In such case we pad the vector with zeros
        selected_motif_attachments = torch.zeros((
            batch_size,
            self._max_atoms_per_motif,
            self._motif_node_hidden_dim
        ))
        weighted_hidden = node_hidden * mlp_output
        for bi in range(batch_size):  # TODO slow iteration on batches (didn't find a way to avoid it)
            non_padding_hidden = weighted_hidden[batch_indices == bi]  # (<MN, MNH)
            selected_motif_attachments[bi][:non_padding_hidden.shape[0]] = non_padding_hidden
        return selected_motif_attachments  # (B, MN, MNH)


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

    select_motif_attachment = SelectMotifAttachment(mol_repr_dim=mol_repr_dim, motif_node_features_dim=5,
                                                    motif_edge_features_dim=1, motif_node_hidden_dim=32,
                                                    motif_edge_hidden_dim=32, max_atoms_per_motif=16)

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
