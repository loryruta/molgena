from common import *
import torch
from torch import nn
import pandas as pd
from typing import *
from tensor_graph import TensorGraph, batch_tensor_graphs
from encode_mol import EncodeMol
from select_motif_mlp import SelectMotifMlp
from select_mol_attachment import SelectMolAttachment
from classify_mol_bond import ClassifyMolBond
from utils.tensor_utils import *


class Molgena(nn.Module):
    """ The main class for the Molgena model.
    It can be used in two modes:
    - Reconstruction (see training): given a partial molecule and the molecule to reconstruct the model is supposed to
        produce a molecule more similar to the input molecule. This mode is used for training.
    - Property optimization: given only a partial molecule, the model may produce a molecule with the aim of optimizing
        a chemical property (e.g. molecular weight). Such process may require more iterations.
    """

    def __init__(self, **kwargs):
        super().__init__()

        self._reconstruction_mode = kwargs['reconstruction_mode'] if 'reconstruction_mode' in kwargs else False

        self._encode_mol = EncodeMol(  # TODO EncodeMol is pre-trained and should be externally provided
            num_steps=100,
            node_features_dim=5,  # TODO constant
            edge_features_dim=1,  # TODO constant
            node_hidden_dim=32,
            edge_hidden_dim=32
        )

        self._select_motif = SelectMotifMlp(
            mol_repr_dim=256,
            num_motifs=8522,
            reconstruction_mode=self._reconstruction_mode
        )

        self._select_mol_attachment = SelectMolAttachment(
            mol_a_dim=256,
            mol_b_node_features_dim=5,  # TODO constant
            mol_b_edge_features_dim=1,  # TODO constant
            mol_b_node_hidden_dim=32,
            mol_b_edge_hidden_dim=32,
            select_motif_atom_threshold=0.5
        )

        self._classify_mol_bond = ClassifyMolBond(  # TODO Make externally configurable
        )

    def forward(self, partial_mol_graphs: TensorGraph, motif_df: pd.DataFrame,
                recon_mol_graphs: Optional[TensorGraph] = None):
        """
        :param partial_mol_graphs:
            Molecular graphs of partial molecules.
        :param motif_df:
            A dataframe of TensorGraph for all vocabulary motifs.
        :param recon_mol_graphs:
            Molecular graphs of reconstructed molecules (ground truth). Only required when reconstructing.
        """

        assert self._reconstruction_mode and recon_mol_graphs is not None

        # B = batch size
        # MF_N = total number of motif nodes (all batches)
        # MF_E = total number of motif edges (all batches)
        # MO_N = total number of input molecule nodes (all batches)
        # MO_E = total number of input molecule edges (all batches)

        # Encoding
        partial_mol_reprs = self._encode_mol(partial_mol_graphs)  # (B, 256)

        recon_mol_reprs = None
        if self._reconstruction_mode:
            recon_mol_reprs = self._encode_mol(recon_mol_graphs)

        # Decoding
        selected_motif_distr = self._select_motif(partial_mol_reprs, recon_mol_reprs)  # (B, 8522)
        selected_motif_indices = torch.argmax(selected_motif_distr, dim=1)  # (B, 1)

        selected_motif_graphs = batch_tensor_graphs(  # (MF_N, MF_E, MF_E, MF_N, MF_E, MF_N)
            motif_df[selected_motif_indices.squeeze().cpu()]
        )

        motif_reprs = self._encode_mol(selected_motif_graphs)  # (B, 256)

        selected_motif_attachments = self._select_mol_attachment(partial_mol_reprs, selected_motif_graphs)  # (MF_N, 1)
        selected_mol_attachments = self._select_mol_attachment(motif_reprs, partial_mol_graphs)  # (MO_N, 1)

        selected_mol_atom_indices = torch.nonzero(selected_mol_attachments)
        selected_motif_atom_indices = torch.nonzero(selected_motif_attachments)
        proposed_bonds = torch.cartesian_prod(selected_mol_atom_indices, selected_motif_atom_indices)
        classified_bonds = self._classify_mol_bond(partial_mol_graphs, selected_motif_graphs, proposed_bonds)

        return selected_motif_distr, classified_bonds


def _main():
    molgena = Molgena()
    print(f"Model params: {num_model_params(molgena)}")


if __name__ == "__main__":
    _main()

