from common import *
import torch
from torch import nn
import pandas as pd
from typing import *
from tensor_graph import TensorGraph, batch_tensor_graphs
from model.encode_mol import EncodeMol
from model.select_motif_mlp import SelectMotifMlp
from model.select_attachment_clusters import SelectAttachmentClusters
from model.select_attachment_atom import SelectAttachmentAtom
from model.select_attachment_bond_type import SelectAttachmentBondType
from utils.tensor_utils import *


class Molgena(nn.Module):
    """ The main class for the Molgena model.
    It can be used in two modes:
    - Reconstruction (see training): given a partial molecule and the molecule to reconstruct the model is supposed to
        produce a molecule more similar to the input molecule. This mode is used for training.
    - Property optimization: given only a partial molecule, the model may produce a molecule with the aim of optimizing
        a chemical property (e.g. molecular weight). Such process may require more iterations.
    """

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self.mod_encode_mol = EncodeMol(params['encode_mol'])
        self.mod_encode_mgraph = EncodeMol(params['encode_mgraph'])  # Use EncodeMol to compute node_hiddens for mgraphs
        self.mod_select_motif_mlp = SelectMotifMlp(params['select_motif_mlp'])
        self.mod_select_attachment_clusters = SelectAttachmentClusters(params['select_attachment_clusters'])
        self.mod_select_attachment_cluster1_atom = SelectAttachmentAtom(params['select_attachment_cluster1_atom'])
        self.mod_select_attachment_cluster2_atom = SelectAttachmentAtom(params['select_attachment_cluster2_atom'])
        self.mod_select_attachment_bond_type = SelectAttachmentBondType(params['select_attachment_bond_type'])

    def encode(self, mol_graphs: TensorGraph) -> torch.Tensor:
        return self.mod_encode_mol(mol_graphs)  # (B, 256)

    def select_motif(self,
                     partial_mol_reprs: torch.Tensor,
                     recon_mol_reprs: torch.Tensor) -> int:
        # TODO
        return self.mod_select_motif_mlp(partial_mol_reprs, recon_mol_reprs)  # (B, 8522)

    def select_attachment_clusters(self,
                                   partial_mol_mgraph: TensorGraph,
                                   motif_features: torch.Tensor) -> List[int]:
        pass  # TODO

    def select_attachment_atom(self,
                               src_mol_graph: TensorGraph,
                               dst_mol_repr: torch.Tensor,
                               target_mol_repr: torch.Tensor) -> int:
        pass  # TODO

    def select_attachment_bond_type(self,
                                    mol_graph1: TensorGraph,
                                    a1: int,
                                    mol_graph2: TensorGraph,
                                    a2: int,
                                    target_mol_repr: torch.Tensor
                                    ):
        pass  # TODO

    def forward(self):
        raise NotImplementedError("Molgena forward() is invalid, you may inference individual modules")
