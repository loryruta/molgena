from common import *
from rdkit import Chem
import torch
from torch import nn
import torch.distributions as dist
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

    def forward(self):
        raise NotImplementedError("Molgena forward() is invalid, you may inference individual modules")

    # Inference utils

    def _check_single_batch_tgraph(self, tgraph: TensorGraph):
        assert tgraph.check_tightly_packed_batch()
        assert tgraph.batch_size() == 1

    def encode(self, molgraph: TensorGraph) -> torch.Tensor:
        if not molgraph.is_empty():
            self._check_single_batch_tgraph(molgraph)
        return self.mod_encode_mol(molgraph, 1).squeeze()

    def encode_mgraph(self, mgraph: TensorGraph) -> None:  # Only to calculate node_hiddens
        self._check_single_batch_tgraph(mgraph)
        self.mod_encode_mgraph(mgraph, 1)

    def select_motif(self,
                     partial_molrepr: torch.Tensor,
                     recon_molrepr: torch.Tensor) -> int:
        # Input validation
        assert partial_molrepr.ndim == 1
        assert recon_molrepr.ndim == 1

        next_motif_distr = self.mod_select_motif_mlp(partial_molrepr.unsqueeze(dim=0), recon_molrepr.unsqueeze(dim=0))
        return torch.argmax(next_motif_distr, dim=1).item()

    def select_attachment_cluster(self, pmol_mgraph: TensorGraph, motif_mrepr: torch.Tensor) -> int:
        # Input validation
        self._check_single_batch_tgraph(pmol_mgraph)
        assert motif_mrepr.ndim == 1

        next_cluster_distr = self.mod_select_attachment_clusters(pmol_mgraph, motif_mrepr.unsqueeze(dim=0))
        next_cluster_idx = dist.Categorical(probs=next_cluster_distr).sample().item()
        return next_cluster_idx

    def select_attachment_cluster1_atom(self,
                                        cluster1_molgraph: TensorGraph,
                                        cluster2_molrepr: torch.Tensor,
                                        target_molrepr: torch.Tensor) -> int:
        # Input validation
        self._check_single_batch_tgraph(cluster1_molgraph)
        assert cluster2_molrepr.ndim == 1
        assert target_molrepr.ndim == 1
        bond_atom_distr = self.mod_select_attachment_cluster1_atom(cluster1_molgraph,
                                                                   cluster2_molrepr.unsqueeze(dim=0),
                                                                   target_molrepr.unsqueeze(dim=0))
        bond_atom_idx = dist.Categorical(probs=bond_atom_distr).sample().item()
        return bond_atom_idx

    def select_attachment_cluster2_atom(self,
                                        cluster2_molgraph: TensorGraph,
                                        cluster1_molrepr: torch.Tensor,
                                        target_molrepr: torch.Tensor) -> int:
        # Input validation
        self._check_single_batch_tgraph(cluster2_molgraph)
        assert cluster1_molrepr.ndim == 1
        assert target_molrepr.ndim == 1
        bond_atom_distr = self.mod_select_attachment_cluster2_atom(cluster2_molgraph,
                                                                   cluster1_molrepr.unsqueeze(dim=0),
                                                                   target_molrepr.unsqueeze(dim=0))
        bond_atom_idx = dist.Categorical(probs=bond_atom_distr).sample().item()
        return bond_atom_idx

    def select_attachment_bond_type(self,
                                    cluster1_molgraph: TensorGraph,
                                    cluster1_ai: int,
                                    cluster2_molgraph: TensorGraph,
                                    cluster2_ai: int,
                                    target_molrepr: torch.Tensor
                                    ) -> Chem.BondType:
        # Input validation
        self._check_single_batch_tgraph(cluster1_molgraph)
        assert 0 <= cluster1_ai < cluster1_molgraph.num_nodes()
        assert cluster1_molgraph.node_hiddens is not None
        self._check_single_batch_tgraph(cluster2_molgraph)
        assert 0 <= cluster2_ai < cluster2_molgraph.num_nodes()
        assert cluster2_molgraph.node_hiddens is not None

        cluster1_node_hiddens = cluster1_molgraph.node_hiddens[cluster1_ai].unsqueeze(dim=0)
        cluster2_node_hiddens = cluster2_molgraph.node_hiddens[cluster2_ai].unsqueeze(dim=0)
        target_molrepr = target_molrepr.unsqueeze(dim=0)

        bond_type_distr = self.mod_select_attachment_bond_type(cluster1_node_hiddens,
                                                               cluster2_node_hiddens,
                                                               target_molrepr)
        bond_type_int = torch.argmax(bond_type_distr, dim=1).item()
        return {
            0: Chem.BondType.UNSPECIFIED,  # Available, but the network isn't trained to output it
            1: Chem.BondType.SINGLE,
            2: Chem.BondType.DOUBLE,
            3: Chem.BondType.TRIPLE,
        }[bond_type_int]
