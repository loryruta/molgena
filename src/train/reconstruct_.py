import logging
from typing import *
from rdkit import Chem
import torch
from model.molgena import Molgena
from mol_graph import tensorize_smiles_list
from motif_graph.construct_motif_graph import construct_motif_graph
from motif_graph.convert_motif_graph import convert_motif_graph_to_smiles
from motif_graph.tensorize_motif_graph import create_mgraph_node_feature_vector, tensorize_mgraph
from motif_vocab import MotifVocab


class Reconstructor:
    def __init__(self, molgena: Molgena):
        self._molgena = molgena
        self._motif_vocab = MotifVocab.load()

    def attach_molecules(self,
                         mol1_smiles: str, mol1_ai: int,
                         mol2_smiles: str, mol2_ai: int,
                         bond_type: Chem.BondType
                         ) -> str:
        return  # TODO

    def reconstruct(self, mol_smiles: str) -> Union[str, False]:
        """ Runs inference to reconstruct the input molecule from none.

        :return:
            The SMILES of the reconstructed molecule, or an error if the generation failed (in case, see logs).
        """

        target_mol_repr = \
            self._molgena.encode(tensorize_smiles_list([mol_smiles]))

        partial_mol_smiles = ""

        iteration = 0

        while True:
            # Encode the partial molecule
            partial_mol_repr = \
                self._molgena.encode(tensorize_smiles_list([partial_mol_smiles]))

            # Select the next motif
            next_mid = self._molgena.select_motif(partial_mol_repr, target_mol_repr)
            if next_mid == self._motif_vocab.end_motif_id():
                break  # If we selected the END token, generation finished

            # If it's not the first step, we have to find the attachment
            if partial_mol_smiles != "":
                motif_features = create_mgraph_node_feature_vector(next_mid)

                # Select partial molecule's clusters to form the attachment with (potentially >1)
                partial_mol_mgraph = construct_motif_graph(partial_mol_smiles, self._motif_vocab)
                partial_mol_tensor_mgraph = tensorize_mgraph(partial_mol_mgraph, self._motif_vocab)
                cluster_indices = \
                    self._molgena.select_attachment_clusters(partial_mol_tensor_mgraph, motif_features)
                if len(cluster_indices) == 0:
                    logging.warning(f"Generation terminated: "
                                    f"no cluster selected for attachment (shouldn't happen)")
                    return False

                motif2_smiles = self._motif_vocab.at_id(next_mid)
                motif2_graph = tensorize_smiles_list([motif2_smiles])
                motif2_repr = self._molgena.encode(motif2_graph)

                for cid in cluster_indices:
                    mid = partial_mol_mgraph.nodes[cid]['motif_id']
                    motif1_smiles = self._motif_vocab.at_id(mid)
                    motif1_graph = tensorize_smiles_list([motif1_smiles])
                    motif1_repr = self._molgena.encode(motif1_graph)

                    # Considering a cluster1<->cluster2 connection, select atoms and bond type
                    motif1_ai = self._molgena.select_attachment_atom(motif1_graph, motif2_repr, target_mol_repr)
                    motif2_ai = self._molgena.select_attachment_atom(motif2_graph, motif1_repr, target_mol_repr)
                    bond_type = self._molgena.select_attachment_bond_type(
                        motif1_graph, motif1_ai,
                        motif2_graph, motif2_ai,
                        target_mol_repr
                    )

                    # Convert the partial molecule mgraph to SMILES just to obtain the [cid, motif_ai] -> ai mapping
                    _, cluster_atom_mapping = \
                        convert_motif_graph_to_smiles(partial_mol_mgraph, self._motif_vocab)

                    # Attach selected Motif to the partial molecule (don't rebuild the mgraph!)
                    partial_mol_smiles = self.attach_molecules(
                        partial_mol_smiles,
                        cluster_atom_mapping[cid, motif1_ai],
                        motif2_smiles,
                        motif2_ai,
                        bond_type
                        )

                iteration += 1
                if iteration >= 100:
                    logging.warning(f"Generation terminated: "
                                    f"exceeded max number of iterations: {iteration}")
                    return False

            return partial_mol_smiles
