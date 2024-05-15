import logging
from os import path
import sys
from typing import *
from rdkit import Chem
import torch
from mol_dataset import ZincDataset
from model.molgena import Molgena
from mol_graph import tensorize_smiles_list
from motif_graph.cache import MgraphCache
from motif_graph.construct_motif_graph import construct_motif_graph
from motif_graph.convert_motif_graph import convert_motif_graph_to_smiles
from motif_graph.tensorize_motif_graph import create_mgraph_node_feature_vector, tensorize_mgraph
from motif_vocab import MotifVocab
from runtime_context import RuntimeContext, parse_runtime_context_from_cmdline
from utils.chem_utils import attach_molecules, check_smiles_chemical_validity
from utils.misc_utils import stopwatch


class MaxIterationReachedException(Exception):
    """ Exception thrown if the max number of iterations is reached. """
    pass


class InvalidMolException(Exception):
    """ Exception thrown if the partial molecule becomes chemically invalid. """
    pass


class MissingMotifException(Exception):
    """ Exception thrown if vocabulary is missing a motif. """
    pass


class Stats:
    num_tests: int = 0
    """ Number of reconstructions performed. """

    num_succeeded: int = 0
    """ Number of reconstructions that succeeded (reconstructed SMILES equals to target SMILES). """

    num_max_iteration_reached: int = 0
    """ Number of reconstructions that reached MAX_ITERATIONS. """

    min_iterations: int = sys.maxsize
    """ Min number of iterations performed to reconstruct the final molecule. """

    max_iterations: int = -1
    """ Max number of iterations performed to reconstruct the final molecule (excluding MAX_ITERATIONS). """

    num_chemically_invalid: int = 0
    num_missing_motifs: int = 0

    def __iadd__(self, other: 'Stats') -> 'Stats':
        self.num_tests += other.num_tests
        self.num_succeeded += other.num_succeeded
        self.num_max_iteration_reached += other.num_max_iteration_reached
        self.min_iterations = min(self.min_iterations, other.min_iterations)
        self.max_iterations = max(self.max_iterations, other.max_iterations)
        self.num_chemically_invalid += other.num_chemically_invalid
        self.num_missing_motifs += other.num_missing_motifs
        return self

    def __str__(self):
        return f"Num tests: {self.num_succeeded} succeeded/{self.num_tests}, " \
               f"Max iterations reached: {self.num_max_iteration_reached}, " \
               f"Iterations: {self.min_iterations} min/{self.max_iterations} max, " \
               f"Num chemically invalid: {self.num_chemically_invalid}, " \
               f"Num missing motifs: {self.num_missing_motifs}"


class ReconstructTask:
    MAX_ITERATIONS = 256

    def __init__(self, context: RuntimeContext):
        self._molgena = Molgena(context.config)
        self._motif_vocab = MotifVocab.load()

        # Load latest checkpoint
        if not path.isfile(context.latest_checkpoint_file):
            raise Exception("Latest checkpoint not found: can't initialize the model")
        checkpoint = torch.load(context.latest_checkpoint_file)
        self._molgena.load_state_dict(checkpoint["model"])

    def _run_reconstruct(self, mol_smiles: str) -> Tuple[str, int]:
        """ Runs inference to reconstruct the input molecule from scratch.

        :return:
            The SMILES of the reconstructed molecule.
        TODO raises
        """

        target_molrepr = self._molgena.encode(tensorize_smiles_list([mol_smiles]))

        pmol_smiles = ""

        iteration = 0
        while True:
            logging.debug(f"[Reconstruct] {iteration + 1:03}/{self.MAX_ITERATIONS} "
                          f"Target mol: \"{mol_smiles}\", "
                          f"Partial mol: \"{pmol_smiles}\"")

            iteration += 1
            if iteration > self.MAX_ITERATIONS:  # Max number of iterations reached
                raise MaxIterationReachedException()

            # Check chemical validity
            if not check_smiles_chemical_validity(pmol_smiles):
                raise InvalidMolException()

            # Encode the partial molecule
            pmol_repr = self._molgena.encode(tensorize_smiles_list([pmol_smiles]))

            # Select the next motif
            next_mid = self._molgena.select_motif(pmol_repr, target_molrepr)
            if next_mid == self._motif_vocab.end_motif_id():
                logging.debug(f"[Reconstruct] Done; Partial mol: \"{pmol_smiles}\"")
                break  # If we selected the END token, generation finished

            # First step of generation
            if pmol_smiles == "":
                motif_smiles = self._motif_vocab.at_id(next_mid)['smiles']
                pmol_smiles = motif_smiles
                logging.debug(f"[Reconstruct] Initial step; Attaching: \"{motif_smiles}\" (MID: {next_mid})")
                continue

            # If it's not the first step, we have to find the attachment
            motif_mrepr = create_mgraph_node_feature_vector(next_mid)

            # Select partial molecule's clusters to form the attachment with (potentially >1)
            try:
                pmol_mgraph = MgraphCache.instance().get_or_construct_mgraph(pmol_smiles)
            except Exception as _:  # Motif not found
                raise MissingMotifException()

            pmol_tensor_mgraph, cid_mappings = tensorize_mgraph(pmol_mgraph, self._motif_vocab)
            pmol_tensor_mgraph.make_batched()
            # cid_mappings: cid <-> node_idx
            self._molgena.encode_mgraph(pmol_tensor_mgraph)  # Compute mgraph node_hiddens
            next_cluster_node_idx = self._molgena.select_attachment_cluster(pmol_tensor_mgraph, motif_mrepr)

            cluster1_cid = cid_mappings.inverse[next_cluster_node_idx]
            cluster1_mid = pmol_mgraph.nodes[cluster1_cid]['motif_id']
            cluster1_smiles = self._motif_vocab.at_id(cluster1_mid)['smiles']
            cluster1_molgraph = tensorize_smiles_list([cluster1_smiles])
            cluster1_molrepr = self._molgena.encode(cluster1_molgraph)

            _, cluster_atom_map = convert_motif_graph_to_smiles(pmol_mgraph, self._motif_vocab)
            # cluster_atom_map: (cid, motif_ai) -> node_idx

            cluster2_mid = next_mid
            cluster2_smiles = self._motif_vocab.at_id(cluster2_mid)['smiles']
            cluster2_molgraph = tensorize_smiles_list([cluster2_smiles])
            cluster2_molrepr = self._molgena.encode(cluster2_molgraph)

            cluster1_ai = self._molgena.select_attachment_cluster1_atom(cluster1_molgraph,
                                                                        cluster2_molrepr,
                                                                        target_molrepr)
            cluster2_ai = self._molgena.select_attachment_cluster2_atom(cluster2_molgraph,
                                                                        cluster1_molrepr,
                                                                        target_molrepr)

            bond_type = self._molgena.select_attachment_bond_type(cluster1_molgraph, cluster1_ai,
                                                                  cluster2_molgraph, cluster2_ai, target_molrepr)

            # logging.debug(f"[Reconstruct] Attachment:")
            # logging.debug(
            #     f"[Reconstruct]   Cluster 1: \"{cluster1_smiles}\" (MID: {cluster1_mid}), Atom: {cluster1_ai}, CID: {cluster1_cid}")
            # logging.debug(
            #     f"[Reconstruct]   Cluster 2: \"{cluster2_smiles}\" (MID: {cluster2_mid}), Atom: {cluster2_ai}")
            # logging.debug(f"[Reconstruct]   Bond type: {bond_type.name}")

            pmol_ai = cluster_atom_map[cluster1_cid, cluster1_ai]
            motif_ai = cluster2_ai
            pmol_smiles = \
                attach_molecules(pmol_smiles, pmol_ai, cluster2_smiles, motif_ai, bond_type)
            # logging.debug(f"[Reconstruct] Attachment done; Partial mol: \"{pmol_smiles}\"")
        return pmol_smiles, iteration

    def reconstruct(self, mol_smiles: str) -> Stats:
        stats = Stats()
        stats.num_tests = 1
        try:
            reconstructed_mol_smiles, iteration = self._run_reconstruct(mol_smiles)
            if reconstructed_mol_smiles == mol_smiles:
                stats.num_succeeded = 1
                stats.min_iterations = iteration
                stats.max_iterations = iteration
        except MaxIterationReachedException as _:
            stats.num_max_iteration_reached = 1
        except InvalidMolException as _:
            stats.num_chemically_invalid = 1
        except MissingMotifException as _:
            stats.num_missing_motifs = 1
        return stats

    def reconstruct_batch(self, mol_smiles_list: List[str]) -> Stats:
        stats = Stats()
        log_stopwatch = stopwatch()
        for i, mol_smiles in enumerate(mol_smiles_list):
            stats += self.reconstruct(mol_smiles)

            if log_stopwatch() > 2.:
                logging.info(f"[Reconstruct] {i + 1}/{len(mol_smiles_list)} mol(s) ready; Stats:")
                logging.info(f"[Reconstruct]   {stats}")
                log_stopwatch = stopwatch()

        return stats


def _main():
    context = parse_runtime_context_from_cmdline()
    task = ReconstructTask(context)

    test_set = ZincDataset.test_set()
    test_batch = test_set.df['smiles'].tolist()  # mol_smiles_list
    stats = task.reconstruct_batch(test_batch)
    logging.info(f"Reconstruct stats; {stats}")


if __name__ == "__main__":
    _main()
