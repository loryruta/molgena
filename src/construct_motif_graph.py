from common import *
import os
from rdkit import Chem
import networkx as nx
from gen_motif_vocab import *
from math import *


def construct_motif_graph(mol_smiles: str, motif_vocab: MotifVocab) -> nx.Graph:
    """ Given a molecule and the motif vocabulary, construct the motif graph.
    The motif graph is a graph where nodes are motifs. """

    mol_smiles = set_atommap_to_indices(mol_smiles)

    atom_clusters: Dict[int, Tuple[int, int]] = {}  # Mol atom <-> (Motif ID, Cluster ID)
    cluster_id: int = 0

    # Assign every mol atom to its motif ID
    candidates = extract_motif_candidates(mol_smiles)
    for candidate_idx, candidate in enumerate(candidates):
        cand_smiles = to_vocabulary_format(candidate)
        is_motif = cand_smiles in motif_vocab.index
        if is_motif:
            # The candidate was a Motif as is (frequent enough in training set)
            motif_id = int(motif_vocab.loc[cand_smiles]['id'])
            for atom in mol_from_smiles(candidate).GetAtoms():
                atom_clusters[atom.GetAtomMapNum()] = (motif_id, cluster_id)
            cluster_id += 1
            # logging.debug(f"{mol_smiles} motif: \"{candidate_smiles}\"")
        else:
            # The candidate must have been split in bonds and rings
            parts, _ = decompose_to_bonds_and_rings(candidate)
            for part in parts:
                part_smiles = to_vocabulary_format(part)
                # logging.debug(f"{mol_smiles} candidate motif \"{candidate_smiles}\" decomposed to: \"{part_smiles}\"")
                if part_smiles not in motif_vocab.index:
                    raise Exception(f"Missing motif in vocabulary;"
                                    f"\nSMILES \"{mol_smiles}\""
                                    f"\nCandidate: \"{cand_smiles}\""
                                    f"\nCandidate (atommapped): \"{candidate}\""
                                    f"\nMissing part: \"{part_smiles}\""
                                    f"\nMissing part (atommapped): \"{part}\"")
                motif_id = int(motif_vocab.loc[part_smiles]['id'])
                for atom in Chem.MolFromSmiles(part).GetAtoms():
                    atom_clusters[atom.GetAtomMapNum()] = (motif_id, cluster_id)
                cluster_id += 1

    motif_graph = nx.DiGraph()

    for motif_id, cluster_id in set(atom_clusters.values()):
        assert cluster_id not in motif_graph.nodes
        motif_graph.add_node(cluster_id, motif_id=motif_id)

    mol = Chem.MolFromSmiles(mol_smiles)
    for bond in mol.GetBonds():
        _, cluster_1 = atom_clusters[bond.GetBeginAtomIdx()]
        _, cluster_2 = atom_clusters[bond.GetEndAtomIdx()]
        if cluster_1 != cluster_2:
            motif_graph.add_edge(cluster_1, cluster_2)

    return motif_graph


def construct_all_motif_graphs(write_file: bool):
    """ Tests that motif graph can be constructed for all training set samples. """

    training_set = pd.read_csv(ZINC_TRAINING_SET_CSV)
    motif_vocab = MotifVocab.load()

    started_at = time()
    logged_at = time()
    for mol_id, mol_smiles in training_set['smiles'].items():
        motif_graph = construct_motif_graph(mol_smiles, motif_vocab)

        # Save the motif graph to file (caching)!
        if write_file:
            nx.write_gml(motif_graph, motif_graph_gml_path(mol_id))

        if time() - logged_at > 5.0:
            num_left = len(training_set) - (mol_id + 1)
            time_left = num_left / ((mol_id + 1) / (time() - started_at))
            logging.info(f"Constructed motif graph for {mol_id + 1}/{len(training_set)} molecules; "
                         f"Time left: {time_left:.1f}s, "
                         f"Directory size: ")
            logged_at = time()

    logging.info("Constructed motif graphs for all training set molecules")


def visualize_motif_graph():
    """ Visualizes a random molecule, its decomposition in motifs/bonds/rings and the motif graph. """

    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw

    dataset = pd.read_csv(ZINC_TRAINING_SET_CSV)
    motif_vocab = MotifVocab.load()

    # Sample and display a molecule
    mol_smiles = dataset['smiles'].sample().iloc[0]

    plt.axis('off')
    plt.imshow(Draw.MolToImage(Chem.MolFromSmiles(mol_smiles)))
    plt.text(5, -10, mol_smiles)
    plt.show()

    # Split and display molecule's motifs
    motifs = decompose_mol(mol_smiles, motif_vocab)
    num_motifs = len(motifs)

    num_rows, num_cols = ceil(num_motifs / 5), 5
    fig, axs = plt.subplots(num_rows, num_cols, figsize=(10, 10))
    for ax in axs.flat:
        ax.axis('off')
    for i, motif in enumerate(motifs):
        motif_id = motif_vocab[motif]['id']

        ax = axs[i // 5, i % 5]
        ax.axis('off')
        ax.imshow(Draw.MolToImage(Chem.MolFromSmiles(motif)))
        ax.text(5, -10, f"Motif ID {motif_id}\n\"{motif}\"")
    plt.show()

    # Construct and display motif graph
    motif_graph = construct_motif_graph(mol_smiles, motif_vocab)

    motif_ids = nx.get_node_attributes(motif_graph, 'motif_id')
    nx.draw(motif_graph, labels=motif_ids)
    plt.show()


if __name__ == "__main__":
    # visualize_motif_graph()
    construct_all_motif_graphs(write_file=False)
