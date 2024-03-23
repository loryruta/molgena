from common import *
import os
from rdkit import Chem
import pickle
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
        row = motif_vocab.get_or_null(candidate)
        if row is not None:
            # The candidate was a Motif as is (frequent enough in training set)
            motif_id = int(row['id'])
            for atom in Chem.MolFromSmiles(candidate).GetAtoms():
                atom_clusters[atom.GetAtomMapNum()] = (motif_id, cluster_id)
            cluster_id += 1
        else:
            # The candidate must have been split in bonds and rings
            parts, _ = decompose_to_bonds_and_rings(candidate)
            for part in parts:
                row = motif_vocab.get_or_null(part)
                if row is None:
                    raise Exception(f"Missing motif in vocabulary;"
                                    f"\nSMILES \"{mol_smiles}\""
                                    f"\nCandidate: \"{candidate}\""
                                    f"\nMissing part: \"{part}\"")
                motif_id = int(row['id'])
                for atom in Chem.MolFromSmiles(part).GetAtoms():
                    atom_clusters[atom.GetAtomMapNum()] = (motif_id, cluster_id)
                cluster_id += 1

    # Finally construct motif graph
    motif_graph = nx.DiGraph()

    for motif_id, cluster_id in set(atom_clusters.values()):
        assert cluster_id not in motif_graph.nodes
        motif_graph.add_node(cluster_id, motif_id=motif_id)

    mol = Chem.MolFromSmiles(mol_smiles)
    for bond in mol.GetBonds():
        _, cluster_1 = atom_clusters[bond.GetBeginAtomIdx()]
        _, cluster_2 = atom_clusters[bond.GetEndAtomIdx()]
        if cluster_1 != cluster_2:
            # TODO attachment atoms!
            motif_graph.add_edge(cluster_1, cluster_2)
            motif_graph.add_edge(cluster_2, cluster_1)

    return motif_graph


def construct_and_save_motif_graphs():
    """ Constructs motif graphs for all training set samples; saves the final result to a .pkl file. """

    if path.exists(MOTIF_GRAPHS_PKL):
        in_ = input(f"File already exists \"{MOTIF_GRAPHS_PKL}\", overwrite? (y/N) ")
        if in_.lower() != "y":
            return 0

    training_set = pd.read_csv(ZINC_TRAINING_SET_CSV)
    motif_vocab = MotifVocab.load()

    started_at = time()
    logged_at = time()

    num_samples = len(training_set)

    print(f"Constructing motif graphs for {num_samples} training set samples...")

    motif_graphs = []
    for mol_id, mol_smiles in training_set['smiles'].items():
        motif_graph = construct_motif_graph(mol_smiles, motif_vocab)
        motif_graphs.append(motif_graph)

        if time() - logged_at > 5.0:
            num_left = num_samples - (mol_id + 1)
            time_left = num_left / ((mol_id + 1) / (time() - started_at))
            logging.info(f"Constructed motif graph for {mol_id + 1}/{num_samples} molecules; "
                         f"Time left: {time_left:.1f}s, "
                         f"Directory size: ")
            logged_at = time()

    print("Constructed motif graphs for all training set molecules!")

    print(f"Saving .pkl file: \"{MOTIF_GRAPHS_PKL}\"")

    with open(MOTIF_GRAPHS_PKL, 'wb') as file:
        pickle.dump(motif_graphs, file)

    print(f"Done!")

    return 0


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

    for i, motif in enumerate(motifs):
        motif_id = motif_vocab[motif]['id']

        plt.subplot(ceil(num_motifs / 5), 5, i + 1)
        plt.axis('off')
        plt.imshow(Draw.MolToImage(Chem.MolFromSmiles(motif)))
        plt.text(5, -10, f"Motif ID {motif_id}\n\"{motif}\"")
    plt.show()

    # Construct and display motif graph
    motif_graph = construct_motif_graph(mol_smiles, motif_vocab)

    motif_ids = nx.get_node_attributes(motif_graph, 'motif_id')
    nx.draw(motif_graph, labels=motif_ids)
    plt.show()


if __name__ == "__main__":
    # visualize_motif_graph()
    construct_and_save_motif_graphs()
