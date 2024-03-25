from common import *
import os
import sys
from rdkit import Chem
import pickle
import networkx as nx
from gen_motif_vocab import *
from math import *


# Useful resources:
# https://github.com/rdkit/rdkit/discussions/5091


def construct_motif_graph(mol_smiles: str, motif_vocab: MotifVocab) -> nx.DiGraph:
    """ Given a molecule and the motif vocabulary, construct the motif graph.
    The motif graph is a graph where nodes are motifs. """

    mol_smiles = set_atommap_to_indices(mol_smiles)

    # CID = Cluster ID
    # MID = Motif ID
    # Cluster = group of atoms assigned to a Motif

    class ClusterInfo:
        cid: int  # Cluster ID
        mid: int  # Motif ID
        smiles: str  # SMILES for the cluster; atommapped to input molecule atom indices

    atom_clusters: Dict[int, ClusterInfo] = {}  # Mol atom -> ClusterInfo

    # Map input molecule atom indices to cluster -relative indices
    mol_atom_cluster_atom_indices: Dict[int, int] = {}

    next_cid: int = 0

    # Assign every mol atom to its motif ID
    candidates = extract_motif_candidates(mol_smiles)
    for candidate_idx, candidate in enumerate(candidates):
        row = motif_vocab.at_smiles_or_null(candidate)
        if row is not None:
            # The candidate was a Motif as is (frequent enough in training set)
            info = ClusterInfo()
            info.cid = next_cid
            info.mid = int(row['id'])  # motif_id
            info.smiles = candidate
            for atom in Chem.MolFromSmiles(candidate).GetAtoms():
                atom_idx = atom.GetAtomMapNum()
                assert atom_idx not in atom_clusters  # TODO an atom could be assigned to many clusters!
                atom_clusters[atom_idx] = info
                mol_atom_cluster_atom_indices[atom_idx] = atom.GetIdx()
            next_cid += 1
        else:
            # The candidate must have been split in bonds and rings
            parts, _ = decompose_to_bonds_and_rings(candidate)
            for part in parts:
                row = motif_vocab.at_smiles_or_null(part)
                if row is None:
                    raise Exception(f"Missing motif in vocabulary;"
                                    f"\nSMILES \"{mol_smiles}\""
                                    f"\nCandidate: \"{candidate}\""
                                    f"\nMissing part: \"{part}\"")
                info = ClusterInfo()
                info.cid = next_cid
                info.mid = int(row['id'])  # motif_id
                info.smiles = part
                for atom in Chem.MolFromSmiles(part).GetAtoms():
                    atom_idx = atom.GetAtomMapNum()
                    assert atom_idx not in atom_clusters  # TODO an atom could be assigned to many clusters!
                    atom_clusters[atom_idx] = info
                    mol_atom_cluster_atom_indices[atom_idx] = atom.GetIdx()
                next_cid += 1

    # Finally construct motif graph
    motif_graph = nx.DiGraph()

    clusters: Set[ClusterInfo] = set(atom_clusters.values())  # Set of unique clusters

    # Cache motif indices per cluster; used to convert cluster -relative indices to motif -relative indices
    # This is important to express attachments canonically
    cluster_motif_indices: Dict[int, List[int]] = {}
    for cluster in clusters:
        # Clear atommap + canonization (to make cluster compatible with Motif vocabulary)
        _, (canon_atom_order, _) = canon_smiles(cluster.smiles)
        assert cluster.cid not in cluster_motif_indices
        cluster_motif_indices[cluster.cid] = canon_atom_order

    # Add nodes
    for cluster in clusters:
        motif_graph.add_node(cluster.cid, motif_id=cluster.mid)

    # Add edges
    mol = Chem.MolFromSmiles(mol_smiles)
    Chem.Kekulize(mol)  # No aromatic bonds in the attachments!

    for bond in mol.GetBonds():
        a1 = bond.GetBeginAtomIdx()  # Input molecule -relative index
        a2 = bond.GetEndAtomIdx()  # Input molecule -relative index
        cluster1 = atom_clusters[a1]
        cluster2 = atom_clusters[a2]

        # The two atoms are in the same cluster; no inter-cluster bond to create!
        if cluster1.cid == cluster2.cid:
            continue

        # Convert bond atom indices; first to candidate -relative indices, then to motif -relative indices
        cluster_a1 = mol_atom_cluster_atom_indices[a1]
        cluster_a2 = mol_atom_cluster_atom_indices[a2]
        motif_a1 = cluster_motif_indices[cluster1.cid].index(cluster_a1)
        motif_a2 = cluster_motif_indices[cluster2.cid].index(cluster_a2)

        # !!! DEBUG ONLY: verify that -relative indices match the atom !!!
        cluster_mol1 = Chem.MolFromSmiles(cluster1.smiles)
        cluster_mol2 = Chem.MolFromSmiles(cluster2.smiles)
        motif_mol1 = Chem.MolFromSmiles(motif_vocab.at_id(cluster1.mid)['smiles'])
        motif_mol2 = Chem.MolFromSmiles(motif_vocab.at_id(cluster2.mid)['smiles'])
        assert atom_equals(motif_mol1.GetAtomWithIdx(motif_a1), cluster_mol1.GetAtomWithIdx(cluster_a1))
        assert atom_equals(motif_mol2.GetAtomWithIdx(motif_a2), cluster_mol2.GetAtomWithIdx(cluster_a2))

        cid1 = cluster1.cid
        cid2 = cluster2.cid

        # Sanity check: if bond was present, it must be bidirectional
        assert ((cid1, cid2) in motif_graph.edges) == ((cid2, cid1) in motif_graph.edges)

        # Add bond cluster1<->cluster2 with attachment information
        if (cid1, cid2) not in motif_graph.edges:
            motif_graph.add_edge(cid1, cid2, attachment={})
            motif_graph.add_edge(cid2, cid1, attachment={})
        motif_graph.edges[cid1, cid2]['attachment'][motif_a1, motif_a2] = bond.GetBondType()
        motif_graph.edges[cid2, cid1]['attachment'][motif_a2, motif_a1] = bond.GetBondType()

    return motif_graph


def convert_motif_graph_to_smiles(motif_graph: nx.DiGraph, motif_vocab: MotifVocab) -> str:
    """ Converts the input motif graph to SMILES. """

    new_mol = Chem.RWMol()

    # (Cluster ID, Motif atom ID) -> New molecule atom ID
    # Where atom at cluster, motif ID can be found in the new molecule
    cluster_atom_map = {}

    # Add clusters to the final molecule
    for cid in motif_graph.nodes:
        motif_id = motif_graph.nodes[cid]['motif_id']
        motif_smiles = motif_vocab.at_id(motif_id)['smiles']

        print(f"Cluster {cid} -> Motif {motif_smiles}")

        motif_mol = Chem.MolFromSmiles(motif_smiles)
        Chem.Kekulize(motif_mol)  # No aromatic bonds when building...

        for atom in motif_mol.GetAtoms():
            new_idx = new_mol.AddAtom(copy_atom(atom))
            atom.SetAtomMapNum(new_idx)  # Save the new_idx, so later we can make bonds
            cluster_atom_map[(cid, atom.GetIdx())] = new_idx

        for bond in motif_mol.GetBonds():
            new_atom1 = bond.GetBeginAtom()
            new_atom2 = bond.GetEndAtom()
            new_mol.AddBond(new_atom1.GetAtomMapNum(), new_atom2.GetAtomMapNum(), bond.GetBondType())

    # Interconnect the clusters using attachment information
    for cid1, cid2 in motif_graph.edges:
        attachment = motif_graph.edges[cid1, cid2]['attachment']

        print(f"cid1: {cid1}, cid2: {cid2}, attachment: {attachment}")

        for (motif_a1, motif_a2), bond_type in attachment.items():
            new_a1 = cluster_atom_map[(cid1, motif_a1)]
            new_a2 = cluster_atom_map[(cid2, motif_a2)]
            if new_mol.GetBondBetweenAtoms(new_a1, new_a2) is None:
                new_mol.AddBond(new_a1, new_a2, bond_type)

    smiles = Chem.MolToSmiles(new_mol)
    return canon_smiles(smiles)[0]


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
        # logging.debug(f"{mol_id + 1}/{num_samples} - Constructing motif graph for molecule \"{mol_smiles}\"...")

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


def load_motif_graphs() -> List[nx.Graph]:
    try:
        with open(MOTIF_GRAPHS_PKL, "rb") as file:
            return pickle.load(file)
    except Exception as ex:
        print(
            f"Motif graph not found at \"{MOTIF_GRAPHS_PKL}\"; you have to generate it using: construct_motif_graph.py",
            file=sys.stderr)
        raise ex


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
        motif_id = motif_vocab.at_smiles(motif)['id']

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
