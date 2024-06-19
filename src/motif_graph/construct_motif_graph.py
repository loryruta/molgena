from common import *
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
    candidates = sorted(candidates)  # For consistent mgraph across multiple runs

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
                assert atom_idx not in atom_clusters  # An atom must not be assigned to multiple clusters!
                atom_clusters[atom_idx] = info
                mol_atom_cluster_atom_indices[atom_idx] = atom.GetIdx()
            next_cid += 1
        else:
            # The candidate must have been split in bonds and rings
            parts = decompose_motif_candidate(candidate)
            parts = sorted(parts)  # For consistent mgraph across multiple runs

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
                    assert atom_idx not in atom_clusters  # An atom must not be assigned to multiple clusters!
                    atom_clusters[atom_idx] = info
                    mol_atom_cluster_atom_indices[atom_idx] = atom.GetIdx()
                next_cid += 1

    # Finally construct motif graph
    mgraph = nx.DiGraph()

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
    for cluster in sorted(clusters, key=lambda cluster_: cluster_.cid):
        mgraph.add_node(cluster.cid, motif_id=cluster.mid)

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
        cluster_a1 = mol_atom_cluster_atom_indices[a1]  # Cluster -relative index
        cluster_a2 = mol_atom_cluster_atom_indices[a2]  # Cluster -relative index
        motif_a1 = cluster_motif_indices[cluster1.cid].index(cluster_a1)  # Motif -relative index
        motif_a2 = cluster_motif_indices[cluster2.cid].index(cluster_a2)  # Motif -relative index

        # !!! DEBUG ONLY: verify that -relative indices match the atom !!!
        # cluster_mol1 = Chem.MolFromSmiles(cluster1.smiles)
        # cluster_mol2 = Chem.MolFromSmiles(cluster2.smiles)
        # motif_mol1 = Chem.MolFromSmiles(motif_vocab.at_id(cluster1.mid)['smiles'])
        # motif_mol2 = Chem.MolFromSmiles(motif_vocab.at_id(cluster2.mid)['smiles'])
        # assert atom_equals(motif_mol1.GetAtomWithIdx(motif_a1), cluster_mol1.GetAtomWithIdx(cluster_a1))
        # assert atom_equals(motif_mol2.GetAtomWithIdx(motif_a2), cluster_mol2.GetAtomWithIdx(cluster_a2))

        cid1 = cluster1.cid
        cid2 = cluster2.cid

        # Sanity check: if bond was present, it must be bidirectional
        assert ((cid1, cid2) in mgraph.edges) == ((cid2, cid1) in mgraph.edges)

        # Add bond cluster1<->cluster2 with attachment information

        # Every (cluster1, cluster2) pair must have exactly 1 attachment (i.e. 1 bond),
        # therefore when reaching this point, attachment information shouldn't be already present
        assert (cid1, cid2) not in mgraph.edges
        assert (cid2, cid1) not in mgraph.edges

        mgraph.add_edge(cid1, cid2, attachment={})
        mgraph.add_edge(cid2, cid1, attachment={})

        # Store attachment information as a (motif atom index 1, motif atom index 2, bond type)
        mgraph.edges[cid1, cid2]['attachment'] = (motif_a1, motif_a2, bond.GetBondType())
        mgraph.edges[cid2, cid1]['attachment'] = (motif_a2, motif_a1, bond.GetBondType())

    return mgraph


def _visualize_motif_graph():
    """Visualizes a random molecule, its decomposition in motifs/bonds/rings and the motif graph."""

    import matplotlib.pyplot as plt
    from rdkit.Chem import Draw
    from time import time_ns

    def plt_img_caption(text: str, size: int = 8):
        plt.text(0.5, -0.1, text, size=size, ha='center', transform=plt.gca().transAxes)

    dataset = pd.read_csv(TRAINING_CSV)
    motif_vocab = MotifVocab.load()

    # Sample and display a molecule
    seed = (time_ns() & ((1 << 32) - 1))
    # mol_smiles = dataset['smiles'].sample(n=1, random_state=seed).iloc[0]
    mol_smiles = "CCCCOc1ccc(-c2nnc(C[NH+]3CC[C@H](C)[C@H](O)C3)o2)cc1"

    plt.axis('off')
    plt.imshow(smiles_to_image(mol_smiles))
    plt_img_caption(mol_smiles)
    plt.show()

    # Split and display molecule's motifs
    motifs = decompose_mol(mol_smiles, motif_vocab)
    num_motifs = len(motifs)

    for i, motif in enumerate(motifs):
        motif_id = motif_vocab.at_smiles(motif)['id']

        plt.subplot(ceil(num_motifs / 3), 3, i + 1)
        plt.axis('off')
        plt.imshow(smiles_to_image(motif))
        plt_img_caption(f"Motif ID {motif_id}\n\"{motif}\"")
    plt.show()

    # Construct and display motif graph
    motif_graph = construct_motif_graph(mol_smiles, motif_vocab)

    motif_ids = nx.get_node_attributes(motif_graph, 'motif_id')
    nx.draw(motif_graph, labels=motif_ids)
    plt.show()


if __name__ == "__main__":
    _visualize_motif_graph()
