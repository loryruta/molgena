from common import *
import networkx as nx
from typing import *
from rdkit import Chem
from motif_vocab import MotifVocab
from utils.chem_utils import *


def convert_motif_graph_to_smiles(motif_graph: nx.DiGraph, motif_vocab: MotifVocab) \
        -> Tuple[str, Dict[Tuple[int, int], int]]:
    """ Converts the input motif graph to SMILES.

    :return:
        A tuple (generated_smiles, cluster_atom_map), where cluster_atom_map is a mapping from (cluster_id, motif_atom_index) to atom_index.
        It basically tells where an atom, from the original Motif graph, went in the final molecule.
    """

    new_mol = Chem.RWMol()

    # (Cluster ID, Motif -relative index) -> Atom index
    # Given cluster and motif -relative index pointing to an atom, tells the generated molecule
    cluster_atom_map = {}

    # Add clusters to the final molecule
    for cid in motif_graph.nodes:
        motif_id = motif_graph.nodes[cid]['motif_id']
        motif_smiles = motif_vocab.at_id(motif_id)['smiles']

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

        for (motif_a1, motif_a2), bond_type in attachment.items():
            new_a1 = cluster_atom_map[(cid1, motif_a1)]
            new_a2 = cluster_atom_map[(cid2, motif_a2)]
            if new_mol.GetBondBetweenAtoms(new_a1, new_a2) is None:
                new_mol.AddBond(new_a1, new_a2, bond_type)

    smiles = Chem.MolToSmiles(new_mol)
    return canon_smiles(smiles)[0], cluster_atom_map
