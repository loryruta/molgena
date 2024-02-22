import torch
from torch import Tensor
import networkx as nx
from rdkit import Chem
from rdkit.Chem import Draw
import matplotlib.pyplot as plt
from typing import *


def create_atom_feature_vector(atom) -> Tensor:
    atomic_num = atom.GetAtomicNum()
    # Atom.GetChiralTag()
    explicit_valence = atom.GetExplicitValence()
    formal_charge = atom.GetFormalCharge()
    # Atom.GetHybridization()
    # Atom.GetImplicitValence()
    isotope = atom.GetIsotope()
    mass = atom.GetMass()
    # Atom.GetMonomerInfo()
    # Atom.GetNoImplicit()
    # Atom.GetNumExplicitHs()
    # Atom.GetNumImplicitHs()
    # Atom.GetNumRadicalElectrons()
    # Atom.GetPDBResidueInfo()
    # Atom.GetTotalValence()
    return torch.tensor([atomic_num, explicit_valence, formal_charge, isotope, mass])


def create_bond_feature_vector(bond) -> Tensor:
    # Bond.GetBondDir()

    # Bond types:
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType
    bond_type = bond.GetBondType()
    # TODO Filter bond_type for only managed bonds (i.e. ZERO, SINGLE, DOUBLE, TRIPLE)?

    # Bond.GetIsConjugated()
    # Bond.GetStereo()
    # Bond.GetStereoAtoms()
    # Bond.GetValenceContrib()
    return torch.tensor([bond_type])


def create_mol_graph(smiles: str) -> Tuple[nx.Graph, Chem.Mol]:
    mol = Chem.MolFromSmiles(smiles)
    mol_graph = nx.Graph()

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
        features = create_atom_feature_vector(atom)
        mol_graph.add_node(atom.GetIdx(), features=features)

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        features = create_bond_feature_vector(bond)
        mol_graph.add_edge(u, v, features=features)

    return mol_graph, mol


def main():
    smiles = "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1"

    mol_graph, mol = create_mol_graph(smiles)

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    mol_img = Chem.Draw.MolToImage(mol)
    axs[0].set_axis_off()
    axs[0].imshow(mol_img)

    nx.draw(mol_graph, with_labels=True, ax=axs[1])

    plt.show()


if __name__ == "__main__":
    main()
