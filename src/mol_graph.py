from common import *
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from tensor_graph import TensorGraph, batch_tensor_graphs
from typing import *


# TODO rename to tensor_mol_graph (mol_graph is too generic)

def _create_atom_features(atom) -> List[float]:
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
    return [atomic_num, explicit_valence, formal_charge, isotope, mass]


def _create_bond_features(bond) -> List[float]:
    # Bond.GetBondDir()

    # Bond types:
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType
    bond_type = bond.GetBondType()
    # bond_type = bond.GetBondTypeAsDouble()
    # TODO Filter bond_type for only managed bonds (i.e. ZERO, SINGLE, DOUBLE, TRIPLE)?

    # Bond.GetIsConjugated()
    # Bond.GetStereo()
    # Bond.GetStereoAtoms()
    # Bond.GetValenceContrib()
    return [bond_type]


# TODO rename to create_tensor_graph_from_smiles
def create_mol_graph_from_smiles(smiles: Optional[str]) -> TensorGraph:
    """ Parses the input SMILES string to a TensorGraph.
    Not: SMILES is allowed to be None; if such, or empty string, an empty TensorGraph is returned.
    """

    # Reference:
    # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

    tensor_graph: TensorGraph = TensorGraph()

    atom_features = []
    bond_features = []
    bonds = []

    if smiles:  # smiles != "" and smiles is not None
        mol = Chem.MolFromSmiles(smiles)

        for atom in mol.GetAtoms():
            atom.SetAtomMapNum(atom.GetIdx())
            atom_features.append(_create_atom_features(atom))

        for bond in mol.GetBonds():
            u = bond.GetBeginAtomIdx()
            v = bond.GetEndAtomIdx()
            bonds.append([u, v])
            bonds.append([v, u])

            tmp = _create_bond_features(bond)
            bond_features.extend([tmp, tmp])

    tensor_graph.node_features = torch.tensor(atom_features, dtype=torch.float)
    tensor_graph.edge_features = torch.tensor(bond_features, dtype=torch.float)
    tensor_graph.edges = torch.tensor(bonds, dtype=torch.long).t().contiguous()
    return tensor_graph


# TODO remove for tensorize_smiles_list
def create_tensor_graph_from_smiles_list(smiles_list: List[str]) -> TensorGraph:
    return batch_tensor_graphs([
        create_mol_graph_from_smiles(smiles) for smiles in smiles_list
    ])


def tensorize_smiles_list(smiles_list: List[str]) -> TensorGraph:
    return batch_tensor_graphs([
        create_mol_graph_from_smiles(smiles) for smiles in smiles_list
    ])


def _main():
    import torch_geometric
    import networkx as nx
    import matplotlib.pyplot as plt

    smiles = "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1"

    tensor_graph: TensorGraph = create_mol_graph_from_smiles(smiles)

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Draw molecule with rdkit
    mol_img = Chem.Draw.MolToImage(Chem.MolFromSmiles(smiles))
    axs[0].set_axis_off()
    axs[0].imshow(mol_img)

    # Draw molecule graph with networkx
    data = tensor_graph.to_torch_geometric()
    nx_graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(nx_graph, with_labels=True, ax=axs[1])

    plt.show()


if __name__ == "__main__":
    _main()
