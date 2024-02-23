import torch
from rdkit import Chem
from rdkit.Chem import Draw
from typing import *


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


def create_mol_graph(smiles: str):
    """ Creates the torch_geometric graph of the input molecule. """

    # Reference:
    # https://pytorch-geometric.readthedocs.io/en/latest/get_started/introduction.html

    node_features = []
    edge_features = []
    edges = []

    mol = Chem.MolFromSmiles(smiles)

    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
        node_features.append(_create_atom_features(atom))

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        edges.append([u, v])
        edges.append([v, u])

        t = _create_bond_features(bond)
        edge_features.append(t)
        edge_features.append(t)

    return (
        torch.tensor(node_features, dtype=torch.float),
        torch.tensor(edge_features, dtype=torch.float),
        torch.tensor(edges, dtype=torch.long).t().contiguous()
    )


def _main():
    import torch_geometric
    import networkx as nx
    import matplotlib.pyplot as plt

    smiles = "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1"

    node_features, edge_features, edges = create_mol_graph(smiles)

    fig, axs = plt.subplots(2, 1, figsize=(8, 10))

    # Draw molecule with rdkit
    mol_img = Chem.Draw.MolToImage(Chem.MolFromSmiles(smiles))
    axs[0].set_axis_off()
    axs[0].imshow(mol_img)

    # Draw molecule graph with networkx
    data = torch_geometric.data.Data(x=node_features, edge_index=edges, edge_attr=edge_features)
    nx_graph = torch_geometric.utils.to_networkx(data, to_undirected=True)
    nx.draw(nx_graph, with_labels=True, ax=axs[1])

    plt.show()


if __name__ == "__main__":
    _main()
