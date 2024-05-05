from common import *
import torch
from rdkit import Chem
from rdkit.Chem import Draw
from tensor_graph import TensorGraph, batch_tensor_graphs
from typing import *
from utils.tensor_utils import *


# inspect_dataset latest output:
# 2024-04-17 21:22:23,036 [INFO ] Atomic number;    Min: 6.00000, Max: 53.00000, Mean: 6.69403, Std: 2.24432
# 2024-04-17 21:22:23,364 [INFO ] Explicit valence; Min: 1.00000, Max: 6.00000, Mean: 2.76499, Std: 1.00490
# 2024-04-17 21:22:23,697 [INFO ] Formal charge;    Min: -1.00000, Max: 1.00000, Mean: 0.00875, Std: 0.13331
# 2024-04-17 21:22:24,024 [INFO ] Isotope;          Min: 0.00000, Max: 0.00000, Mean: 0.00000, Std: 0.00000
# 2024-04-17 21:22:24,299 [INFO ] Mass;             Min: 12.01100, Max: 126.90400, Mean: 13.44766, Std: 4.91093
# 2024-04-17 21:22:24,606 [INFO ] Bond type;        Min: 1.00000, Max: 3.00000, Mean: 1.27735, Std: 0.31801

# TODO rename to tensor_mol_graph (mol_graph is too generic)

def create_atom_features(atom) -> List[float]:
    atomic_num = atom.GetAtomicNum()
    # Atom.GetChiralTag()
    explicit_valence = atom.GetExplicitValence()
    formal_charge = atom.GetFormalCharge()
    # Atom.GetHybridization()
    # Atom.GetImplicitValence()
    # isotope = atom.GetIsotope()  # Always zero in our dataset
    mass = atom.GetMass()
    # Atom.GetMonomerInfo()
    # Atom.GetNoImplicit()
    # Atom.GetNumExplicitHs()
    # Atom.GetNumImplicitHs()
    # Atom.GetNumRadicalElectrons()
    # Atom.GetPDBResidueInfo()
    # Atom.GetTotalValence()

    # Normalization to 0 mean and unit std
    # Mean/std computed with inspect_dataset.py
    atomic_num = (atomic_num - 6.69403) / 2.24432
    explicit_valence = (explicit_valence - 2.76499) / 1.00490
    formal_charge = (formal_charge - 0.00875) / 0.13331
    # isotope = isotope  # It's always zero! Useless information
    mass = (mass - 13.44766) / 4.91093

    return [atomic_num, explicit_valence, formal_charge, 0, mass]


def create_bond_type_features(bond_type: Chem.BondType) -> torch.Tensor:
    # TODO one-hot feature vector?
    # TODO call this function from _create_bond_features
    return torch.tensor([int(bond_type)], dtype=torch.float32)


def _create_bond_features(bond) -> List[float]:
    # Bond.GetBondDir()

    # Bond types:
    # https://www.rdkit.org/docs/source/rdkit.Chem.rdchem.html#rdkit.Chem.rdchem.BondType
    bond_type = bond.GetBondTypeAsDouble()
    # bond_type = bond.GetBondTypeAsDouble()
    # TODO Filter bond_type for only managed bonds (i.e. ZERO, SINGLE, DOUBLE, TRIPLE)?

    # Bond.GetIsConjugated()
    # Bond.GetStereo()
    # Bond.GetStereoAtoms()
    # Bond.GetValenceContrib()

    # Normalization to 0 mean and unit std
    # Mean/std computed with inspect_dataset.py
    bond_type = (bond_type - 1.27735) / 0.31801

    return [bond_type]


# TODO rename to create_tensor_graph_from_smiles
def tensorize_smiles(smiles: Optional[str]) -> TensorGraph:
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

        for atom in mol.GetAtoms():  # Atoms are iterated in index order (from 0 to N)
            atom.SetAtomMapNum(atom.GetIdx())
            atom_features.append(create_atom_features(atom))

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


def tensorize_smiles_list(smiles_list: List[str]) -> TensorGraph:
    return batch_tensor_graphs([tensorize_smiles(smiles) for smiles in smiles_list])


def _main():
    import torch_geometric
    import networkx as nx
    import matplotlib.pyplot as plt

    smiles = "CCOC(=O)[C@@H]1CCCN(C(=O)c2nc(-c3ccc(C)cc3)n3c2CCCCC3)C1"

    tensor_graph: TensorGraph = tensorize_smiles(smiles)

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
