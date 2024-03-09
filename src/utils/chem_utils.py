from rdkit import Chem
from typing import *


def mol_from_smiles(smiles: str, sanitize=True, kekulize=True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise Exception(f"Invalid SMILES string: \"{smiles}\"")
    if kekulize:
        Chem.Kekulize(mol)
    return mol


def clear_atommap(smiles: str) -> str:
    mol = mol_from_smiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol, kekuleSmiles=True)


def copy_atom(atom: Chem.Atom) -> Chem.Atom:
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def extract_mol_fragment(mol: Chem.Mol, atom_indices: Set[int]) -> Chem.Mol:
    """ Extracts a fragment (subset) of the input molecule keeping only the specified atoms.
    The chemical validity of the output fragment is ensured by incrementally building it with RWMol.
    """

    frag_mol = Chem.RWMol()

    old_to_new_atom_idx = {}

    for i in atom_indices:
        new_idx = frag_mol.AddAtom(copy_atom(mol.GetAtomWithIdx(i)))
        old_to_new_atom_idx[i] = new_idx

    for bond in mol.GetBonds():
        u = bond.GetBeginAtomIdx()
        v = bond.GetEndAtomIdx()
        if u in atom_indices and v in atom_indices:
            frag_mol.AddBond(
                old_to_new_atom_idx[u],
                old_to_new_atom_idx[v],
                bond.GetBondType()
            )

    Chem.Kekulize(frag_mol)

    return frag_mol.GetMol()