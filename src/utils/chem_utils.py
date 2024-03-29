from rdkit import Chem
from typing import *


def mol_from_smiles(smiles: str, sanitize=True, kekulize=True) -> Chem.Mol:
    mol = Chem.MolFromSmiles(smiles, sanitize=sanitize)
    if mol is None:
        raise Exception(f"Invalid SMILES string: \"{smiles}\"")
    if kekulize:
        Chem.Kekulize(mol)
    return mol


def copy_atom(atom: Chem.Atom) -> Chem.Atom:
    new_atom = Chem.Atom(atom.GetSymbol())
    new_atom.SetFormalCharge(atom.GetFormalCharge())
    new_atom.SetAtomMapNum(atom.GetAtomMapNum())
    return new_atom


def atom_equals(atom1: Chem.Atom, atom2: Chem.Atom) -> bool:
    """ Checks that atom1 is equals to atom2. """
    result = True
    result &= atom1.GetSymbol() == atom2.GetSymbol()
    result &= atom1.GetFormalCharge() == atom2.GetFormalCharge()
    # result &= atom1.GetAtomMapNum() == atom2.GetAtomMapNum()
    result &= atom1.GetDegree() == atom2.GetDegree()
    # TODO anything else?

    if not result:
        print("ATOMS AREN'T EQUAL!")

    return result


def read_output_order(mol: Chem.Mol) -> Tuple[List[int], List[int]]:
    """ Reads the output atom/bond order after calling Chem.MolToSmiles.
    Useful for recovering the initial atom/bond order after e.g. canonizing or clearing atommap. """

    # Source:
    # https://github.com/rdkit/rdkit/discussions/5091#discussioncomment-2352356

    if not mol.HasProp("_smilesAtomOutputOrder"):
        raise Exception("Property _smilesAtomOutputOrder not found; probably lacking call to Chem.MolToSmiles")

    # For an unknown reason, the result is a string; e.g. '[1, 2, 18, 17, 16, ...]'

    atom_order = list(map(int, mol.GetProp("_smilesAtomOutputOrder")[1:-2].split(",")))
    bond_order = list(map(int, mol.GetProp("_smilesBondOutputOrder")[1:-2].split(",")))
    return atom_order, bond_order


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

    return frag_mol.GetMol()


def attach_molecules(mol_a_smiles: str, mol_b_smiles: str, attachment: List[Tuple[int, int, Chem.BondType]]) -> str:
    # TODO NOT TESTED YET!

    mol_a = mol_from_smiles(mol_a_smiles)
    mol_b = mol_from_smiles(mol_b_smiles)

    new_mol = Chem.RWMol(mol_a)

    mol_b_to_new_mol_indices = []  # Map mol_b atom indices to new_mol atom indices
    for i, atom in enumerate(mol_b.GetAtoms()):
        new_idx = new_mol.AddAtom(copy_atom(atom))
        mol_b_to_new_mol_indices[i] = new_idx

    for bond in mol_b.GetBonds():
        new_mol.AddBond(
            bond.GetBeginAtomIdx(),
            bond.GetEndAtomIdx(),
            bond.GetBondType()
        )

    for mol_a_atom_idx, mol_b_atom_idx, bond_type in attachment:
        if bond_type != Chem.BondType.UNSPECIFIED:  # Ignore BondType.UNSPECIFIED (that is 0)
            continue

        # We should be making only SINGLE, DOUBLE, TRIPLE covalent bonds
        assert bond_type in [Chem.BondType.SINGLE, Chem.BondType.DOUBLE, Chem.BondType.TRIPLE]

        new_mol.AddBond(
            new_mol.GetAtomWithIdx(mol_a_atom_idx),
            new_mol.GetAtomWithIdx(mol_b_to_new_mol_indices[mol_b_atom_idx]),
            bond_type
        )

    return Chem.MolToSmiles(new_mol, kekuleSmiles=True)


def clear_atommap(mol_smiles: str) -> str:
    """ Given a SMILES, clears its atommap. """
    mol = Chem.MolFromSmiles(mol_smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    return Chem.MolToSmiles(mol)


def canon_smiles(smiles: str) -> Tuple[str, Tuple[List[int], List[int]]]:
    """ Given a SMILES, returns a unique version for it (clears atommap!).
    Useful for fetching SMILES in Motif vocabulary. """
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)  # Clear atom map
    out_smiles = Chem.MolToSmiles(mol)  # Canonize

    # Retrieve the ordering of the output molecule (atoms and bonds)
    atom_order, bond_order = read_output_order(mol)

    return out_smiles, (atom_order, bond_order)


def set_atommap_to_indices(mol_smiles: str):
    """ Given a SMILES, sets every atoms' map number to its index. """
    mol = Chem.MolFromSmiles(mol_smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(atom.GetIdx())
    return Chem.MolToSmiles(mol)
