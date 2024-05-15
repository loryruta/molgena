import logging
from utils.chem_utils import *


def test_attach_molecules():
    """ Tests attach_molecules() function. """

    mol1_smiles = "c1ncon1"
    mol2_smiles = "c1ccccc1"
    mol1 = Chem.MolFromSmiles(mol1_smiles)
    mol2 = Chem.MolFromSmiles(mol2_smiles)
    mol1_ai = 0
    mol2_ai = 5

    # Run attach molecules
    mol1_atom = mol1.GetAtomWithIdx(mol1_ai)
    mol2_atom = mol2.GetAtomWithIdx(mol2_ai)
    logging.debug(f"Mol 1: \"{mol1_smiles}\" (atom: {mol1_atom.GetSymbol()} ({mol1_ai})), "
                  f"Mol 2: \"{mol2_smiles}\" (atom: {mol2_atom.GetSymbol()} ({mol2_ai}))")

    attached_mol_smiles = attach_molecules(mol1_smiles, mol1_ai,
                                           mol2_smiles, mol2_ai,
                                           Chem.BondType.SINGLE)
    logging.debug(f"Attached mol: \"{attached_mol_smiles}\"")
