from common import *
import sys
import pytest
from mol_dataset import ZincDataset
from construct_motif_graph import *
from motif_vocab import MotifVocab

SAMPLE_SMILES = [
    'Cc1ccc(C(=O)/C=C/c2cccc(O)c2)cc1',
    'COc1ccc([C@@H](C[NH2+][C@@H]2CCCc3c2cnn3C)N(C)C)cc1OC',
    'COc1ccc(-c2cc3c(c(C)n2)CNC3=O)cc1',
    'COc1ccc(CCC(=O)NCC2(c3ccccc3)CCOCC2)cc1',
    'CN(Cc1ccco1)C(=O)C1=COCCO1',
    'COc1ccc(CNC(=O)[C@@H](C)[NH+]2CCCN(S(C)(=O)=O)CC2)cc1OC',
    'C[C@H]1C[NH2+]C[C@H](C)N1CCC(C)(C)O',
    'C[C@H](O)CC(C)(C)CNC(=O)N1CC=C(c2ccccc2Cl)CC1',
    'Cn1c([C@H]2C(c3ccccc3)=NCN2CCc2c[nH]c[nH+]2)cc2ccccc21',
    'C[C@@H](OC(=O)CC[C@@H]1NC(=O)NC1=O)C(=O)Nc1ccccc1C#N'
]


@pytest.mark.skip(reason="Complete the implementation please! :D")
def test_clear_atommap_output_order():
    """ This test proofs that clearing atom map also changes the atom order and tracks it. """

    smiles = '[CH3:4][C:5](=[O:6])[CH:7]=[CH:8][CH3:9]'

    atom_indices1 = [(atom.GetSymbol(), atom.GetDegree()) for atom in Chem.MolFromSmiles(smiles).GetAtoms()]
    smiles_no_atommap = clear_atommap(smiles)  # CC=CC(C)=O
    atom_indices2 = [(atom.GetSymbol(), atom.GetDegree()) for atom in Chem.MolFromSmiles(smiles_no_atommap).GetAtoms()]
    print(f"Atom indices (atommap): {atom_indices1}")
    print(f"Atom indices (no atommap): {atom_indices2}")
    assert atom_indices1 != atom_indices2

    # Clear atommap manually
    mol = Chem.MolFromSmiles(smiles)
    for atom in mol.GetAtoms():
        atom.SetAtomMapNum(0)
    smiles_no_atommap = Chem.MolToSmiles(mol)
    atom_order, _ = read_output_order(mol)
    print("Atom order: ", atom_order)

    mol = Chem.MolFromSmiles(smiles)
    print("Atommapped SMILES", [
        (atom.GetSymbol(), atom.GetDegree())
        for atom in mol.GetAtoms()
    ])

    mol_no_atommap = Chem.MolFromSmiles(smiles_no_atommap)
    print("Cleared atommap SMILES", [
        (atom.GetSymbol(), atom.GetDegree())
        for atom in mol_no_atommap.GetAtoms()
    ])

    # TODO check atoms are correctly mapped after reordering (using atom_order)
    # print("Fixed order SMILES", [
    #     (mol.GetAtomWithIdx(atom.GetIdx()).GetSymbol(), mol.GetAtomWithIdx(atom.GetIdx()).GetDegree())
    #     for atom in Chem.MolFromSmiles(smiles_no_atommap).GetAtoms()
    # ])


def test_atom_output_order():
    """ Tests the intended meaning of atom output order after canonicalization.
    That is, every entry is an index to the atom in the initial SMILES (before canonicalization):

    For example:
      Output order: [18, 17, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 4, 3, 2, 1, 0]
      Atom 0 of canonical SMILES was atom 18 in initial SMILES!
    """

    # https://github.com/rdkit/rdkit/discussions/5091

    smiles = 'C[C@H]1CCC[C@H](NC(=O)[C@@H](C)Sc2ncn[nH]2)[C@@H]1C'  # Non-canonical SMILES from training set

    # Canonize
    mol = Chem.MolFromSmiles(smiles)
    canon_smiles = Chem.MolToSmiles(mol)

    assert smiles != canon_smiles

    atom_order, _ = read_output_order(mol)
    print(f"SMILES: {smiles}")
    print(f"Canonical SMILES: {canon_smiles}")
    print(f"Output atom order: {atom_order}")

    mol1 = Chem.MolFromSmiles(smiles)
    mol2 = Chem.MolFromSmiles(canon_smiles)

    assert len(mol1.GetAtoms()) == len(mol2.GetAtoms())

    for atom2 in mol2.GetAtoms():
        initial_idx = atom_order[atom2.GetIdx()]
        atom1 = mol1.GetAtomWithIdx(initial_idx)
        assert atom_equals(atom1, atom2)


def test_construction():
    """ Tests that motif graph can be constructed on sample molecules. """
    motif_vocab = MotifVocab.load()

    for i, smiles in enumerate(SAMPLE_SMILES):
        try:
            construct_motif_graph(smiles, motif_vocab)
            print(f"{i} -> OK")
        except Exception as ex:
            print(f"{i} -> ERROR", file=sys.stderr)
            print(ex, file=sys.stderr)


def test_motif_graph_conversion():
    """ Tests that motif graph is constructed and converted back to the original SMILES. """
    motif_vocab = MotifVocab.load()

    for i, smiles in enumerate(SAMPLE_SMILES):
        print(f"{i + 1}/{len(SAMPLE_SMILES)} Converting {smiles}...")
        motif_graph = construct_motif_graph(smiles, motif_vocab)
        converted_smiles = convert_motif_graph_to_smiles(motif_graph, motif_vocab)
        print(f"{i + 1}/{len(SAMPLE_SMILES)} Original: {smiles}; Re-converted: {converted_smiles}")
