from common import *
from rdkit import Chem
import pandas as pd
from typing import *
from collections import Counter
import logging
from time import time
from utils.chem_utils import *
from motif_vocab import MotifVocab


def to_vocabulary_format(mol_smiles: str):
    mol_smiles = clear_atommap(mol_smiles)
    mol_smiles = Chem.CanonSmiles(mol_smiles)
    return mol_smiles


def extract_motif_candidates(mol_smiles: str) -> Set[str]:
    """ Extracts Motif candidates from the input molecular graph, according to the technique described in:
    "Hierarchical Generation of Molecular Graphs using Structural Motifs", Appendix A

     :param mol_smiles:
        A canonical SMILES string, atommapped (important: first canonized and then atommaped).
     """

    # HierVAE code reference:
    # https://github.com/wengong-jin/hgraph2graph/blob/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767/polymers/poly_hgraph/chemutils.py#L44

    mol = Chem.MolFromSmiles(mol_smiles)
    Chem.Kekulize(mol)

    new_mol = Chem.RWMol(mol)

    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()

        if bond.IsInRing():
            continue

        if u.IsInRing() and v.IsInRing():
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif u.IsInRing() and v.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(u))  # Also copies atommap
            new_mol.AddBond(new_idx, v.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif v.IsInRing() and u.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(v))  # Also copies atommap
            new_mol.AddBond(u.GetIdx(), new_idx, bond.GetBondType())
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())

    candidates = set()
    for atom_indices in Chem.GetMolFrags(new_mol):
        cand_mol = extract_mol_fragment(new_mol, atom_indices)
        cand_smiles = Chem.MolToSmiles(cand_mol)
        candidates.add(cand_smiles)
    return candidates


def decompose_to_bonds_and_rings(mol_smiles: str) -> Tuple[Set[str], Tuple[int, int]]:
    mol = Chem.MolFromSmiles(mol_smiles)
    Chem.Kekulize(mol)

    rings_mol = Chem.RWMol(mol)  # A molecule that eventually holds only rings

    # Extract bonds that aren't part of a ring
    bonds = set()

    rings_mol.BeginBatchEdit()
    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()
        if not bond.IsInRing():
            rings_mol.RemoveBond(u.GetIdx(), v.GetIdx())
            if not u.IsInRing():
                rings_mol.RemoveAtom(u.GetIdx())  # Atom already considered for the bond
            if not v.IsInRing():
                rings_mol.RemoveAtom(v.GetIdx())  # Atom already considered for the bond

            bond_mol = extract_mol_fragment(mol, {u.GetIdx(), v.GetIdx()})
            bond_smiles = Chem.MolToSmiles(bond_mol)
            bonds.add(bond_smiles)
    rings_mol.CommitBatchEdit()

    # By now, only rings should be left...
    # for atom in rings_mol.GetAtoms():
    #     assert atom.IsInRing()

    # Extract rings
    rings = set()
    for atom_indices in Chem.GetMolFrags(rings_mol):
        ring_mol = extract_mol_fragment(rings_mol, atom_indices)
        ring_smiles = Chem.MolToSmiles(ring_mol)
        rings.add(ring_smiles)

    # TODO check that the input molecule was fully decomposed (i.e. is extracting rings and bonds "enough")?

    return rings.union(bonds), (len(rings), len(bonds))


def generate_motif_vocabulary(training_set: pd.DataFrame, min_frequency=100) -> List[Tuple[int, str, bool]]:
    candidate_counter = Counter()
    motifs: Dict[str, int] = {}  # Motif -> Motif index
    vocab_rows: List[Tuple[int, str, bool]] = []  # Motif vocabulary (id, smiles, passed_frequency_test)

    def add_motif(smiles_: str, passed_frequency_test: bool):
        if smiles_ not in motifs:
            motif_idx = len(motifs)
            motifs[smiles_] = motif_idx
            vocab_rows.append((motif_idx, smiles_, passed_frequency_test))

    started_at = time()
    logged_at = time()
    for i, mol_smiles in enumerate(training_set['smiles']):
        candidates = extract_motif_candidates(mol_smiles)
        for cand_smiles in candidates:
            candidate_counter[cand_smiles] += 1
        if time() - logged_at > 1.0:
            time_left = (len(training_set) - (i + 1)) / ((i + 1) / (time() - started_at))
            logging.info(f"{i}/{len(training_set)} training set SMILES processed; "
                         f"Candidate motifs: {len(candidate_counter)}, "
                         f"Time left: {time_left:.1f}s")
            logged_at = time()

    # If the candidate isn't frequent enough in training set, split it into bonds and rings

    num_candidates = len(candidate_counter)
    for i, (cand_smiles, count) in enumerate(candidate_counter.items()):
        if count < min_frequency:
            parts, _ = decompose_to_bonds_and_rings(cand_smiles)
            for part_smiles in parts:
                add_motif(to_vocabulary_format(part_smiles), False)
        else:
            add_motif(to_vocabulary_format(cand_smiles), True)
        if i % 100 == 0:
            logging.info(f"{i}/{num_candidates} candidate motifs processed; Output motifs: {len(motifs)}")

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])  # Only for logging
    num_motifs = len(vocab_rows)
    num_passed_freq_test = len(df[df['passed_frequency_test']])
    logging.info(f"Motif vocabulary built; "
                 f"Size: {num_motifs}; "
                 f"Passed frequency test: {num_passed_freq_test}/{num_motifs}")
    return vocab_rows


def decompose_mol(mol_smiles: str, motif_vocab: MotifVocab) -> Set[str]:
    motifs = set({})
    candidates = extract_motif_candidates(mol_smiles)
    for candidate in candidates:
        if motif_vocab.has(candidate):
            motifs.add(candidate)
        else:
            parts, _ = decompose_to_bonds_and_rings(candidate)
            motifs.update(parts)
    return motifs


# TODO test that every motif SMILES in vocabulary is a canonical SMILES


def _main():
    training_set = pd.read_csv(ZINC_TRAINING_SET_CSV)
    vocab_rows = generate_motif_vocabulary(training_set, min_frequency=100)

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])
    df.set_index('id', inplace=True)
    df.to_csv(MOTIF_VOCAB_CSV)


if __name__ == "__main__":
    _main()
