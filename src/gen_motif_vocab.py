from common import *
from rdkit import Chem
import pandas as pd
from typing import *
from collections import Counter
import logging
from time import time
from utils.chem_utils import *


def extract_motif_candidates(mol_smiles: str) -> Set[str]:
    """ Extracts motif candidates from the input molecular graph, according to the technique described in:
     "Hierarchical Generation of Molecular Graphs using Structural Motifs", Appendix A
     """

    # HierVAE code reference:
    # https://github.com/wengong-jin/hgraph2graph/blob/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767/polymers/poly_hgraph/chemutils.py#L44

    # We perform the splitting of the input molecular graph using RWMol.
    # This class permits to remove atomic bonds while maintaining chemical validity (we can't use networkx for that!)

    mol = mol_from_smiles(mol_smiles)
    new_mol = Chem.RWMol(mol)  # The molecule on which we work for splitting

    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()

        if bond.IsInRing():
            continue

        if u.IsInRing() and v.IsInRing():
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif u.IsInRing() and v.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(u))
            new_mol.AddBond(new_idx, v.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif v.IsInRing() and u.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(v))
            new_mol.AddBond(u.GetIdx(), new_idx, bond.GetBondType())
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())

    candidates = set()
    for atom_indices in Chem.GetMolFrags(new_mol):
        frag_mol = extract_mol_fragment(new_mol, atom_indices)
        candidates.add(Chem.MolToSmiles(frag_mol))
    return candidates


def decompose_to_bonds_and_rings(mol_smiles: str) -> Tuple[Set[str], Tuple[int, int]]:
    mol = mol_from_smiles(mol_smiles)

    bonds = set()
    rings_mol = Chem.RWMol(mol)

    # Extract all bonds that are not part of any ring
    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()
        if not bond.IsInRing():
            rings_mol.RemoveBond(u.GetIdx(), v.GetIdx())

            frag_mol = extract_mol_fragment(mol, {u.GetIdx(), v.GetIdx()})
            bonds.add(Chem.MolToSmiles(frag_mol))

    # Remove atoms that were left disconnected after bond removal
    rings_mol.BeginBatchEdit()
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0:
            rings_mol.RemoveAtom(atom.GetIdx())
    rings_mol.CommitBatchEdit()

    # Now only rings should be remained, create a fragment for each of them
    rings = set()
    for atom_indices in Chem.GetSymmSSSR(rings_mol):
        ring_mol = extract_mol_fragment(rings_mol, atom_indices)
        rings.add(Chem.MolToSmiles(ring_mol))

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
        for candidate_smiles in candidates:
            candidate_counter[candidate_smiles] += 1
        if time() - logged_at > 1.0:
            time_left = (len(training_set) - (i + 1)) / ((i + 1) / (time() - started_at))
            logging.info(f"{i}/{len(training_set)} training set SMILES processed; "
                         f"Candidate motifs: {len(candidate_counter)}, "
                         f"Time left: {time_left:.1f}s")
            logged_at = time()

    # If the candidate isn't frequent enough in training set, split it into bonds and rings

    num_candidates = len(candidate_counter)
    for i, (candidate_smiles, count) in enumerate(candidate_counter.items()):
        if count < min_frequency:
            parts_smiles, _ = decompose_to_bonds_and_rings(candidate_smiles)
            for part_smiles in parts_smiles:
                add_motif(part_smiles, False)
        else:
            add_motif(candidate_smiles, True)
        if i % 100 == 0:
            logging.info(f"{i}/{num_candidates} candidate motifs processed; Output motifs: {len(motifs)}")

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])  # Only for logging
    num_motifs = len(vocab_rows)
    num_passed_freq_test = len(df[df['passed_frequency_test']])
    logging.info(f"Motif vocabulary built; "
                 f"Size: {num_motifs}; "
                 f"Passed frequency test: {num_passed_freq_test}/{num_motifs}")
    return vocab_rows


def generate_and_save_motif_vocabulary():
    training_set = pd.read_csv(DATASET_PATH)
    vocab_rows = generate_motif_vocabulary(training_set, min_frequency=100)

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])
    df.set_index('id', inplace=True)
    df.to_csv(MOTIF_VOCAB_CSV)


def _main():
    generate_and_save_motif_vocabulary()


if __name__ == "__main__":
    _main()
