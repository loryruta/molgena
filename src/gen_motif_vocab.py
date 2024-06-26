from common import *
from collections import Counter
import pandas as pd
from time import time
from utils.chem_utils import *
from motif_vocab import MotifVocab


def to_vocabulary_format(mol_smiles: str):
    mol_smiles = clear_atommap(mol_smiles)
    mol_smiles = Chem.CanonSmiles(mol_smiles)
    return mol_smiles


def extract_motif_candidates(mol_smiles: str) -> Set[str]:
    """ Extracts Motif candidates from the input molecular graph by just _breaking bonds_.
    Based on:
      "Hierarchical Generation of Molecular Graphs using Structural Motifs" Jin et al., 2020, Appendix A
     """

    mol = Chem.MolFromSmiles(mol_smiles)
    Chem.Kekulize(mol)

    new_mol = Chem.RWMol(mol)

    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()

        if bond.IsInRing():  # Both atoms in the same ring
            continue

        if u.IsInRing() and v.IsInRing():  # u and v in two different rings
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif u.IsInRing() and v.GetDegree() > 1:  # u in ring and v not a leaf atom
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif v.IsInRing() and u.GetDegree() > 1:  # v in ring and u not a leaf atom
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())

    candidates = set()
    for atom_indices in Chem.GetMolFrags(new_mol):
        cand_mol = extract_mol_fragment(new_mol, atom_indices)
        cand_smiles = Chem.MolToSmiles(cand_mol)
        candidates.add(cand_smiles)
    return candidates


def decompose_motif_candidate(mol_smiles: str) -> Set[str]:
    """ If the Motif candidate is considered not to be "frequent enough", this function is used to decompose it into
      smaller parts. """

    mol = Chem.MolFromSmiles(mol_smiles)
    Chem.Kekulize(mol)

    new_mol = Chem.RWMol(mol)
    for bond in mol.GetBonds():
        # Never break rings!
        # Rings should always come from the vocabulary, we don't want the model to learn how to do them
        if bond.IsInRing():
            continue
        new_mol.RemoveBond(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx())

    parts = set()
    for atom_indices in Chem.GetMolFrags(new_mol):
        frag_mol = extract_mol_fragment(new_mol, atom_indices)
        frag_smiles = Chem.MolToSmiles(frag_mol)
        parts.add(frag_smiles)
    return parts


def generate_motif_vocabulary(training_set: pd.DataFrame, min_frequency=100) -> List[Tuple[int, str, bool]]:
    print(f"Generating motif vocabulary on {len(training_set)} (min frequency: {min_frequency})...")

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
            print(f"{i}/{len(training_set)} training set SMILES processed; "
                  f"Candidate motifs: {len(candidate_counter)}, "
                  f"Time left: {time_left:.1f}s")
            logged_at = time()

    # If the candidate isn't frequent enough in training set, split it into bonds and rings

    num_candidates = len(candidate_counter)
    for i, (cand_smiles, count) in enumerate(candidate_counter.items()):
        if count < min_frequency:
            parts = decompose_motif_candidate(cand_smiles)
            for part_smiles in parts:
                add_motif(to_vocabulary_format(part_smiles), False)
        else:
            add_motif(to_vocabulary_format(cand_smiles), True)
        if i % 100 == 0:
            print(f"{i}/{num_candidates} candidate motifs processed; Output motifs: {len(motifs)}")

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])  # Only for logging
    num_motifs = len(vocab_rows)
    num_passed_freq_test = len(df[df['passed_frequency_test']])
    print(f"Motif vocabulary built; "
          f"Size: {num_motifs}; "
          f"Passed frequency test: {num_passed_freq_test}/{num_motifs}")
    return vocab_rows


def decompose_mol(mol_smiles: str, motif_vocab: MotifVocab) -> Set[str]:
    motifs = set({})
    candidates = extract_motif_candidates(mol_smiles)
    for candidate in candidates:
        if motif_vocab.has_smiles(candidate):
            motifs.add(candidate)
        else:
            parts = decompose_motif_candidate(candidate)
            motifs.update(parts)
    return motifs


def _main():
    if path.exists(MOTIF_VOCAB_CSV):
        print("Motif vocabulary already exists, skipping")
        return

    training_set = pd.read_csv(TRAINING_CSV)
    vocab_rows = generate_motif_vocabulary(training_set, min_frequency=100)

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])
    df.set_index('id', inplace=True)
    df.to_csv(MOTIF_VOCAB_CSV)


if __name__ == "__main__":
    _main()
