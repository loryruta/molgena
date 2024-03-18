import pandas as pd
from rdkit import Chem
from common import *
from utils.chem_utils import *


class MotifVocab:
    """ A class representing the Motif vocabulary dataframe.
    SMILES stored are canonical SMILES (not kekulized, kekulization is done on loading).
    """

    def __init__(self, path_: str):
        self._df = pd.read_csv(path_, index_col=['smiles'])

    @staticmethod
    def _canon_format(smiles: str) -> str:
        return Chem.CanonSmiles(clear_atommap(smiles))

    def has(self, smiles: str) -> bool:
        return self._canon_format(smiles) in self._df.index

    def __iter__(self):
        for _, row in self._df.iterrows():
            yield row['smiles']

    def get_or_null(self, smiles: str):
        smiles = self._canon_format(smiles)
        return self._df.loc[smiles] if smiles in self._df.index else None

    def __getitem__(self, smiles: str):
        smiles = self._canon_format(smiles)
        if smiles not in self._df.index:
            raise KeyError(f"SMILES \"{smiles}\" not in motif vocabulary")
        return self._df.loc[smiles]

    @staticmethod
    def load():
        return MotifVocab(MOTIF_VOCAB_CSV)


def _main():
    from rdkit import Chem

    vocab = MotifVocab.load()
    motifs = list(vocab)
    print(f"Motif vocabulary has {len(motifs)} entries")

    mols = [Chem.MolFromSmiles(motif) for motif in motifs]
    num_atoms = [mol.GetNumAtoms() for mol in mols]
    avg_atoms = sum(num_atoms) / len(num_atoms)
    print(f"Min atoms: {min(num_atoms)}, Max atoms: {max(num_atoms)}, Avg atoms: {avg_atoms:.3f}")


if __name__ == '__main__':
    _main()
