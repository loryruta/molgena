import pandas as pd
from common import *


class MotifVocab(pd.DataFrame):
    """ A class representing the Motif vocabulary dataframe. """

    def __iter__(self):
        for i, row in self.iterrows():
            yield row['motif']

    @staticmethod
    def load():
        df = pd.read_csv(VOCAB_PATH)
        return MotifVocab(df[df['is_motif']]['motif'])


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
