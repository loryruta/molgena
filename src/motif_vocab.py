import pandas as pd
from rdkit import Chem
from common import *
from utils.chem_utils import *


class MotifVocab:
    """ A class representing the Motif vocabulary dataframe.
    SMILES stored are canonical SMILES (not kekulized, kekulization is done on loading).
    """

    df_id: pd.DataFrame  # Dataframe indexed by `id` column
    df_smiles: pd.DataFrame  # Dataframe indexed by `smiles` column

    def __init__(self, path_: str):
        self.df_id = pd.read_csv(path_, index_col=['id'])
        self.df_smiles = pd.read_csv(path_, index_col=['smiles'])

    def has_smiles(self, smiles: str) -> bool:
        smiles, _ = canon_smiles(smiles)
        return smiles in self.df_smiles.index

    def at_id(self, id_: int):
        """ Gets the row at the corresponding ID or throws. """
        if id_ not in self.df_id.index:
            raise KeyError(f"Motif ID {id_} not found in motif vocabulary")
        return self.df_id.loc[id_]

    def at_smiles(self, smiles: str):
        """ Gets the row at the corresponding SMILES or throws. """
        smiles, _ = canon_smiles(smiles)
        if smiles not in self.df_smiles.index:
            raise KeyError(f"SMILES \"{smiles}\" not in motif vocabulary")
        return self.df_smiles.loc[smiles]

    def at_smiles_or_null(self, smiles: str):
        """ Gets the row at the corresponding SMILES. Returns null if not found. """
        smiles, _ = canon_smiles(smiles)
        return self.df_smiles.loc[smiles] if smiles in self.df_smiles.index else None

    # TODO utility function that given ID returns SMILES (no at_id(id)['smiles'])

    def end_motif_id(self) -> int:
        return len(self)

    def __iter__(self):
        for _, row in self.df_id.iterrows():
            yield row['smiles']

    def __len__(self):
        return len(self.df_id)

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
