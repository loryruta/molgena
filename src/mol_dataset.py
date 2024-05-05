from common import *

import pandas as pd
from torch.utils.data import Dataset


class MolDataset(Dataset):
    """ A dataset stored in a pandas DataFrame with a "smiles" column.
    SMILES stored are not canonical and not said to be kekulized.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        mol_smiles = self.df.iloc[i]['smiles']
        return i, mol_smiles


class ZincDataset(MolDataset):
    def __init__(self, df: pd.DataFrame):
        super().__init__(df)

    @staticmethod
    def training_set():
        return ZincDataset(pd.read_csv(TRAINING_CSV))

    @staticmethod
    def validation_set():
        return ZincDataset(pd.read_csv(VALIDATION_CSV))

    @staticmethod
    def test_set():
        return ZincDataset(pd.read_csv(TEST_CSV))

    @staticmethod
    def all():
        """ Returns the union of training/validation and test sets. """
        return ZincDataset(pd.concat([
            ZincDataset.training_set().df,
            ZincDataset.validation_set().df,
            ZincDataset.test_set().df
        ]))


def _main():
    training_set = ZincDataset.training_set()
    print(training_set[0])


if __name__ == "__main__":
    _main()
