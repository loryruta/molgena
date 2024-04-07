from common import *

import pandas as pd
from torch.utils.data import Dataset


class MolDataset(Dataset):
    """ A dataset stored in a pandas DataFrame with a "smiles" column.
    SMILES stored are not canonical and not said to be kekulized.
    """

    def __init__(self, path_: str):
        self.df = pd.read_csv(path_)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i: int):
        mol_smiles = self.df.iloc[i]['smiles']
        return i, mol_smiles


class ZincDataset(MolDataset):
    def __init__(self, path_: str):  # Private
        super().__init__(path_)

    @staticmethod
    def all():
        return ZincDataset(ZINC_DATASET_CSV)

    @staticmethod
    def training_set():
        return ZincDataset(ZINC_TRAINING_SET_CSV)

    @staticmethod
    def validation_set():
        return ZincDataset(ZINC_VALIDATION_SET_CSV)

    @staticmethod
    def test_set():
        return ZincDataset(ZINC_TEST_SET_CSV)


def _main():
    training_set = ZincDataset.training_set()
    print(training_set[0])


if __name__ == "__main__":
    _main()
