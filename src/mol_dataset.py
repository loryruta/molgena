from common import *
from torch.utils.data import Dataset
from os import path
import pandas as pd
import networkx as nx
from typing import *
from mol_graph import create_mol_graph_from_smiles
from construct_motif_graph import construct_motif_graph
from tensor_graph import TensorGraph

motif_vocab = pd.read_csv(MOTIF_VOCAB_CSV, index_col=['smiles'])  # TODO put it somewhere else (e.g. Dataset member)


class MolDataset(Dataset):
    """ A dataset stored in a pandas DataFrame with a "smiles" column. """

    def __init__(self, path_: str):
        self._df = pd.read_csv(path_)

    def __len__(self):
        return len(self._df)

    def __getitem__(self, i: int) -> Tuple[str, TensorGraph, nx.Graph]:
        mol_smiles = self._df.iloc[i]['smiles']
        print(i, mol_smiles)
        return (
            mol_smiles,
            create_mol_graph_from_smiles(mol_smiles),
            construct_motif_graph(mol_smiles, motif_vocab)
        )


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
