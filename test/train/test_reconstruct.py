from common import *
import logging
from mol_dataset import ZincDataset
from train.annotations import Annotator
from utils.misc_utils import stopwatch


def test_annotate():
    dataset = ZincDataset.training_set()
    dataset_len = len(dataset)

    annotator = Annotator()

    stopwatch_ = stopwatch()

    num_pmol_empty = 0
    num_pmol_full = 0
    num_pmol_nodes_sum = 0.  # Used to log the pmol node count average

    for i, mol_smiles in dataset:
        labels = annotator.annotate(mol_smiles)

        num_pmol_nodes_sum += len(labels.mgraph_subgraph_indices)

        if labels.is_empty:
            num_pmol_empty += 1
        elif labels.is_full:
            num_pmol_full += 1

        if stopwatch_() > 2.:
            logging.debug(f"Annotated {i}/{dataset_len} molecules; "
                          f"{num_pmol_empty} empty/{num_pmol_full} full/{i}, "
                          f"Avg nodes: {num_pmol_nodes_sum / i:.1f}")
            stopwatch_ = stopwatch()
