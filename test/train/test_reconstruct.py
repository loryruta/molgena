from common import *
import logging
import pytest
from mol_dataset import ZincDataset
from motif_graph.cache import MgraphCache
from train.reconstruct.annotations import Annotator
from utils.misc_utils import *


@pytest.mark.skip(reason="Slow")
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


def test_create_batched_annotations_perf():
    dataset = ZincDataset.training_set()

    annotator = Annotator()
    MgraphCache.instance()  # Preload mgraph cache

    for batch_size in [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]:
        mol_smiles_list = dataset.df.sample(n=batch_size)['smiles'].tolist()

        stopwatch_ = stopwatch_str()
        annotator.create_batched_annotations(mol_smiles_list)
        logging.info(f"Annotations for {batch_size} molecules created in {stopwatch_()}")
