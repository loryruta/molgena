import os
from os import path
from concurrent.futures import ThreadPoolExecutor
import logging
import torch

# ------------------------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------------------------

BASE_DIR = path.abspath(path.dirname(__file__))

DATA_DIR = path.join(BASE_DIR, "data")

ZINC_DATASET_CSV = path.join(DATA_DIR, "zinc.csv")
ZINC_TRAINING_SET_CSV = path.join(DATA_DIR, "zinc_training_set.csv")
ZINC_VALIDATION_SET_CSV = path.join(DATA_DIR, "zinc_validation_set.csv")
ZINC_TEST_SET_CSV = path.join(DATA_DIR, "zinc_test_set.csv")

DATASET_PATH = ZINC_DATASET_CSV  # TODO old name, delete

MOTIF_VOCAB_CSV = path.join(DATA_DIR, "motif_vocab.csv")
MOTIF_GRAPHS_PKL = path.join(DATA_DIR, "motif_graphs.pkl")

# ------------------------------------------------------------------------------------------------

# process_pool = multiprocess.Pool()
threadpool = ThreadPoolExecutor(max_workers=16)
# fast_iterator = FastIterator(process_pool, threadpool)


def _on_import():
    # TODO Probably doesn't handle multi-GPU scenario
    selected_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(selected_dev)
    print(f"Default PyTorch device to: \"{selected_dev}\"")

    # Setup logging
    logging.basicConfig(format='%(asctime)s %(levelname)s: %(message)s', encoding='utf-8', level=logging.DEBUG)


_on_import()
