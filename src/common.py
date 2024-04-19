import os
from os import path
import logging
import torch


def _check_required_env_vars():
    for env_var in ['DATASET_DIR']:
        if env_var not in os.environ:
            raise Exception(f"Missing required env variable: \"{env_var}\"")


_check_required_env_vars()

# ------------------------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------------------------

DATASET_DIR = os.environ["DATASET_DIR"]

DATASET_CSV = path.join(DATASET_DIR, "dataset.csv")
TRAINING_CSV = path.join(DATASET_DIR, "training.csv")
VALIDATION_CSV = path.join(DATASET_DIR, "validation.csv")
TEST_CSV = path.join(DATASET_DIR, "test.csv")

MOTIF_VOCAB_CSV = path.join(DATASET_DIR, "motif_vocab.csv")

TRAINING_MOTIF_GRAPHS_PKL = path.join(DATASET_DIR, "training_motif_graphs.pkl")
VALIDATION_MOTIF_GRAPHS_PKL = path.join(DATASET_DIR, "validation_motif_graphs.pkl")
TEST_MOTIF_GRAPHS_PKL = path.join(DATASET_DIR, "test_motif_graphs.pkl")


# ------------------------------------------------------------------------------------------------


def _on_import():
    # Setup logging
    logging.basicConfig(format='%(asctime)s [%(levelname)-5s] %(message)s', encoding='utf-8', level=logging.DEBUG)

    # TODO Probably doesn't handle multi-GPU scenario
    selected_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(selected_dev)
    logging.debug(f"Default PyTorch device to: \"{selected_dev}\"")


_on_import()
