import os
from os import path
import logging


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

MGRAPHS_PKL = path.join(DATASET_DIR, "mgraphs.pkl")

# ------------------------------------------------------------------------------------------------


def _on_import():
    import torch

    seed = 865002448

    # Setup logging
    logging.basicConfig(format='%(asctime)s [%(levelname)-5s] %(message)s', encoding='utf-8', level=logging.DEBUG)

    # TODO Probably doesn't handle multi-GPU scenario
    selected_dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.set_default_device(selected_dev)
    logging.debug(f"Default PyTorch device to: \"{selected_dev}\"")

    # Enable deterministic algorithms; see:
    # https://pytorch.org/docs/stable/notes/randomness.html
    logging.debug(f"Using pytorch's deterministic algorithms")
    # RuntimeError: index_reduce_cuda does not have a deterministic implementation
    # torch.use_deterministic_algorithms(True)

    # Use fixed random seed
    import random
    import numpy as np

    logging.info(f"Using fixed random seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


_on_import()
