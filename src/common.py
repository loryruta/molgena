from os import path
from concurrent.futures import ThreadPoolExecutor
import logging
import torch

# ------------------------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------------------------

BASE_DIR = path.abspath(path.dirname(__file__))

DATA_DIR = path.join(BASE_DIR, "data")
DATASET_PATH = path.join(DATA_DIR, "zinc.csv")
VOCAB_PATH = path.join(DATA_DIR, "vocab.csv")  # TODO old, rename
MOTIF_VOCAB_CSV = path.join(DATA_DIR, "motif_vocab.csv")
MOTIF_GRAPH_PATH = path.join(DATA_DIR, "motif_graph.gml")

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
    logging.basicConfig(format='%(asctime)s %(level)s: %(message)s', encoding='utf-8', level=logging.DEBUG)


_on_import()
