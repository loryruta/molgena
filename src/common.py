from os import path
import multiprocess
from concurrent.futures import ThreadPoolExecutor
from fast_iter import FastIterator

# ------------------------------------------------------------------------------------------------
# Configuration constants
# ------------------------------------------------------------------------------------------------

BASE_DIR = path.abspath(path.dirname(__file__))

DATA_DIR = path.join(BASE_DIR, "data")
DATASET_PATH = path.join(DATA_DIR, "zinc.csv")
VOCAB_PATH = path.join(DATA_DIR, "vocab.csv")
MOTIF_GRAPH_PATH = path.join(DATA_DIR, "motif_graph.gml")

# ------------------------------------------------------------------------------------------------

# process_pool = multiprocess.Pool()
threadpool = ThreadPoolExecutor(max_workers=16)
# fast_iterator = FastIterator(process_pool, threadpool)
