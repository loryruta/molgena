from os import path

BASE_DIR = path.abspath(path.dirname(__file__))

DATASET_DIR = path.join(BASE_DIR, "data")
DATASET_PATH = path.join(DATASET_DIR, "zinc.csv")
VOCAB_PATH = path.join(DATASET_DIR, "vocab.csv")
