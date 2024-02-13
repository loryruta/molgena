# Script taken from wengong-jin/hgraph2hgraph (official HierVAE repository):
# https://github.com/wengong-jin/hgraph2graph/blob/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767/polymers/get_vocab.py

import os
import argparse
from mol_graph import MolGraph
from multiprocessing import Pool
from common import *
import pandas as pd


def process(params):
    i, data = params
    vocab = set()
    for j, smiles in enumerate(data):
        hmol = MolGraph(smiles)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add(attr['label'])
            for i, s in attr['inter_label']:
                vocab.add((smiles, s))
        if j % 500 == 0:
            print(f"Process {i:>2}; Processed {j + 1:>3}/{len(data)}")
    return vocab


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ncpu', type=int, default=os.cpu_count())
    args = parser.parse_args()

    print("Loading dataset...")
    df = pd.read_csv(DATASET_PATH)

    data = []
    for i, row in df.iterrows():
        data.append(row['smiles'])
        if i % 50000 == 0:
            print(f"Loaded {i} smiles...")

    data = list(set(data))
    print(f"Read {len(data)} unique smiles...")

    batch_size = len(data) // args.ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, enumerate(batches))
    vocab = [(x, y) for vocab in vocab_list for x, y in vocab]
    vocab = list(set(vocab))

    # Create a dataframe for the vocabulary:
    # - x is the molecule SMILES from which the Motif is originated
    # - y is the Motif SMILES
    vocab_df = pd.DataFrame(vocab, columns=['x', 'y'])
    vocab_df.to_csv(VOCAB_PATH)
    print(f"Vocabulary saved to: {VOCAB_PATH} ({len(vocab)} entries)")


if __name__ == "__main__":
    main()
