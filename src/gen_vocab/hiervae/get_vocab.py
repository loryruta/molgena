import sys
import os
import argparse
from collections import Counter
from mol_graph import *
from rdkit import Chem
from multiprocessing import Pool
from common import *
import pandas as pd


def process(params):
    process_idx, data = params
    vocab = set()
    for smiles_idx, smiles in enumerate(data):
        hmol = MolGraph(smiles)
        for node, attr in hmol.mol_tree.nodes(data=True):
            smiles = attr['smiles']
            vocab.add(attr['label'])
            for i, s in attr['inter_label']:
                vocab.add((smiles, s))
        if smiles_idx % 500 == 0:
            print(f"Process {process_idx:>2}; Processed {smiles_idx + 1:>3}/{len(data)}")
    return vocab


def fragment_process(params):
    process_idx, data = params
    counter = Counter()
    for smiles_idx, smiles in enumerate(data):
        mol = get_mol(smiles)
        fragments = find_fragments(mol)
        for fsmiles, _ in fragments:
            counter[fsmiles] += 1
        if smiles_idx % 500 == 0:
            print(f"Process {process_idx:>2}; Processed {smiles_idx + 1:>3}/{len(data)}")
    return counter


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--min_frequency', type=int, default=100)
    parser.add_argument('--ncpu', type=int, default=os.cpu_count())
    args = parser.parse_args()

    # Load dataset into a unique list of SMILES
    print("Loading dataset...")

    df = pd.read_csv(DATASET_PATH)
    data = df['smiles'].unique().tolist()

    print(f"Loaded {len(data)} SMILES")

    # Split the work into batches for processes
    batch_size = len(data) // args.ncpu + 1
    batches = [data[i: i + batch_size] for i in range(0, len(data), batch_size)]

    pool = Pool(args.ncpu)
    counter_list = pool.map(fragment_process, enumerate(batches))
    counter = Counter()
    for cc in counter_list:
        counter += cc

    # Get a list of the most common Motifs (also called "fragments"),
    # that are the fragments that appear with a minimum frequency (e.g. 100) in the dataset
    fragments = [fragment for fragment, cnt in counter.most_common() if cnt >= args.min_frequency]
    MolGraph.load_fragments(fragments)

    pool = Pool(args.ncpu)
    vocab_list = pool.map(process, enumerate(batches))

    # Linearize the vocabulary and compute evaluate fragments frequency in the dataset
    fragments = set(fragments)

    def is_fragment(smiles: str) -> bool:
        mol = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))  # Dekekulize
        return mol in fragments

    vocab = [(x, y, is_fragment(x)) for vocab in vocab_list for x, y in vocab]
    vocab = list(set(vocab))

    # Create a dataframe for the vocabulary:
    # - src_mol  the molecule SMILES from which the Motif is originated
    # - motif    the Motif SMILES
    # - is_motif a boolean stating if the source molecule appears frequently enough
    vocab_df = pd.DataFrame(vocab, columns=['src_mol', 'motif', 'is_motif'])
    vocab_df.to_csv(VOCAB_PATH)
    num_motifs = len(vocab_df[vocab_df['is_motif']])
    print(f"Vocabulary saved; Path: {VOCAB_PATH}, Rows: {len(vocab_df)}, Motifs: {num_motifs}")


if __name__ == "__main__":
    main()
