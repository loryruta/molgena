from common import *
import logging
import tdc.generation
from os import path
import pandas as pd
from motif_vocab import MotifVocab
from gen_motif_vocab import decompose_mol
from motif_graph import construct_motif_graph, convert_motif_graph_to_smiles
from utils.misc_utils import *


def generate_dataset():
    if path.exists(DATASET_CSV):
        print("Dataset already exists, skipped")
        return

    print("Downloading dataset though TDC library...")

    if not path.exists(DATASET_DIR):
        os.mkdir(DATASET_DIR)

    dataset = tdc.generation.MolGen(name='ZINC').get_data()
    dataset.to_csv(DATASET_CSV, index_label='id')


def split_dataset():
    import argparse

    if not path.exists(DATASET_DIR):
        raise Exception("Dataset doesn't exist. Call `gen_dataset.py generate <dataset-dir>` first")

    parser = argparse.ArgumentParser()
    parser.add_argument("--training-frac", type=float, default=0.8)
    parser.add_argument("--validation-frac", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=None)

    args, _ = parser.parse_known_args()

    training_frac = args.training_frac
    validation_frac = args.validation_frac
    seed = args.seed

    if training_frac + validation_frac >= 1.0:
        raise Exception("Training + validation fractions mustn't exceed 1.0")

    print("Splitting dataset into training/validation/test set...")

    dataset = pd.read_csv(DATASET_CSV)

    training_set = dataset.sample(frac=training_frac, random_state=seed)
    dataset = dataset.drop(training_set.index)
    validation_set = dataset.sample(frac=args.validation_frac / (1.0 - training_frac), random_state=seed)
    test_set = dataset.drop(validation_set.index)

    test_frac = 1.0 - training_frac - validation_frac

    print(f"Dataset split:\n"
          f"  Training set {len(training_set)} ({training_frac * 100:.1f}%)\n"
          f"  Validation set {len(validation_set)} ({validation_frac * 100:.1f}%)\n"
          f"  Test set {len(test_set)} ({test_frac * 100:.1f}%)")

    print(f"Saving training set at: {TRAINING_CSV}")
    training_set.to_csv(TRAINING_CSV, index=False)

    print(f"Saving validation set at: {VALIDATION_CSV}")
    validation_set.to_csv(VALIDATION_CSV, index=False)

    print(f"Saving test set at: {TEST_CSV}")
    test_set.to_csv(TEST_CSV, index=False)


def _filter_dataset_file(dataset_filepath: str,
                         test_motif_vocab: bool = True,
                         test_motif_graph_identity: bool = True):
    if not test_motif_vocab and not test_motif_graph_identity:
        return

    options_str = ', '.join([
        'test_motif_vocab' if test_motif_vocab else '',
        'test_motif_graph_identity' if test_motif_vocab else ''
    ])
    print(f"Filtering dataset: {dataset_filepath} ({options_str})")

    dataset = pd.read_csv(dataset_filepath)
    num_samples = len(dataset)

    motif_vocab = MotifVocab.load()

    dt = stopwatch()

    mask = [False] * num_samples
    for i, (_, row) in enumerate(dataset.iterrows()):
        mol_smiles = row['smiles']

        # Keep molecules whose Motif decomposition is present in the vocabulary
        if test_motif_vocab:
            motifs = decompose_mol(mol_smiles, motif_vocab)
            in_ = [motif_vocab.at_smiles_or_null(motif) is not None for motif in motifs]
            if not all(in_):
                logging.debug(f"Discarding \"{mol_smiles}\": Motif decomposition not in vocabulary")
                continue

        # Keep molecules that have Motif graph identity
        if test_motif_graph_identity:
            try:
                smiles2, _ = convert_motif_graph_to_smiles(construct_motif_graph(mol_smiles, motif_vocab), motif_vocab)
                # if canon_smiles(smiles1)[0] != smiles2:
                #     logging.debug(f"Discarding \"{mol_smiles}\": Motif graph identity not working")
                #     continue
            except Exception as e:
                logging.debug(f"Discarding \"{mol_smiles}\": {str(e)}")
                continue

        mask[i] = True

        if dt() >= 5.0:
            logging.debug(f"{i + 1}/{num_samples} molecules processed...")
            dt = stopwatch()

    filtered_dataset = dataset[mask]

    print(f"Saving filtered dataset at: {dataset_filepath}")
    filtered_dataset.to_csv(dataset_filepath, index=False)


def filter_dataset():
    _filter_dataset_file(TRAINING_CSV, test_motif_graph_identity=False)
    _filter_dataset_file(VALIDATION_CSV, test_motif_vocab=True, test_motif_graph_identity=False)
    _filter_dataset_file(TEST_CSV, test_motif_vocab=True, test_motif_graph_identity=False)


def _main():
    import sys

    # Usage: gen_dataset.py <generate|split|filter> <dataset-dir>

    if len(sys.argv) < 2:
        print("Invalid syntax: gen_dataset.py <generate|split|filter>")
        exit(1)

    action = sys.argv[1]

    if action == 'generate':
        generate_dataset()
    elif action == 'split':
        split_dataset()
    elif action == 'filter':
        filter_dataset()
    else:
        print(f"Unrecognized action: \"{action}\"")
        exit(1)


if __name__ == "__main__":
    _main()
