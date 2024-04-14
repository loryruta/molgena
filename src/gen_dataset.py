from common import *
import logging
import tdc.generation
from os import path
from typing import *
import pandas as pd
from motif_vocab import MotifVocab
from gen_motif_vocab import decompose_mol
from construct_motif_graph import construct_motif_graph, convert_motif_graph_to_smiles
from utils.chem_utils import *
from utils.misc_utils import *


def split_dataset(dataset: pd.DataFrame,
                  training_frac: float,
                  validation_frac: float,
                  random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_set = dataset.sample(frac=training_frac, random_state=random_state)
    dataset = dataset.drop(training_set.index)
    validation_set = dataset.sample(frac=validation_frac / (1.0 - training_frac), random_state=random_state)
    test_set = dataset.drop(validation_set.index)
    return training_set, validation_set, test_set


def filter_mol_dataset(mol_dataset: pd.DataFrame, **kwargs) -> pd.DataFrame:
    """ Filters SMILES for which a Motif graph can be aggregated and disaggregated into the initial SMILES. """

    motif_vocab = MotifVocab.load()

    dt = stopwatch()

    mask = [False] * len(mol_dataset)
    for i, (_, row) in enumerate(mol_dataset.iterrows()):
        mol_smiles = row['smiles']

        # Keep molecules whose Motif decomposition is present in the vocabulary
        if ('in_motif_vocabulary' in kwargs) and kwargs['in_motif_vocabulary']:
            motifs = decompose_mol(mol_smiles, motif_vocab)
            in_ = [motif_vocab.at_smiles_or_null(motif) is not None for motif in motifs]
            if not all(in_):
                logging.debug(f"Discarding \"{mol_smiles}\": Motif decomposition not in vocabulary")
                continue

        # Keep molecules that have Motif graph identity
        if ('run_motif_graph_identity' in kwargs) and kwargs['run_motif_graph_identity']:
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
            logging.debug(f"{i + 1}/{len(mol_dataset)} molecules processed...")
            dt = stopwatch()

    return mol_dataset[mask]


def _main():
    RANDOM_STATE = 23
    TRAINING_FRAC = 0.8
    VALIDATION_FRAC = 0.1
    TEST_FRAC = 1.0 - TRAINING_FRAC - VALIDATION_FRAC

    dataset = tdc.generation.MolGen(name='ZINC').get_data()  # Also TDC provides splits (are those better?)
    dataset.to_csv(ZINC_DATASET_CSV)

    logging.info(f"Dataset showcase {len(dataset)} entries")

    training_set, validation_set, test_set = split_dataset(dataset,
                                                           training_frac=TRAINING_FRAC,
                                                           validation_frac=VALIDATION_FRAC,
                                                           random_state=RANDOM_STATE)
    logging.info(f"Dataset split; "
                 f"Training set {len(training_set)} ({TRAINING_FRAC * 100:.1f}%), "
                 f"Validation set {len(validation_set)} ({VALIDATION_FRAC * 100:.1f}%), "
                 f"Test set {len(test_set)} ({TEST_FRAC * 100:.1f}%)")

    logging.info(f"Saving unfiltered splits...")
    training_set.to_csv(ZINC_TRAINING_SET_CSV)
    validation_set.to_csv(ZINC_VALIDATION_SET_CSV)
    test_set.to_csv(ZINC_TEST_SET_CSV)

    # Filter training set
    logging.info(f"Filtering training set...")
    initial_size = len(training_set)
    training_set = filter_mol_dataset(training_set, run_motif_graph_identity=True)
    logging.info(f"Filtered training set; "
                 f"Size: {initial_size} -> {len(training_set)}, "
                 f"Frac: {len(training_set) / initial_size * 100:.2f}")

    # Filter validation set
    logging.info(f"Filtering validation set...")
    initial_size = len(validation_set)
    validation_set = filter_mol_dataset(validation_set, in_motif_vocabulary=True, run_motif_graph_identity=True)
    logging.info(f"Filtered validation set; "
                 f"Size: {initial_size} -> {len(validation_set)}, "
                 f"Frac: {len(validation_set) / initial_size * 100:.2f}")

    # Filter test set
    logging.info(f"Filtering test set...")
    initial_size = len(test_set)
    test_set = filter_mol_dataset(test_set, in_motif_vocabulary=True, run_motif_graph_identity=True)
    logging.info(f"Filtered test set; "
                 f"Size: {initial_size} -> {len(test_set)}, "
                 f"Frac: {len(test_set) / initial_size * 100:.2f}")

    logging.info(f"Saving filtered splits...")
    training_set.to_csv(ZINC_TRAINING_SET_CSV)
    validation_set.to_csv(ZINC_VALIDATION_SET_CSV)
    test_set.to_csv(ZINC_TEST_SET_CSV)


if __name__ == "__main__":
    _main()
