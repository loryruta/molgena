from common import *
import logging
import tdc.generation
from os import path
from typing import *
import pandas as pd


def split_dataset(dataset: pd.DataFrame, training_frac: float, validation_frac: float,
                  random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    training_set = dataset.sample(frac=training_frac, random_state=random_state)
    dataset = dataset.drop(training_set.index)
    validation_set = dataset.sample(frac=validation_frac / (1.0 - training_frac), random_state=random_state)
    test_set = dataset.drop(validation_set.index)
    return training_set, validation_set, test_set


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

    training_set.to_csv(ZINC_TRAINING_SET_CSV)
    validation_set.to_csv(ZINC_VALIDATION_SET_CSV)
    test_set.to_csv(ZINC_TEST_SET_CSV)

    logging.info(f"Dataset split; "
                 f"Training set {len(training_set)} ({TRAINING_FRAC * 100:.1f}%), "
                 f"Validation set {len(validation_set)} ({VALIDATION_FRAC * 100:.1f}%), "
                 f"Test set {len(test_set)} ({TEST_FRAC * 100:.1f}%)")


if __name__ == "__main__":
    _main()
