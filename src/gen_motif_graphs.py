from common import *
import pandas as pd
import pickle
from time import time
from motif_graph import construct_motif_graph
from motif_vocab import MotifVocab


def _construct_and_save_motif_graphs(dataset_filepath: str, pkl_filepath: str, force: bool) -> bool:
    """ Constructs motif graphs for all dataset samples. Saves the final result to a .pkl file. """

    if not path.exists(dataset_filepath):
        raise Exception(f"Dataset not found: \"{dataset_filepath}\"")

    if path.exists(pkl_filepath) and not force:
        print(f"File already exists \"{pkl_filepath}\", skipping!")
        return False

    mol_dataset = pd.read_csv(dataset_filepath)
    num_samples = len(mol_dataset)

    motif_vocab = MotifVocab.load()

    started_at = time()
    logged_at = time()

    print(f"Constructing motif graphs for {num_samples} samples...")

    motif_graphs = []
    for mol_id, mol_smiles in mol_dataset['smiles'].items():
        motif_graph = construct_motif_graph(mol_smiles, motif_vocab)
        motif_graphs.append(motif_graph)

        if time() - logged_at > 5.0:
            num_left = num_samples - (mol_id + 1)
            time_left = num_left / ((mol_id + 1) / (time() - started_at))
            logging.info(f"Constructed motif graph for {mol_id + 1}/{num_samples} molecules; "
                         f"Time left: {time_left:.1f}s, "
                         f"Directory size: ")
            logged_at = time()

    print("Constructed motif graphs for all dataset molecules!")

    print(f"Saving .pkl file: \"{pkl_filepath}\"")

    with open(pkl_filepath, 'wb') as file:
        pickle.dump(motif_graphs, file)

    print(f"Done!")

    return True


def _main():
    print(f"Generating motif graphs...")

    _construct_and_save_motif_graphs(TRAINING_CSV, TRAINING_MOTIF_GRAPHS_PKL, force=False)
    _construct_and_save_motif_graphs(VALIDATION_CSV, VALIDATION_MOTIF_GRAPHS_PKL, force=False)
    _construct_and_save_motif_graphs(TEST_CSV, TEST_MOTIF_GRAPHS_PKL, force=False)


if __name__ == "__main__":
    _main()
