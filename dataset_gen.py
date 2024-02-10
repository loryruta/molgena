import rdkit
from rdkit import Chem
from rdkit.Chem import Crippen, Descriptors, AllChem
from tdc.generation import MolGen
from concurrent.futures import ThreadPoolExecutor
from math import ceil
import sascorer
from time import time
import multiprocess
from os import path
import pandas as pd
from fast_iter import FastIterator
from tabulate import tabulate
import numpy as np


DATASET_PATH = 'data/zinc.csv'
DATASET_EXPECTED_SIZE = 249456


def is_dataset_valid():
    if not path.exists(DATASET_PATH):
        return (False, None,)

    df = pd.read_csv(DATASET_PATH)

    is_valid = True
    is_valid &= set(df.columns) != {'smile', 'logp', 'mw', 'qed', 'sa'}
    is_valid &= len(df) != DATASET_EXPECTED_SIZE
    return (is_valid, df,)


def generate_dataset(fast_iterator: FastIterator):
    # Start from the ZINC dataset from TDC
    dataset = MolGen(name = 'ZINC')
    df = dataset.get_data()

    df.rename({'smiles': 'smile'}, axis=1, inplace=True)
    df['logp'] = None
    df['mw'] = None
    df['qed'] = None
    df['sa'] = None

    def calc_mol_props(i: int):
        # Function called in multiprocess manner
        mol_smile = df.at[i, 'smile']  # TODO Require dt, does this mean dataframe is copied to every process?
        mol = Chem.MolFromSmiles(mol_smile)

        logp = Crippen.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        qed = Chem.QED.qed(mol)
        sa = sascorer.calculateScore(mol)
        return (logp, mw, qed, sa,)

    def apply_mol_props(i, mol_props):  # Function called in multithreaded manner
        logp, mw, qed, sa = mol_props
        df.at[i, 'logp'] = logp
        df.at[i, 'mw'] = mw
        df.at[i, 'qed'] = qed
        df.at[i, 'sa'] = sa

    fast_iterator.iterate(len(df), calc_mol_props, apply_mol_props)

    print(f"Saving dataset to {DATASET_PATH}...")
    df.to_csv(DATASET_PATH)
    return df


def print_similarities(df, fast_iterator: FastIterator, seed: int = 0):
    # Sample a few molecules from the dataset and calculate similarity with all other molecules
    sampled_smiles = df.sample(n=100, random_state=seed)

    for i, row in sampled_smiles.iterrows():
        sample_smile = row['smile']
        sample_mol = Chem.MolFromSmiles(sample_smile)
        sample_mol_fingerprint = Chem.RDKFingerprint(sample_mol)

        def calc_tanimoto_similarity(i: int):
            smile = df.at[i, 'smile']
            mol = Chem.MolFromSmiles(smile)
            mol_fingerprint = Chem.RDKFingerprint(mol)
            similarity = rdkit.DataStructs.TanimotoSimilarity(sample_mol_fingerprint, mol_fingerprint)
            return similarity

        def apply_tanimoto_similarity(j, similarity):
            if j != i:
                return [j, similarity]
            else:
                return None  # Don't return similarity for the sample molecule itself

        similarities = fast_iterator.iterate(len(df), calc_tanimoto_similarity, apply_tanimoto_similarity)
        similarities = np.array(similarities)
        min_similarity = similarities[np.argmin(similarities[:,1])]
        max_similarity = similarities[np.argmax(similarities[:,1])]

        print(f"Min similarity at {int(min_similarity[0])}: {min_similarity[1]}")
        print(f"Max similarity at {int(max_similarity[0])}: {max_similarity[1]}")

        print(tabulate([
            ["Sample index", i],
            ["Sample molecule", sample_smile],
            ["Min similarity", min_similarity[1], df.at[int(min_similarity[0]), 'smile']],
            ["Max similarity", max_similarity[1], df.at[int(max_similarity[0]), 'smile']],
        ]))


if __name__ == "__main__":
    process_pool = multiprocess.Pool()  # Number of processes is CPU count
    thread_pool = ThreadPoolExecutor(max_workers=16)

    fast_iterator = FastIterator(process_pool, thread_pool)

    (is_valid, df) = is_dataset_valid()
    if not is_valid:
        df = generate_dataset(fast_iterator)

    # Print the range (min/max/avg) of the chemical properties
    max_logp = df.iloc[df['logp'].idxmax()]
    min_logp = df.iloc[df['logp'].idxmin()]
    avg_logp = df['logp'].mean()

    max_mw = df.iloc[df['mw'].idxmax()]
    min_mw = df.iloc[df['mw'].idxmin()]
    avg_mw = df['mw'].mean()

    max_sa = df.iloc[df['sa'].idxmax()]
    min_sa = df.iloc[df['sa'].idxmin()]
    avg_sa = df['sa'].mean()

    max_qed = df.iloc[df['qed'].idxmax()]
    min_qed = df.iloc[df['qed'].idxmin()]
    avg_qed = df['qed'].mean()

    print(tabulate([
        ["Max logp", max_logp["logp"], max_logp["smile"]],
        ["Min logp", min_logp["logp"], min_logp["smile"]],
        ["Avg logp", avg_logp],

        ["Max MW", max_mw["mw"], max_mw["smile"]],
        ["Min MW", min_mw["mw"], min_mw["smile"]],
        ["Avg MW", avg_mw],

        ["Max SA", max_sa["sa"], max_sa["smile"]],
        ["Min SA", min_sa["sa"], min_sa["smile"]],
        ["Avg SA", avg_sa],

        ["Max QED", max_qed["qed"], max_qed["smile"]],
        ["Min QED", min_qed["qed"], min_qed["smile"]],
        ["Avg QED", avg_qed],
    ]))

    print_similarities(df, fast_iterator, seed=23)

    process_pool.close()

