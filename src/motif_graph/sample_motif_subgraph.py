from common import *
from random import Random
import networkx as nx
from typing import *


def _sample_subgraph_uniformly(graph: nx.DiGraph, seed: Optional[int]) -> Set[int]:
    rand = Random(seed)

    num_subgraph_nodes = rand.randint(0, len(graph.nodes) - 1)

    if num_subgraph_nodes == 0:
        rand.seed()
        return set({})

    initial_node = rand.randint(0, len(graph.nodes) - 1)
    if num_subgraph_nodes == 1:
        return {initial_node}

    taken_nodes = {initial_node}
    while len(taken_nodes) < num_subgraph_nodes:
        neighbors = set({})
        for u in taken_nodes:
            try:
                neighbors.update([v for _, v in graph.out_edges(u) if v not in taken_nodes])
            except:
                pass
        taken_nodes.add(rand.choice(list(neighbors)))
    return taken_nodes


def sample_motif_subgraph(motif_graph: nx.DiGraph, seed: Optional[int] = None) -> Set[int]:
    return _sample_subgraph_uniformly(motif_graph, seed)


def _visualize_sampling_histogram():
    import numpy as np
    from mol_dataset import ZincDataset
    from motif_vocab import MotifVocab
    from construct_motif_graph import construct_motif_graph
    import matplotlib.pyplot as plt

    NUM_SAMPLES = 100000
    SEED = 2
    NUM_HIST_BINS = 10

    training_set = ZincDataset.training_set()
    motif_vocab = MotifVocab.load()

    mol_smiles_list = training_set.df.sample(n=NUM_SAMPLES)['smiles'].tolist()

    data = []
    for i, mol_smiles in enumerate(mol_smiles_list):
        motif_graph = construct_motif_graph(mol_smiles, motif_vocab)
        motif_subgraph_indices = sample_motif_subgraph(motif_graph, SEED)

        sampled_frac = len(motif_subgraph_indices) / len(motif_graph.nodes)
        data.append(sampled_frac)
        # y[int(np.floor((sampled_frac * 0.999) * NUM_PLOT_SAMPLES))] += 1

        if (i+1) % 10000 == 0:
            logging.info(f"Sampled {i+1} motif subgraphs...")

    plt.hist(data, bins=NUM_HIST_BINS)
    plt.show()


if __name__ == "__main__":
    _visualize_sampling_histogram()
