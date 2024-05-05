from common import *
import networkx as nx
import pickle
from typing import *

from motif_graph.construct_motif_graph import construct_motif_graph
from motif_graph.convert_motif_graph import convert_motif_graph_to_smiles
from motif_graph.sample_motif_subgraph import sample_motif_subgraph
from motif_graph.tensorize_motif_graph import tensorize_mgraph


def load_motif_graphs_pkl(pkl_filepath: str) -> List[nx.Graph]:
    """ Loads a .pkl file containing motif graphs at the specified path. """
    try:
        with open(pkl_filepath, "rb") as file:
            return pickle.load(file)
    except Exception as _:
        raise Exception(f"Motif graph not found: {pkl_filepath}")
