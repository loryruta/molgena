from __future__ import annotations
from common import *
import pickle
from typing import *
import networkx as nx
from motif_vocab import MotifVocab
from motif_graph.construct_motif_graph import construct_motif_graph


class MgraphCache:
    _singleton: 'MgraphCache'

    def __init__(self):
        self._motif_vocab = MotifVocab.load()
        self._cached_mgraphs: Dict[str, nx.DiGraph] = {}

    def __del__(self):
        logging.warning(f"[MgraphCache] Cache is being destroyed")
        # TODO it would be a good idea to save cached mgraphs to a file

    def _load_cached_mgraphs(self):
        if not path.exists(MGRAPHS_PKL):
            logging.warning(f"[MgraphCache] Cached mgraphs not found: {MGRAPHS_PKL}")

        with open(MGRAPHS_PKL, "rb") as file:
            logging.info(f"[MgraphCache] Loading cached mgraphs at {MGRAPHS_PKL} (might take a while)")
            self._cached_mgraphs = pickle.load(file)
            logging.info(f"[MgraphCache] Loaded {len(self._cached_mgraphs)} mgraphs")

    def _get_or_construct_mgraph(self, mol_smiles: str) -> nx.DiGraph:
        if mol_smiles in self._cached_mgraphs:
            return self._cached_mgraphs[mol_smiles]
        mgraph = construct_motif_graph(mol_smiles, self._motif_vocab)
        self._cached_mgraphs[mol_smiles] = mgraph
        return mgraph

    # TODO method for getting the cache size?

    @staticmethod
    def instance() -> 'MgraphCache':
        if not hasattr(MgraphCache, "_singleton"):
            MgraphCache._singleton = MgraphCache()
            MgraphCache._singleton._load_cached_mgraphs()
        return MgraphCache._singleton

    @staticmethod
    def get_or_construct_mgraph(mol_smiles: str) -> nx.DiGraph:
        return MgraphCache.instance()._get_or_construct_mgraph(mol_smiles)
