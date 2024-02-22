from motif_vocab import MotifVocab
from rdkit import Chem
import rdkit.DataStructs
import networkx as nx
from common import *
import random
from typing import *


def _get_isolated_nodes(g: nx.Graph) -> List[int]:
    return [i for i in g.nodes() if g.degree(i) == 0]


class MotifGraph:
    def __init__(self, g: Optional[nx.Graph] = None):
        self.do_log = True

        # Variables used for building
        self._vocab = None
        self._num_motifs = None
        self._similarity_threshold = None
        self._loaded_motifs = None

        self.g = g
        self._spanning_tree = None  # Lazily initialized
        self._spanning_tree_root = None

    def _load_motifs(self):
        """ Parses the Motifs SMILES from the vocabulary and calculates the fingerprint. """

        def parse_motif(smiles: str):
            mol = Chem.MolFromSmiles(smiles)
            fingerprint = Chem.RDKFingerprint(mol)
            return mol, fingerprint,

        self._loaded_motifs = [parse_motif(smiles) for smiles in self._vocab]

    def _gather_batch_similar_connections(self, worker_i: int, batch: List[int]):
        """ Gathers the connections similar to the i-th motifs. Easily parallelizable. """

        connections = []
        for i, motif_i in enumerate(batch):
            _, fingerprint_i = self._loaded_motifs[motif_i]
            for motif_j, (mol_j, fingerprint_j) in enumerate(self._loaded_motifs):
                if motif_i == motif_j:
                    continue
                similarity = rdkit.DataStructs.TanimotoSimilarity(fingerprint_i, fingerprint_j)
                # Add connection only if the current Motif is "enough" similar to the target
                if similarity >= self._similarity_threshold:
                    connections.append((motif_i, motif_j))
            if i % 100 == 0:
                print(f"Worker {worker_i}; Processed {i}/{len(batch)}...")
        return connections

    def _connect_isolated_nodes(self):
        """ Isolated nodes are randomly connected to other nodes until no more node is isolated. """

        isolated_nodes = _get_isolated_nodes(self.g)
        if self.do_log:
            print(f"Connecting {len(isolated_nodes)} isolated nodes...")
        connected_nodes = list(set(self.g.nodes()) - set(isolated_nodes))
        for isolated_node in isolated_nodes:
            self.g.add_edge(isolated_node, random.choice(connected_nodes))
        assert len(_get_isolated_nodes(self.g)) == 0

    def build(self, vocab: MotifVocab, similarity_threshold: float = 0.6) -> nx.Graph:
        self._vocab = vocab
        self._num_motifs = len(vocab)
        self._similarity_threshold = similarity_threshold

        self.g = nx.empty_graph(self._num_motifs)

        self._load_motifs()

        futures = []

        indices = list(range(self._num_motifs))
        batch_size = 2900  # Found to be good empirically
        batches = [indices[i:i + batch_size] for i in range(0, len(indices), batch_size)]

        for worker_i, batch in enumerate(batches):
            futures.append(threadpool.submit(self._gather_batch_similar_connections, worker_i, batch))
            print(f"Worker {worker_i} enqueued...")

        for worker_i, future in enumerate(futures):
            edges = future.result()
            for edge in edges:
                self.g.add_edge(*edge)
            print(f"Worker {worker_i} completed; Edges: {len(edges)}")

        self._connect_isolated_nodes()

        # Clears all the build variables
        self._vocab = None
        self._num_motifs = None
        self._similarity_threshold = None
        self._loaded_motifs = None

        return self.g

    def minimum_spanning_tree(self) -> Tuple[nx.Graph, int]:
        """ Lazily constructs a spanning tree for the graph. Once created, it's cached for future calls. """

        if self.g is None:
            raise Exception("Missing graph: it should be already loaded or constructed")

        if (self._spanning_tree is None) or (self._spanning_tree_root is None):
            self._spanning_tree = nx.minimum_spanning_tree(self.g)
            # Take the first node of the spanning tree as the root (actually any node could be the root)
            self._spanning_tree_root = 0
        return self._spanning_tree, self._spanning_tree_root


def load_motif_graph() -> MotifGraph:
    try:
        print(f"Loading MotifGraph at \"{MOTIF_GRAPH_PATH}\"...")
        g = nx.read_gml(MOTIF_GRAPH_PATH)
        print(f"Loaded")

    except nx.exception.NetworkXError:
        print(f"MotifGraph not found, constructing it...")
        g = MotifGraph().build(MotifVocab.load())
        print(f"Constructed")

        print(f"Saving MotifGraph at \"{MOTIF_GRAPH_PATH}\"...")
        nx.write_gml(g, MOTIF_GRAPH_PATH)
        print(f"Saved")

    return MotifGraph(g)


def _main():
    try:
        motif_graph = load_motif_graph()
    except KeyboardInterrupt:
        print("Keyboard interrupted :(")


if __name__ == "__main__":
    _main()
