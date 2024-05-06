from common import *
import logging
from random import Random
from typing import *
import torch
from rdkit import Chem
from motif_graph.construct_motif_graph import construct_motif_graph
from motif_graph.convert_motif_graph import convert_motif_graph_to_smiles
from motif_graph.sample_motif_subgraph import sample_motif_subgraph
from motif_vocab import MotifVocab
import networkx as nx


class Annotations:
    """ Annotations for a single molecule SMILES. Used only as an intermediate representation. """

    mol_smiles: str
    """ The target molecule SMILES. """

    mol_mgraph: nx.DiGraph
    """ The target molecule's mgraph. """

    partial_mol_smiles: str
    """ The partial molecule SMILES. """

    mgraph_subgraph_indices: List[int]
    """ The node indices of the initial molecule's mgraph, that makes the partial molecule. """

    partial_mol_mgraph: nx.DiGraph
    """ The partial molecule's mgraph. """

    is_empty: bool = False
    """ Is the partial molecule empty (empty SMILES)? It's the initial step of reconstruction. """

    is_full: bool = False
    """ Is the partial molecule full (equal to the input)? It's the final step of reconstruction. """

    next_cluster_id: int
    """ The cluster ID for the Motif (IDs are w.r.t. the complete mgraph) """

    next_motif_id: int
    """ The next Motif ID to select. """

    attachment_cluster_id: int
    """ The cluster ID for the partial molecule to form the attachment with. 
    By analyzing the dataset, it was found that most of the times the Motif is attached to the partial molecule with
    *only one* cluster!
    """

    cluster1_motif_ai: int
    """ The atom selected for bond of cluster1 (partial molecule). A motif -relative index. """

    cluster2_motif_ai: int
    """ The atom selected for bond of cluster2 (motif). A motif -relative index. """

    bond_type: Chem.BondType
    """ The bond type to form between cluster1 and cluster2 (partial molecule and motif). """

    def _validate_empty_pmol(self):
        assert self.mol_smiles
        assert self.mol_mgraph
        assert self.partial_mol_smiles == ""
        assert len(self.mgraph_subgraph_indices) == 0
        assert hasattr(self, "partial_mol_mgraph")
        assert nx.is_empty(self.partial_mol_mgraph)
        assert self.is_empty
        assert not self.is_full
        assert self.next_cluster_id in self.mol_mgraph.nodes
        assert hasattr(self, "next_motif_id")  # TODO != END token
        assert not hasattr(self, "attachment_cluster_id")
        assert not hasattr(self, "cluster1_motif_ai")
        assert not hasattr(self, "cluster2_motif_ai")
        assert not hasattr(self, "bond_type")

    def _validate_full_pmol(self):
        assert self.mol_smiles
        assert self.mol_mgraph
        # TODO assert self.partial_mol_smiles == self.mol_smiles
        assert len(self.mgraph_subgraph_indices) == len(self.mol_mgraph.nodes)
        assert hasattr(self, "partial_mol_mgraph")
        assert len(self.partial_mol_mgraph.nodes) == len(self.mol_mgraph.nodes)
        assert not self.is_empty
        assert self.is_full
        assert not hasattr(self, "next_cluster_id")
        assert self.next_motif_id > 0  # TODO == END token
        assert not hasattr(self, "attachment_cluster_id")
        assert not hasattr(self, "cluster1_motif_ai")
        assert not hasattr(self, "cluster2_motif_ai")
        assert not hasattr(self, "bond_type")

    def _validate_pmol(self):
        assert self.mol_smiles
        assert self.mol_mgraph
        assert self.partial_mol_smiles
        assert len(self.mgraph_subgraph_indices) > 0
        assert len(self.mgraph_subgraph_indices) < len(self.mol_mgraph.nodes)
        assert hasattr(self, "partial_mol_mgraph")
        assert len(self.partial_mol_mgraph.nodes) == len(self.mgraph_subgraph_indices)
        assert not self.is_empty
        assert not self.is_full
        assert self.next_cluster_id in self.mol_mgraph.nodes
        assert self.next_cluster_id not in self.mgraph_subgraph_indices
        assert self.mol_mgraph.nodes[self.next_cluster_id]['motif_id'] == self.next_motif_id
        assert hasattr(self, "attachment_cluster_id")
        assert hasattr(self, "cluster1_motif_ai")
        assert hasattr(self, "cluster2_motif_ai")
        assert hasattr(Chem.BondType, self.bond_type.name)

    def validate(self):
        if self.is_empty:
            self._validate_empty_pmol()
        elif self.is_full:
            self._validate_full_pmol()
        else:
            self._validate_pmol()


class BatchedAnnotations:
    # SelectMotifMlp
    next_motif_ids: torch.LongTensor  # (B,)
    """ A tensor of shape (B,) where the i-th element is the next Motif ID for the i-th molecule. """

    # SelectAttachmentClusters
    attachment_cluster_ids: torch.LongTensor
    """ A tensor of shape (B,) where the i-th element is the attachment cluster ID of the i-th partial molecule. """

    # SelectAttachmentAtom(cluster1)
    cluster1_attachment_atoms: torch.FloatTensor
    """ A tensor over batched cluster1 atoms, where the i-th element is {0, 1} if selected for attachment.
     Multiple elements could be 1 as a result of automorphisms. """

    # SelectAttachmentAtom(cluster2)
    cluster2_attachment_atoms: torch.FloatTensor
    """ A tensor over batched cluster2 atoms, where the i-th element is {0, 1} if selected for attachment.
     Multiple elements could be 1 as a result of automorphisms. """

    # SelectAttachmentBondType
    attachment_bond_types: torch.LongTensor  # (B,)
    """ A tensor of shape (B,) where the i-th element is the ID of the `Chem.BondType`. """


class Annotator:
    """ Class in charge of annotating a given molecule for the reconstruction task.

     We start with a partial molecule (special case: empty or full), and annotate the steps required to reach the input
     molecule. """

    def __init__(self, seed: Optional[int] = None):
        self._motif_vocab = MotifVocab.load()
        self._random = Random() if seed is None else Random(seed)

    def _annotate_partial_mol(self, annotations: Annotations):
        mol_smiles = annotations.mol_smiles

        mgraph = construct_motif_graph(mol_smiles, self._motif_vocab)
        mgraph_subgraph_indices = sample_motif_subgraph(mgraph, seed=self._random.randint(0, (1 << 31) - 1))

        partial_mol_smiles = ""
        partial_mol_mgraph = mgraph.subgraph(mgraph_subgraph_indices)  # Could be also an empty graph
        if len(mgraph_subgraph_indices) > 0:
            partial_mol_smiles = \
                convert_motif_graph_to_smiles(partial_mol_mgraph, self._motif_vocab)[0]

        # Store annotations
        annotations.mol_mgraph = mgraph
        annotations.partial_mol_smiles = partial_mol_smiles
        annotations.mgraph_subgraph_indices = mgraph_subgraph_indices
        annotations.partial_mol_mgraph = partial_mol_mgraph
        annotations.is_empty = len(mgraph_subgraph_indices) == 0
        annotations.is_full = len(mgraph_subgraph_indices) == len(mgraph.nodes)

    def _annotate_next_motif_id(self, annotations: Annotations):
        # Partial molecule is the full molecule, no cluster should be selected (END token)
        if annotations.is_full:
            # labels.next_cluster_id = None
            annotations.next_motif_id = self._motif_vocab.end_motif_id()
            return

        mgraph = annotations.mol_mgraph
        mgraph_subgraph_indices = annotations.mgraph_subgraph_indices

        next_cluster_id: Optional[int]

        # Partial molecule is empty, next cluster is uniformly sampled
        if annotations.is_empty:
            annotations.next_cluster_id = self._random.randint(0, len(mgraph.nodes) - 1)
            annotations.next_motif_id = mgraph.nodes[annotations.next_cluster_id]['motif_id']
            return

        # Check neighboring clusters that aren't part of the partial molecule
        neighbors = set({})
        for cluster in mgraph_subgraph_indices:
            for _, neighbor in mgraph.out_edges(cluster):
                if neighbor not in mgraph_subgraph_indices:
                    neighbors.add(neighbor)
        assert len(neighbors) > 0  # At least one neighbor should be found as the partial mol isn't full

        # We have a list of neighbors that could be selected as next, take one randomly
        annotations.next_cluster_id = self._random.choice(list(neighbors))
        annotations.next_motif_id = mgraph.nodes[annotations.next_cluster_id]['motif_id']

    def _annotate_attachment_cluster_id(self, annotations: Annotations):
        # If the partial molecule is empty or full, there are no attachment clusters to select
        if annotations.is_empty or annotations.is_full:
            return

        next_cluster_id = annotations.next_cluster_id
        mol_mgraph = annotations.mol_mgraph
        mgraph_subgraph_indices = annotations.mgraph_subgraph_indices

        cluster_ids = set()
        for out_edge in mol_mgraph.out_edges(next_cluster_id):
            cid2 = out_edge[1]
            if cid2 not in mgraph_subgraph_indices:
                continue  # The edge isn't connecting back to the partial molecule, skip it
            cluster_ids.add(cid2)
        assert len(cluster_ids) > 0
        if len(cluster_ids) > 1:
            logging.warning(f"Found rare case where len(cluster_ids) > 1 ({len(cluster_ids)})")

        # Store annotations
        annotations.attachment_cluster_id = list(cluster_ids)[0]

    def _annotate_cluster1_cluster2_bond(self, annotations: Annotations):
        if not hasattr(annotations, 'attachment_cluster_id'):
            return  # No attachment clusters were selected; meaning partial molecule is either empty or full

        mol_mgraph = annotations.mol_mgraph
        cid1 = annotations.attachment_cluster_id
        cid2 = annotations.next_cluster_id

        motif1_ai, motif2_ai, bond_type = mol_mgraph.edges[cid1, cid2]['attachment']

        annotations.cluster1_motif_ai = motif1_ai
        annotations.cluster2_motif_ai = motif2_ai
        annotations.bond_type = bond_type

    def annotate(self, mol_smiles: str) -> Annotations:
        annotations = Annotations()
        annotations.mol_smiles = mol_smiles
        self._annotate_partial_mol(annotations)
        self._annotate_next_motif_id(annotations)
        self._annotate_attachment_cluster_id(annotations)
        self._annotate_cluster1_cluster2_bond(annotations)
        annotations.validate()
        return annotations

    def annotate_batch(self, mol_smiles_list: List[str]) -> BatchedAnnotations:
        batched_annotations = BatchedAnnotations()

        annotations_list = [self.annotate(mol_smiles) for mol_smiles in mol_smiles_list]

        # next_motif_ids
        next_motif_ids = [annotations.next_motif_id for annotations in annotations_list]
        batched_annotations.next_motif_ids = torch.tensor(next_motif_ids, dtype=torch.long)

        # attachment_cluster_ids


        # next_attachment_cluster_ids =

        # cluster1_attachment_atoms
        # cluster2_attachment_atoms
        # attachment_bond_types

        return batched_annotations
