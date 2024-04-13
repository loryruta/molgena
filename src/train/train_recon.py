from common import *
import logging
from typing import *
import pandas as pd
import networkx as nx
import random
import torch
from torch.utils.data import DataLoader
from mol_dataset import ZincDataset
from construct_motif_graph import *
from motif_graph import *
from mol_graph import *
from utils.tensor_utils import *
from model.encode_mol import EncodeMol
from model.select_motif_mlp import SelectMotifMlp
from model.select_mol_attachment import SelectMolAttachment
from model.classify_mol_bond import ClassifyMolBond

MOL_REPR_DIM = 256  # The embedding size for the molecule

PARAMS = {  # TODO put somewhere else? (e.g. a configuration file)
    'encode_mol': {
        'num_steps': 15,
        'node_features_dim': 5,
        'edge_features_dim': 1,
        'node_hidden_dim': MOL_REPR_DIM,
        'edge_hidden_dim': 64
    },
    'select_motif_mlp': {
        'mol_repr_dim': MOL_REPR_DIM,
        'num_motifs': 4331 + 1,  # Vocabulary size + END token
        'reconstruction_mode': True
    },
    'select_mol_attachment': {
        'num_mpn_steps': 8,
        'mol_a_repr_dim': MOL_REPR_DIM,
        'mol_b_node_features_dim': 5,
        'mol_b_edge_features_dim': 1,
        'mol_b_node_hidden_dim': 128,
        'mol_b_edge_hidden_dim': 64,
    },
    'classify_mol_bond': {
        'num_steps': 8,
        'atom_features_dim': 5,
        'bond_features_dim': 1,
        'atom_hidden_dim': 128,
        'bond_hidden_dim': 64
    }
}


class Predictions:
    """ A dataclass to gather predictions done for the reconstruction task. """

    # B   = batch size
    # Nmf = num motifs (vocabulary size)
    # Npm = num of all partial molecules atoms
    # Nnm = num of all next motifs atoms
    # Nap = num of atom pairs to classify

    batched_motif_distr: torch.FloatTensor  # (B, Nmf)
    batched_partial_mol_candidates: torch.FloatTensor  # (Npm,)
    batched_motif_candidates: torch.FloatTensor  # (Nnm,)
    batched_bond_types: torch.FloatTensor  # (Nap, 4)

    def validate(self) -> None:
        # TODO validate more in detail?
        assert self.batched_motif_distr is not None
        assert self.batched_partial_mol_candidates is not None
        assert self.batched_motif_candidates is not None
        assert self.batched_bond_types is not None


class Labels:
    """ A dataclass to gather labels used for the reconstruction task."""

    # Raw labels
    partial_mol_smiles_list: List[str]
    next_motif_ids: List[int]  # Labels for SelectMotifMlp
    bond_labels: List[List[Tuple[int, int, int]]]  # Labels for SelectMolAttachment and ClassifyMolBond

    # Tensorized labels
    batched_motif_distr: torch.FloatTensor  # Labels for SelectMotifMlp
    batched_partial_mol_candidates: torch.FloatTensor  # Labels for SelectMolAttachment(partial_mol, motif)
    batched_motif_candidates: torch.FloatTensor  # Labels for SelectMolAttachment(motif, partial_mol)
    batched_bond_labels: torch.FloatTensor  # Labels for ClassifyMolBond

    def validate(self):
        # TODO more detailed validation
        batch_size = len(self.partial_mol_smiles_list)
        assert len(self.next_motif_ids) == batch_size
        assert self.batched_motif_distr is not None
        assert self.batched_partial_mol_candidates is not None
        assert self.batched_motif_candidates is not None
        assert self.batched_bond_labels is not None


class MolgenaReconstructTask:
    _encode_mol: EncodeMol
    _select_motif_mlp: SelectMotifMlp
    _select_mol_attachment: SelectMolAttachment

    _mol_graphs: TensorGraph
    _partial_mol_graphs: TensorGraph
    _motif_mol_graphs: TensorGraph

    _mol_reprs: torch.FloatTensor
    _partial_mol_reprs: torch.FloatTensor
    _motif_mol_reprs: torch.FloatTensor

    # The input for ClassifyMolBond;
    # a (2, NC) tensor indicating the atom pairs (partial_mol_atom, motif_atom) proposed for bond classification
    # It's generated during the labeling process and cached for later use
    _bond_classification_input: torch.LongTensor

    def __init__(self, params: Dict[str, Any]):  # TODO typing for params
        self._params = params

        # Loading datasets
        self._training_set = ZincDataset.training_set()
        logging.info(f"Training set loaded; Num molecules: {len(self._training_set)}")

        self._motif_vocab = MotifVocab.load()
        logging.info(f"Motif vocabulary loaded; Num motifs: {len(self._motif_vocab)}")

        self._motif_graphs = load_motif_graphs()
        logging.info(f"Motif graphs loaded; Num motif graphs: {len(self._motif_graphs)}")

        assert len(self._training_set) == len(self._motif_graphs)

        self._batch_size = 256
        self._training_dataloader = DataLoader(self._training_set, batch_size=self._batch_size,
                                               collate_fn=lambda batch: self._collate_fn(batch))

        self._num_motifs = len(self._motif_vocab)
        self._end_motif_idx = self._num_motifs

        # Loading model layers
        self._encode_mol = EncodeMol(**params['encode_mol'])
        logging.info(f"EncodeMol loaded; Num parameters: {num_model_params(self._encode_mol)}")

        self._select_motif_mlp = SelectMotifMlp(**params['select_motif_mlp'])
        logging.info(f"SelectMotifMlp loaded; Num parameters: {num_model_params(self._select_motif_mlp)}")

        self._select_mol_attachment = SelectMolAttachment(**params['select_mol_attachment'])
        logging.info(f"SelectMolAttachment loaded; Num parameters: {num_model_params(self._select_mol_attachment)}")

        self._classify_mol_bond = ClassifyMolBond(**params['classify_mol_bond'])
        logging.info(f"ClassifyMolBond loaded; Num parameters: {num_model_params(self._classify_mol_bond)}")

        self._parameters = list(self._encode_mol.parameters()) + \
                           list(self._select_motif_mlp.parameters()) + \
                           list(self._select_mol_attachment.parameters()) + \
                           list(self._classify_mol_bond.parameters())
        #logging.info(f"Total num parameters: {len(self._parameters)}")

        self._optimizer = torch.optim.Adam(self._parameters, lr=1e-3)
        logging.info(f"Optimizer ready")

    def _collate_fn(self, raw_batch: List[Tuple[int, str]]) -> Tuple[List[str], List[nx.Graph], TensorGraph]:
        mol_smiles_list = [mol_smiles for _, mol_smiles in raw_batch]
        motif_graphs = [self._motif_graphs[i] for i, _ in raw_batch]
        mol_graph = create_tensor_graph_from_smiles_list(mol_smiles_list)
        return mol_smiles_list, motif_graphs, mol_graph

    def _sample_labels(self, motif_graphs: List[nx.DiGraph]) -> Labels:
        """ Given (complete) motif graphs, for each of them samples a connected subgraph: the partial molecule.
        Then annotate each partial molecule with:
          - The next motif
          - The attachment it should make with the partial molecule; i.e. list of (partial_mol_atom, motif_atom, bond_type)

        :return:
            A Labels object containing raw labels (see Labels for details).
            Returned labels aren't ready to be used, they have to be tensorized with _tensorize_labels.
        """
        assert len(motif_graphs) == self._batch_size

        # Labels
        labels = Labels()
        labels.partial_mol_smiles_list = []
        labels.next_motif_ids = []
        labels.bond_labels = []

        # Iterate over every batch item
        for i, motif_graph in enumerate(motif_graphs):
            motif_subgraph_indices = sample_motif_subgraph(motif_graph)

            next_cluster_id: Optional[int] = None

            if not motif_subgraph_indices:
                # Partial molecule is empty, next cluster is uniformly sampled
                next_cluster_id = random.randint(0, len(motif_graph.nodes) - 1)
            else:
                neighbors = set({})
                for cluster in motif_subgraph_indices:
                    for _, neighbor in motif_graph.out_edges(cluster):
                        if neighbor not in motif_subgraph_indices:
                            neighbors.add(neighbor)
                if not neighbors:
                    # Partial molecule is the full molecule, no cluster should be selected (END token)
                    next_cluster_id = None
                else:
                    # We have a list of neighbors that could be selected as next, take one randomly
                    next_cluster_id = random.choice(list(neighbors))

            # Full partial molecule (generation ended)
            if next_cluster_id is None:
                # TODO we already have the SMILES; avoid conversion (performance)! (mol_smiles = batch[0])
                partial_mol_smiles = convert_motif_graph_to_smiles(motif_graph, self._motif_vocab)
                labels.partial_mol_smiles_list.append(partial_mol_smiles)
                labels.next_motif_ids.append(self._end_motif_idx)
                labels.bond_labels.append([])
                continue

            next_motif_id = motif_graph.nodes[next_cluster_id]['motif_id']

            # Empty partial molecule
            if not motif_subgraph_indices:
                labels.partial_mol_smiles_list.append("")  # Empty SMILES
                labels.next_motif_ids.append(next_motif_id)
                labels.bond_labels.append([])  # We have nothing to attach to
                continue

            # Partial molecule is not complete, take the next Motif
            labels.next_motif_ids.append(next_motif_id)

            # Generate the SMILES for the partial molecule, and map every atom with its motif -relative index
            motif_subgraph = motif_graph.subgraph(motif_subgraph_indices)
            motif_subgraph_smiles, cluster_atom_map = \
                convert_motif_graph_to_smiles(motif_subgraph, self._motif_vocab)

            labels.partial_mol_smiles_list.append(motif_subgraph_smiles)

            # Determine the bond labels between partial_mol and motif
            local_bond_labels: List[Tuple[int, int, int]] = []

            for out_edge in motif_graph.out_edges(next_cluster_id):
                cid2 = out_edge[1]
                if cid2 not in motif_subgraph_indices:
                    continue  # The edge isn't connecting back to the partial molecule

                attachment = motif_graph.edges[out_edge]['attachment']  # TODO not sure of networkx call syntax
                for (motif_a1, motif_a2), bond_type in attachment.items():
                    ai = cluster_atom_map[cid2, motif_a2]
                    local_bond_labels.append((ai, motif_a1, bond_type))

            labels.bond_labels.append(local_bond_labels)

            if len(local_bond_labels) > 1:
                print(f"ELEMENT {i}; BOND LABELS: {local_bond_labels}")

        return labels

    def _tensorize_labels(self, labels: Labels) -> None:
        """ Tensorizes the given labels to be used for computing the final loss.
        What is computed are mainly 4 tensors:
          - `batched_motif_distr` for SelectMotifMlp
          - `batched_partial_mol_candidates` for SelectMolAttachment(motif, partial_mol)
          - `batched_motif_candidates` for SelectMolAttachment(partial_mol, motif)
          - `batched_bond_labels` for ClassifyMolBond
        """

        #assert self._partial_mol_graphs
        #assert self._motif_mol_graphs

        # Create TensorGraph of partial molecules

        # SelectMotifMlp labels:
        # A distribution over motifs where the i-th motif is 1 if it should be selected as the next motif
        labels.batched_motif_distr = torch.zeros((self._batch_size, self._num_motifs + 1))
        labels.batched_motif_distr[:, labels.next_motif_ids] = 1

        # SelectMolAttachment(motif, partial_mol) labels:
        # A (NC,) distribution over partial_mol atoms (all batches) where the i-th atom is 1 if it should be a candidate
        _, _, partial_mol_batch_offsets = self._partial_mol_graphs.batch_locations(self._batch_size)
        batched_partial_mol_candidates = torch.zeros(size=(self._partial_mol_graphs.num_nodes(),), dtype=torch.float32)

        # SelectMolAttachment(partial_mol, motif) labels:
        # A (NC,) distribution over motif atoms (all batches) where the i-th atom is 1 if it should be a candidate
        _, _, motif_batch_offsets = self._motif_mol_graphs.batch_locations(self._batch_size)
        batched_motif_candidates = torch.zeros(size=(self._motif_mol_graphs.num_nodes(),), dtype=torch.float32)

        # ClassifyMolBond labels:
        # A (NC, 4) tensor that represents NC distributions over bond types, where NC is the number of bonds to classify
        batched_bond_labels = []

        batched_partial_mol_atom_indices = []
        batched_motif_atom_indices = []

        for batch_idx, batch_element in enumerate(labels.bond_labels):
            # partial_mol was complete, this step isn't needed!
            if not batch_element:
                continue

            if len(batch_element) > 1:
                print(f"ELEMENT {batch_idx}; BOND LABELS: {batch_element}")

            # batch_element is a list of (partial_mol_atom, motif_atom, bond_type)

            # tensor([1234]) + tensor([1, 2, 3, 4])

            # Partial molecule true candidates
            partial_mol_atom_indices = partial_mol_batch_offsets[batch_idx] + torch.tensor([
                partial_mol_ai for partial_mol_ai, _, _ in batch_element
            ])
            batched_partial_mol_candidates[partial_mol_atom_indices] = 1
            batched_partial_mol_atom_indices.append(partial_mol_atom_indices)

            # Motif true candidates
            motif_atom_indices = motif_batch_offsets[batch_idx] + torch.tensor([
                motif_ai for _, motif_ai, _ in batch_element
            ])
            batched_motif_candidates[motif_atom_indices] = 1
            batched_motif_atom_indices.append(motif_atom_indices)

            # Bond type labels
            for _, _, bond_type in batch_element:
                bond_distr = torch.zeros(4, dtype=torch.float32)  # TODO use a constant for 4 (max bond types)
                bond_distr[bond_type] = 1
                batched_bond_labels.append(bond_distr)

        labels.batched_partial_mol_candidates = batched_partial_mol_candidates
        labels.batched_motif_candidates = batched_motif_candidates
        labels.batched_bond_labels = torch.stack(batched_bond_labels)

        self._bond_classification_input = torch.stack([
            torch.cat(batched_partial_mol_atom_indices),
            torch.cat(batched_motif_atom_indices)
        ])

        labels.validate()

    def _run_inference(self, batch) -> Predictions:
        """ Runs inference on the model using the input batch.

        Choices of different modules (i.e. SelectMotifMlp, SelectMolAttachment, ...) are not sequential because that
        would lead to a loss harder to train (non-differentiable)!
        """

        mol_smiles_list, motif_graphs, mol_graphs = batch

        self._mol_graphs = create_tensor_graph_from_smiles_list(mol_smiles_list)
        assert self._partial_mol_graphs
        assert self._motif_mol_graphs
        assert self._bond_classification_input is not None

        pred = Predictions()

        # Inference EncodeMol
        node_hidden_dim, edge_hidden_dim = self._params['encode_mol']['node_hidden_dim'], \
            self._params['encode_mol']['edge_hidden_dim']

        self._mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        self._partial_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        self._motif_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)

        self._mol_reprs = self._encode_mol(self._mol_graphs, self._batch_size)
        self._partial_mol_reprs = self._encode_mol(self._partial_mol_graphs, self._batch_size)
        self._motif_mol_reprs = self._encode_mol(self._motif_mol_graphs, self._batch_size)

        # Inference SelectMotifMlp
        pred.batched_motif_distr = \
            self._select_motif_mlp(self._partial_mol_reprs, self._mol_reprs)  # mol_reprs is the mol to reconstruct!

        # The following modules won't run on:
        # - empty partial_mol (initial step)
        # - empty motif (END token drawn)

        # Inference SelectMolAttachment
        node_hidden_dim, edge_hidden_dim = self._params['select_mol_attachment']['mol_b_node_hidden_dim'], \
            self._params['select_mol_attachment']['mol_b_edge_hidden_dim']

        self._motif_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        pred.batched_motif_candidates = self._select_mol_attachment(self._partial_mol_reprs, self._motif_mol_graphs)

        self._partial_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        pred.batched_partial_mol_candidates = \
            self._select_mol_attachment(self._motif_mol_reprs, self._partial_mol_graphs)

        # Inference ClassifyMolBond
        node_hidden_dim, edge_hidden_dim = self._params['classify_mol_bond']['atom_hidden_dim'], \
            self._params['classify_mol_bond']['bond_hidden_dim']

        self._partial_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        self._motif_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)

        pred.batched_bond_types = \
            self._classify_mol_bond(self._partial_mol_graphs, self._motif_mol_graphs, self._bond_classification_input)

        #
        pred.validate()
        return pred

    def _compute_loss(self, pred: Predictions, labels: Labels) -> torch.FloatTensor:
        """ Given the predictions and labels, computes the final loss.
        The final loss is composed of 4 contributions:
          - L1  = loss for SelectMotifMlp
          - L21 = loss for SelectMolAttachment(motif, partial_mol)
          - L22 = loss for SelectMolAttachment(partial_mol, motif)
          - L3  = loss for ClassifyMolBond
        """

        l1 = cross_entropy(pred.batched_motif_distr, labels.batched_motif_distr)

        l21 = cross_entropy(pred.batched_partial_mol_candidates, labels.batched_partial_mol_candidates)
        l22 = cross_entropy(pred.batched_motif_candidates, labels.batched_motif_candidates)

        l3 = cross_entropy(pred.batched_bond_types, labels.batched_bond_labels)

        loss = (l1 + l21 + l22 + l3).mean()
        return loss

    def _train_step(self, batch) -> None:
        mol_smiles_list, motif_graphs, mol_graphs = batch

        # Sample partial molecules using the complete motif graphs, and annotate them
        labels = self._sample_labels(motif_graphs)

        motif_smiles_list = []
        for mid in labels.next_motif_ids:
            motif_smiles = ""
            if mid != self._end_motif_idx:
                motif_smiles = self._motif_vocab.at_id(mid)['smiles']
            motif_smiles_list.append(motif_smiles)
        self._partial_mol_graphs = tensorize_smiles_list(labels.partial_mol_smiles_list)
        self._motif_mol_graphs = tensorize_smiles_list(motif_smiles_list)

        self._tensorize_labels(labels)  # Tensorize for practical usage

        # The actual training step, inference the model and compute the loss using the labels
        self._optimizer.zero_grad()

        pred = self._run_inference(batch)
        loss = self._compute_loss(pred, labels)
        loss.backward()

        self._optimizer.step()

        logging.debug(f"Inference run; Loss: {loss.item()}")

    def _train_epoch(self) -> None:
        num_batches = len(self._training_set) // self._batch_size
        for i, batch in enumerate(self._training_dataloader):
            self._train_step(batch)

            logging.info(f"{i:>3}/{num_batches:>3} batch done")
        # TODO save checkpoint at the end of an epoch?

    def train(self):
        logging.info("Training started...")

        epoch = 1
        while True:
            logging.info(f"---------------------------------------------------------------- Epoch {epoch}")

            self._train_epoch()
            epoch += 1


def _main():
    trainer = MolgenaReconstructTask(PARAMS)
    trainer.train()


if __name__ == "__main__":
    _main()
