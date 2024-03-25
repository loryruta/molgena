from common import *
import logging
from typing import *
import pandas as pd
import networkx as nx
import torch
from torch.utils.data import DataLoader
from mol_dataset import ZincDataset
from construct_motif_graph import *
from mol_graph import *
from utils.tensor_utils import *
from model.encode_mol import EncodeMol
from model.select_motif_mlp import SelectMotifMlp
import random


class SelectMotifMlpTrainer:
    def __init__(self):
        self._training_set = ZincDataset.training_set()
        self._motif_vocab = MotifVocab.load()
        self._motif_graphs = load_motif_graphs()

        self._batch_size = 256

        self._training_dataloader = DataLoader(self._training_set, batch_size=self._batch_size,
                                               collate_fn=lambda batch: self._collate_fn(batch))

        self._num_motifs = len(self._motif_vocab)
        self._end_motif_idx = self._num_motifs

        self._encode_mol = EncodeMol(**{
            'num_steps': 100,
            'node_features_dim': 5,
            'edge_features_dim': 1,
            'node_hidden_dim': 200,  # = mol_repr_dim
            'edge_hidden_dim': 200
        })
        self._select_motif_mlp = SelectMotifMlp(**{
            'mol_repr_dim': 200,
            'num_motifs': len(self._motif_vocab) + 1,
            'reconstruction_mode': True
        })

        self._parameters = list(self._encode_mol.parameters()) + list(self._select_motif_mlp.parameters())
        self._optimizer = torch.optim.Adam(self._parameters, lr=1e-3)

    def _collate_fn(self, raw_batch: List[Tuple[int, str]]) -> Tuple[List[str], List[nx.Graph], TensorGraph]:
        mol_smiles_list = [mol_smiles for _, mol_smiles in raw_batch]
        motif_graphs = [self._motif_graphs[i] for i, _ in raw_batch]
        mol_graph = create_tensor_graph_from_smiles_list(mol_smiles_list)
        return mol_smiles_list, motif_graphs, mol_graph

    def _sample_motif_subgraph(self, motif_graph: nx.DiGraph) -> Set[int]:
        assert motif_graph.nodes

        # TODO Review the sampling algorithm!
        #   Ideally we would like same probability of choosing any number of nodes (from zero to full graph)

        # Probability to add another node to the existing subgraph
        CONTINUATION_PROBABILITY = 0.6

        taken_nodes = set({})
        while True:
            if not taken_nodes:  # If no taken nodes, candidates are all nodes
                neighbors = motif_graph.nodes
            else:  # Otherwise candidates are neighbors of taken nodes
                neighbors = set({})
                for taken_node in taken_nodes:
                    neighbors.update([edge[1] for edge in motif_graph.out_edges(taken_node)])
                neighbors = neighbors - taken_nodes
            if not neighbors:  # Nothing to select!
                break
            continue_ = random.random() > CONTINUATION_PROBABILITY
            if not continue_:
                break
            neighbor = random.choice(list(neighbors))
            taken_nodes.add(neighbor)
        return taken_nodes

    def _sample_partial_molecules(self, motif_graphs: List[nx.DiGraph]) -> Tuple[List[str], torch.Tensor, TensorGraph]:
        """ Given (complete) motif graphs, for each of them samples a subgraph: the partial molecule.

        :return:
            A tuple of:
            - List of partial molecules SMILES
            - A one-hot tensor (B, M+1) indicating which motifs can be selected per molecule (ready for cross entropy!)
            - TensorGraph of all partial molecules
        """
        assert len(motif_graphs) == self._batch_size

        partial_mol_smiles_list: List[str] = []
        selectable_motif_labels: torch.Tensor = torch.zeros((self._batch_size, self._num_motifs + 1))

        for i, motif_graph in enumerate(motif_graphs):
            motif_subgraph = self._sample_motif_subgraph(motif_graph)

            selectable_motif_indices = set({})
            for cluster in motif_subgraph:
                for _, neighbor in motif_graph.out_edges(cluster):
                    if neighbor not in motif_subgraph:
                        selectable_motif_idx = motif_graph[neighbor]['motif_id']
                        selectable_motif_indices.add(selectable_motif_idx)

            # All clusters were selected in motif_subgraph; the only valid motif to select is END
            if not selectable_motif_indices:
                selectable_motif_indices = {self._end_motif_idx}

            partial_mol_smiles_list.append(motif_graph_to_smiles(motif_graph, motif_subgraph, self._motif_vocab))
            selectable_motif_labels[i, list(selectable_motif_indices)] = 1

        return (
            partial_mol_smiles_list,
            selectable_motif_labels,
            create_tensor_graph_from_smiles_list(partial_mol_smiles_list)
        )

    def _compute_loss(self, batch):
        mol_smiles_list, motif_graphs, mol_graphs = batch

        # Sample partial molecules using the (complete) motif_graphs
        (partial_mol_smiles_list, selectable_motif_labels, partial_mol_graphs) = \
            self._sample_partial_molecules(motif_graphs)

        # Encode both complete mol_graphs and partial_mol_graphs; use the encodings to predict the next motif
        mol_reprs = self._encode_mol(mol_graphs)
        partial_mol_reprs = self._encode_mol(partial_mol_graphs)

        pred = self._select_motif_mlp(mol_reprs, partial_mol_reprs)

        # Use cross entropy to predict the loss
        return cross_entropy(selectable_motif_labels, pred).mean()

    def _train_epoch(self):
        for i, batch in enumerate(self._training_dataloader):
            self._optimizer.zero_grad()

            loss = self._compute_loss(batch)
            loss.backward()
            self._optimizer.step()
            print("batch done")
        # TODO save checkpoint at the end of an epoch ???

    def train(self):
        logging.info("Training started...")

        epoch = 1
        while True:
            logging.info(f"---------------------------------------------------------------- Epoch {epoch}")

            self._train_epoch()
            epoch += 1


def _main():
    trainer = SelectMotifMlpTrainer()
    trainer.train()


if __name__ == "__main__":
    _main()
