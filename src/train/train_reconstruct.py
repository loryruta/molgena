from common import *
import logging
from typing import *
import pandas as pd
import networkx as nx
from pathlib import Path
from random import Random
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from mol_dataset import ZincDataset
from motif_graph import *
from mol_graph import *
from motif_vocab import MotifVocab
from utils.misc_utils import *
from utils.tensor_utils import *
from model.encode_mol import EncodeMol
from model.select_motif_mlp import SelectMotifMlp
from model.select_mol_attachment import SelectMolAttachment
from model.classify_mol_bond import ClassifyMolBond


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
    next_motif_ids: List[int]
    bond_labels: List[List[Tuple[int, int, int]]]  # Labels for SelectMolAttachment and ClassifyMolBond

    # Tensorized labels
    next_motif_labels: torch.LongTensor  # Labels for SelectMotifMlp
    batched_partial_mol_candidates: torch.FloatTensor  # Labels for SelectMolAttachment(partial_mol, motif)
    batched_motif_candidates: torch.FloatTensor  # Labels for SelectMolAttachment(motif, partial_mol)
    batched_bond_labels: torch.FloatTensor  # Labels for ClassifyMolBond

    def validate(self):
        # TODO more detailed validation
        batch_size = len(self.partial_mol_smiles_list)
        assert self.next_motif_labels.shape == (batch_size,)
        assert self.batched_partial_mol_candidates is not None
        assert self.batched_motif_candidates is not None
        assert self.batched_bond_labels is not None


class Loss:
    l1_: torch.FloatTensor  # SelectMotifMlp
    l21: torch.FloatTensor  # SelectMolAttachment(partial_mol, motif_graph)
    l22: torch.FloatTensor  # SelectMolAttachment(motif_graph, partial_mol)
    l3_: torch.FloatTensor  # ClassifyMolBond

    total_loss: torch.FloatTensor

    def validate(self):
        assert self.l1_.ndim == 0
        assert self.l21.ndim == 0
        assert self.l22.ndim == 0
        assert self.l3_.ndim == 0


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

    def __init__(self, train_dir: str, config_filepath: str):
        self._config_name = Path(config_filepath).stem
        self._config = load_json_with_vars(config_filepath)

        # Create ./runs/ directory
        self._runs_dir = path.join(train_dir, f"runs-{self._config_name}")
        if not path.exists(self._runs_dir):
            os.mkdir(self._runs_dir)

        # Create ./checkpoints/ directory
        self._checkpoints_dir = path.join(train_dir, f"checkpoints-{self._config_name}")
        if not path.exists(self._checkpoints_dir):
            os.mkdir(self._checkpoints_dir)

        # Load datasets
        self._training_set = ZincDataset.training_set()
        logging.info(f"Training set loaded; Num molecules: {len(self._training_set)}")

        self._test_set = ZincDataset.test_set()
        logging.info(f"Test set loaded; Num molecules: {len(self._test_set)}")

        self._motif_vocab = MotifVocab.load()
        logging.info(f"Motif vocabulary loaded; Num motifs: {len(self._motif_vocab)}")

        self._motif_graphs = load_motif_graphs_pkl(TRAINING_MOTIF_GRAPHS_PKL)
        logging.info(f"Motif graphs loaded; Num motif graphs: {len(self._motif_graphs)}")

        assert len(self._training_set) == len(self._motif_graphs)

        self._batch_size = 256
        self._test_batch_size = 1024

        self._training_dataloader = DataLoader(self._training_set, batch_size=self._batch_size,
                                               collate_fn=lambda batch: self._collate_fn(batch))

        self._num_motifs = len(self._motif_vocab)
        self._end_motif_idx = self._num_motifs

        # Load modules
        self._encode_mol = EncodeMol(self._config['encode_mol'])
        logging.info(f"EncodeMol loaded; Num parameters: {num_model_params(self._encode_mol)}")

        self._select_motif_mlp = SelectMotifMlp(self._config['select_motif_mlp'])
        logging.info(f"SelectMotifMlp loaded; Num parameters: {num_model_params(self._select_motif_mlp)}")

        self._select_mol_attachment = SelectMolAttachment(self._config['select_mol_attachment'])
        logging.info(f"SelectMolAttachment loaded; Num parameters: {num_model_params(self._select_mol_attachment)}")

        self._classify_mol_bond = ClassifyMolBond(self._config['classify_mol_bond'])
        logging.info(f"ClassifyMolBond loaded; Num parameters: {num_model_params(self._classify_mol_bond)}")

        # Create optimizer and LR scheduler
        parameters = [
            *self._encode_mol.parameters(),
            *self._select_motif_mlp.parameters(),
            *self._select_mol_attachment.parameters(),
            # *self._classify_mol_bond.parameters()
        ]
        logging.info(f"Total num parameters: {sum([param.numel() for param in parameters])}")

        self._optimizer = torch.optim.Adam(parameters, lr=0.001)
        self._lr_scheduler = CosineAnnealingLR(self._optimizer, T_max=50)
        logging.info(f"Optimizer ready")

        self._epoch = 0

        # Create tensorboard writer
        self._writer = SummaryWriter(log_dir=self._runs_dir)
        self._writer_step = 0

        # Load latest checkpoint if any
        latest_checkpoint_path = path.join(self._checkpoints_dir, "checkpoint-latest.pt")
        if path.exists(latest_checkpoint_path):
            self._load_checkpoint(latest_checkpoint_path)

    def _collate_fn(self, raw_batch: List[Tuple[int, str]]) -> Tuple[List[str], List[nx.Graph], TensorGraph]:
        mol_smiles_list = [mol_smiles for _, mol_smiles in raw_batch]
        motif_graphs = [self._motif_graphs[i] for i, _ in raw_batch]
        mol_graph = tensorize_smiles_list(mol_smiles_list)
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
        # Labels
        labels = Labels()
        labels.partial_mol_smiles_list = []
        labels.next_motif_ids = []
        labels.bond_labels = []

        # SEED = 23
        rand = Random()  # Random(SEED + 321)

        # Iterate over every batch item
        for i, motif_graph in enumerate(motif_graphs):
            motif_subgraph_indices = sample_motif_subgraph(motif_graph)

            next_cluster_id: Optional[int]

            if not motif_subgraph_indices:
                # Partial molecule is empty, next cluster is uniformly sampled
                next_cluster_id = rand.randint(0, len(motif_graph.nodes) - 1)
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
                    next_cluster_id = rand.choice(list(neighbors))

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

        return labels

    def _tensorize_labels(self, labels: Labels):
        """ Tensorizes the given labels to be used for computing the final loss.
        What is computed are mainly 4 tensors:
          - `next_motif_labels` for SelectMotifMlp
          - `batched_partial_mol_candidates` for SelectMolAttachment(motif, partial_mol)
          - `batched_motif_candidates` for SelectMolAttachment(partial_mol, motif)
          - `batched_bond_labels` for ClassifyMolBond
        """

        # assert self._partial_mol_graphs
        # assert self._motif_mol_graphs

        # Create TensorGraph of partial molecules

        batch_size = len(labels.partial_mol_smiles_list)

        # SelectMotifMlp labels
        labels.next_motif_labels = torch.tensor(labels.next_motif_ids, dtype=torch.long)

        # SelectMolAttachment(motif, partial_mol) labels:
        # A (NC,) distribution over partial_mol atoms (all batches) where the i-th atom is 1 if it should be a candidate
        _, _, partial_mol_batch_offsets = self._partial_mol_graphs.batch_locations(batch_size)
        batched_partial_mol_candidates = torch.zeros(size=(self._partial_mol_graphs.num_nodes(),), dtype=torch.float32)

        # SelectMolAttachment(partial_mol, motif) labels:
        # A (NC,) distribution over motif atoms (all batches) where the i-th atom is 1 if it should be a candidate
        _, _, motif_batch_offsets = self._motif_mol_graphs.batch_locations(batch_size)
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

        batch_size = len(mol_smiles_list)

        self._mol_graphs = tensorize_smiles_list(mol_smiles_list)
        assert self._partial_mol_graphs
        assert self._motif_mol_graphs
        assert self._bond_classification_input is not None

        pred = Predictions()

        # Inference EncodeMol
        self._mol_reprs = self._encode_mol(self._mol_graphs, batch_size)
        self._partial_mol_reprs = self._encode_mol(self._partial_mol_graphs, batch_size)
        self._motif_mol_reprs = self._encode_mol(self._motif_mol_graphs, batch_size)

        # Inference SelectMotifMlp
        pred.batched_motif_distr = \
            self._select_motif_mlp(self._partial_mol_reprs, self._mol_reprs)  # mol_reprs is the mol to reconstruct!

        # The following modules won't run on:
        # - empty partial_mol (initial step)
        # - empty motif (END token drawn)

        # Inference SelectMolAttachment
        pred.batched_motif_candidates = \
            self._select_mol_attachment(self._partial_mol_reprs, self._motif_mol_graphs)

        pred.batched_partial_mol_candidates = \
            self._select_mol_attachment(self._motif_mol_reprs, self._partial_mol_graphs)

        # Inference ClassifyMolBond
        # TODO one thing at a time :)
        # node_hidden_dim, edge_hidden_dim = self._config['classify_mol_bond']['atom_hidden_dim'], \
        #     self._config['classify_mol_bond']['bond_hidden_dim']
        #
        # self._partial_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        # self._motif_mol_graphs.create_hiddens(node_hidden_dim, edge_hidden_dim)
        #
        # pred.batched_bond_types = \
        #     self._classify_mol_bond(self._partial_mol_graphs, self._motif_mol_graphs, self._bond_classification_input)

        # pred.validate()
        return pred

    def _compute_loss(self, pred: Predictions, labels: Labels) -> Loss:
        """ Given the predictions and labels, computes the final loss.
        The final loss is composed of 4 contributions:
          - L1_ = loss for SelectMotifMlp
          - L21 = loss for SelectMolAttachment(motif, partial_mol)
          - L22 = loss for SelectMolAttachment(partial_mol, motif)
          - L3_ = loss for ClassifyMolBond
        """

        a1_ = 1.
        a21 = 10.
        a22 = 10.
        a3_ = 1.

        loss = Loss()
        loss.l1_ = a1_ * F.cross_entropy(pred.batched_motif_distr, labels.next_motif_labels)
        loss.l21 = a21 * F.binary_cross_entropy(pred.batched_partial_mol_candidates,
                                                labels.batched_partial_mol_candidates).mean()
        loss.l22 = a22 * F.binary_cross_entropy(pred.batched_motif_candidates, labels.batched_motif_candidates).mean()
        # loss.l3_ = \
        #     cross_entropy(pred.batched_bond_types, labels.batched_bond_labels, dim=1).mean()
        loss.l3_ = torch.tensor(0)

        loss.total_loss = loss.l1_ + loss.l21 + loss.l22 + loss.l3_

        return loss

    def _train_step(self, batch_idx: int, batch) -> None:
        mol_smiles_list, motif_graphs, mol_graphs = batch

        # Sample partial molecules using the complete motif graphs, and annotate them
        labels = self._sample_labels(motif_graphs)

        self._partial_mol_graphs = tensorize_smiles_list(labels.partial_mol_smiles_list)
        motif_smiles_list = []
        for mid in labels.next_motif_ids:
            motif_smiles = ""
            if mid != self._end_motif_idx:
                motif_smiles = self._motif_vocab.at_id(mid)['smiles']
            motif_smiles_list.append(motif_smiles)
        self._motif_mol_graphs = tensorize_smiles_list(motif_smiles_list)

        self._tensorize_labels(labels)  # Tensorize for practical usage

        torch.autograd.set_detect_anomaly(True)

        self._encode_mol.train()
        self._select_motif_mlp.train()
        self._select_mol_attachment.train()
        self._classify_mol_bond.train()

        # The actual training step, inference the model and compute the loss using the labels
        self._optimizer.zero_grad()

        pred = self._run_inference(batch)

        loss = self._compute_loss(pred, labels)
        loss.total_loss.backward()

        self._optimizer.step()

        # lr = self._optimizer.param_groups[0]['lr']
        lr = self._lr_scheduler.get_last_lr()[0]
        self._lr_scheduler.step()

        # Update tensorboard
        self._writer.add_scalars("loss", {
            "l1": loss.l1_,
            "l21": loss.l21,
            "l22": loss.l22,
            "l3": loss.l3_,
            # "total": loss.total_loss,
        }, self._writer_step)

        # Log
        num_batches = len(self._training_set) // self._batch_size
        logging.debug(f"Batch {batch_idx:>3}/{num_batches:>3} Inference run; "
                      f"L1_: {loss.l1_.item()}, "
                      f"L21: {loss.l21.item():.8f}, "
                      f"L22: {loss.l22.item():.8f}, "
                      f"L3_: {loss.l3_.item():.8f}, "
                      f"Total loss: {loss.total_loss.item():.5f}, "
                      f"LR: {lr:.7f}")

    def _test(self):
        batch_size = self._test_batch_size

        test_smiles_list = self._test_set.df.sample(n=batch_size)['smiles'].tolist()
        motif_graphs = [construct_motif_graph(smiles, self._motif_vocab) for smiles in test_smiles_list]
        mol_graphs = tensorize_smiles_list(test_smiles_list)

        batch = (test_smiles_list, motif_graphs, mol_graphs)

        labels = self._sample_labels(motif_graphs)

        # TODO don't use member vars (self._partial_mol_graphs, self._motif_mol_graphs, ...)
        self._partial_mol_graphs = tensorize_smiles_list(labels.partial_mol_smiles_list)
        motif_smiles_list = []
        for mid in labels.next_motif_ids:
            motif_smiles = ""
            if mid != self._end_motif_idx:
                motif_smiles = self._motif_vocab.at_id(mid)['smiles']
            motif_smiles_list.append(motif_smiles)
        self._motif_mol_graphs = tensorize_smiles_list(motif_smiles_list)
        self._tensorize_labels(labels)

        self._encode_mol.eval()
        self._select_motif_mlp.eval()
        self._select_mol_attachment.eval()
        self._classify_mol_bond.eval()

        with torch.no_grad():
            pred = self._run_inference(batch)

            m1_accuracy = \
                (torch.argmax(pred.batched_motif_distr, dim=1) == labels.next_motif_labels).sum() / batch_size
            m21_accuracy = iou(pred.batched_partial_mol_candidates > 0.5, labels.batched_partial_mol_candidates > 0.5)
            m22_accuracy = iou(pred.batched_motif_candidates > 0.5, labels.batched_motif_candidates > 0.5)
            m3_accuracy = 0.0

            #
            # m3_accuracy = (
            #                       torch.argmax(pred.batched_bond_types, dim=1) == torch.argmax(
            #                   labels.batched_bond_labels,
            #                   dim=1)
            #               ).sum() / batch_size

            # Update tensorboard
            self._writer.add_scalars("accuracy", {
                "m1": m1_accuracy,
                "m21": m21_accuracy,
                "m22": m22_accuracy,
                "m3": m3_accuracy
            }, self._writer_step)

            # Log
            logging.info(f"Test run; Accuracy: "
                         f"SelectMotif: {m1_accuracy:.3f}, "
                         f"SelectMolAttachmentAB: {m21_accuracy:.3f}, "
                         f"SelectMolAttachmentBA: {m22_accuracy:.3f}, "
                         f"ClassifyMolBond: {m3_accuracy:.3f}")

    def _load_checkpoint(self, checkpoint_filepath: str):
        checkpoint = torch.load(checkpoint_filepath)
        self._epoch = checkpoint['epoch']

        self._encode_mol.load_state_dict(checkpoint['model']['encode_mol'])
        self._select_motif_mlp.load_state_dict(checkpoint['model']['select_motif_mlp'])
        self._select_mol_attachment.load_state_dict(checkpoint['model']['select_mol_attachment'])
        self._classify_mol_bond.load_state_dict(checkpoint['model']['classify_mol_bond'])

        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logging.info(f"Checkpoint loaded: {checkpoint_filepath}")

    def _save_checkpoint(self):
        checkpoint_path = path.join(self._checkpoints_dir, f"checkpoint-{self._epoch}.pt")
        torch.save({
            'epoch': self._epoch,
            'model': {
                'encode_mol': self._encode_mol.state_dict(),
                'select_motif_mlp': self._select_motif_mlp.state_dict(),
                'select_mol_attachment': self._select_mol_attachment.state_dict(),
                'classify_mol_bond': self._classify_mol_bond.state_dict()
            },
            'optimizer': self._optimizer.state_dict(),
            'lr_scheduler': self._lr_scheduler.state_dict(),
        }, checkpoint_path)
        logging.info(f"Saved checkpoint to: {checkpoint_path}")

        checkpoints = []
        for f in os.listdir(self._checkpoints_dir):
            f = path.join(self._checkpoints_dir, f)
            if f.endswith(".pt") and path.isfile(f) and not path.islink(f):
                checkpoints.append(f)
        checkpoints.sort(key=lambda f: path.getctime(f), reverse=True)  # Sort by creation time (descending)

        # Create a link to newly created checkpoint
        latest_filepath = path.join(self._checkpoints_dir, "checkpoint-latest.pt")
        if path.exists(latest_filepath):
            os.unlink(latest_filepath)
        os.symlink(checkpoints[0], latest_filepath)

        # Remove last checkpoint if max is exceeded
        if len(checkpoints) > 10:
            os.remove(checkpoints[-1])
            logging.debug(f"Removed old checkpoint: {checkpoints[-1]}")

    def _train_epoch(self):
        for i, batch in enumerate(self._training_dataloader):
            self._train_step(i, batch)

            if (i + 1) % 100 == 0:
                self._test()

            if (i + 1) % 10 == 0:
                self._writer.flush()
            self._writer_step += 1

        self._save_checkpoint()

    def train(self):
        logging.info("Training started...")

        while True:
            logging.info(f"---------------------------------------------------------------- Epoch {self._epoch}")

            self._train_epoch()
            self._epoch += 1


def _main():
    from argparse import ArgumentParser
    from pathlib import Path

    parser = ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--config-file", type=Path, required=True)

    args = parser.parse_args()

    if not path.isdir(args.train_dir):
        print(f"Invalid train directory: {args.train_dir}")
        exit(1)

    if not path.isfile(args.config_file):
        print(f"Invalid config file: {args.config_file}")
        exit(1)

    logging.info(f"Training directory: {args.train_dir}")
    logging.info(f"Loading config file: {args.config_file}")

    trainer = MolgenaReconstructTask(
        train_dir=args.train_dir,
        config_filepath=args.config_file
    )
    trainer.train()


if __name__ == "__main__":
    _main()
