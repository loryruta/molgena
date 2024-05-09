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
from motif_graph.tensorize_motif_graph import create_mgraph_node_feature_vector, tensorize_mgraphs
from motif_graph.cache import MgraphCache
from mol_graph import *
from motif_vocab import MotifVocab
from utils.misc_utils import *
from utils.tensor_utils import *
from model.molgena import Molgena
from annotations import Annotator, BatchedAnnotations


class Predictions:
    """ A dataclass to gather predictions done for the reconstruction task. """

    next_motif_distr: torch.LongTensor
    """ A tensor of shape (B,) where every element is the next Motif ID to select. """

    attachment_cluster_mask: torch.FloatTensor
    # TODO docstring

    cluster1_attachment_mask: torch.FloatTensor
    """ A tensor over batched cluster1 nodes where the i-th element is 1 if a node should be selected for the
    attachment. Nodes in the same orbit will all be set to one. """

    cluster2_attachment_mask: torch.FloatTensor
    """ A tensor over batched cluster2 nodes where an element is set to 1 if a node should be selected for the
    attachment. Nodes in the same orbit will all be set to one. """

    attachment_bond_types: torch.LongTensor
    """ A tensor over batched cluster1/cluster2 (same length) indicating the type of bond to be formed. """


class Loss:
    l1: torch.FloatTensor  # SelectMotifMlp
    l21: torch.FloatTensor  # SelectMolAttachment(M, z_m)
    l22: torch.FloatTensor  # SelectMolAttachment(m, z_M)
    l3: torch.FloatTensor  # ClassifyMolBond

    total_loss: torch.FloatTensor

    def validate(self):
        assert self.l1.ndim == 0
        assert self.l21.ndim == 0
        assert self.l22.ndim == 0
        assert self.l3.ndim == 0


class MolgenaReconstructTask:
    def __init__(self, train_dir: str, config_filepath: str):
        self._config_name = Path(config_filepath).stem
        self._config = load_json_with_vars(config_filepath)

        # Create ./runs/ directory
        self._runs_dir = path.join(train_dir, f"runs-{self._config_name}")
        if not path.exists(self._runs_dir):
            os.mkdir(self._runs_dir)
        logging.info(f"Runs directory: {self._runs_dir}")

        # Create ./checkpoints/ directory
        self._checkpoints_dir = path.join(train_dir, f"checkpoints-{self._config_name}")
        if not path.exists(self._checkpoints_dir):
            os.mkdir(self._checkpoints_dir)
        logging.info(f"Checkpoints directory: {self._runs_dir}")

        # Load datasets
        self._training_set = ZincDataset.training_set()
        logging.info(f"Training set loaded; Num molecules: {len(self._training_set)}")

        self._test_set = ZincDataset.test_set()
        logging.info(f"Test set loaded; Num molecules: {len(self._test_set)}")

        self._motif_vocab = MotifVocab.load()
        logging.info(f"Motif vocabulary loaded; Num motifs: {len(self._motif_vocab)}")

        self._batch_size = 256
        self._test_batch_size = 1024

        self._training_dataloader = DataLoader(self._training_set, batch_size=self._batch_size,
                                               collate_fn=lambda batch: self._collate_fn(batch))

        self._num_motifs = len(self._motif_vocab)
        self._end_motif_idx = self._num_motifs

        # Create annotator
        self._annotator = Annotator()

        # Create model
        self._molgena = Molgena(self._config)

        # Create optimizer and LR scheduler
        num_parameters = sum([param.numel() for param in self._molgena.parameters()])
        logging.info(f"Total num parameters: {num_parameters}")

        self._optimizer = torch.optim.Adam(self._molgena.parameters(), lr=0.001)
        self._lr_scheduler = CosineAnnealingLR(self._optimizer, T_max=50)
        logging.info(f"Optimizer ready")

        # Model debug variables
        self._no_random = False  # If set, disable all random choices within the training
        self._only_first_batch = False  # If set, only use the first batch (always the same)

        self._epoch = 0
        self._run_iteration = 0

        # Create tensorboard writer
        self._writer = SummaryWriter(log_dir=self._runs_dir)

        # Load latest checkpoint if any
        latest_checkpoint_path = path.join(self._checkpoints_dir, "checkpoint-latest.pt")
        if path.exists(latest_checkpoint_path):
            self._load_checkpoint(latest_checkpoint_path)
        else:
            logging.info("Checkpoint not found, starting fresh...")

    def _collate_fn(self, raw_batch: List[Tuple[int, str]]) -> List[str]:
        mol_smiles_list = [mol_smiles for _, mol_smiles in raw_batch]
        return mol_smiles_list

    def _run_inference(self, batch: List[str], annotations: BatchedAnnotations) -> Predictions:
        mol_smiles_list = batch
        batch_size = len(batch)

        # Prepare data for inference
        mol_tensor_molgraphs = tensorize_smiles_list(mol_smiles_list)
        pmol_smiles_list = annotations.pmol_smiles_list
        pmol_tensor_molgraphs = tensorize_smiles_list(pmol_smiles_list)

        # Prepare attachment data for inference
        attachment_pmol_mgraphs = annotations.attachment_pmol_mgraphs
        attachment_pmol_tensor_mgraphs, _ = tensorize_mgraphs(
            attachment_pmol_mgraphs, self._motif_vocab, return_node_mappings=False)
        attachment_tmol_tensor_molgraphs = tensorize_smiles_list(annotations.attachment_tmol_smiles_list)
        num_attachments = annotations.num_attachments
        cluster2_mreprs = torch.stack(
            [create_mgraph_node_feature_vector(mid) for mid in annotations.cluster2_motif_ids], dim=0)
        cluster1_molgraphs = tensorize_smiles_list(
            [self._motif_vocab.at_id(mid)['smiles'] for mid in annotations.cluster1_motif_ids])
        cluster2_molgraphs = tensorize_smiles_list(
            [self._motif_vocab.at_id(mid)['smiles'] for mid in annotations.cluster2_motif_ids])
        cluster1_attachment_node_indices = annotations.cluster1_attachment_node_indices
        cluster2_attachment_node_indices = annotations.cluster2_attachment_node_indices

        # *** INFERENCE ***

        pred = Predictions()

        # Encoding
        mol_reprs = self._molgena.mod_encode_mol(mol_tensor_molgraphs, batch_size)
        pmol_reprs = self._molgena.mod_encode_mol(pmol_tensor_molgraphs, batch_size)
        self._molgena.mod_encode_mgraph(attachment_pmol_tensor_mgraphs, num_attachments)  # Compute node_hiddens
        attachment_tmol_reprs = self._molgena.mod_encode_mol(attachment_tmol_tensor_molgraphs, num_attachments)
        cluster1_reprs = self._molgena.mod_encode_mol(cluster1_molgraphs, num_attachments)
        cluster2_reprs = self._molgena.mod_encode_mol(cluster2_molgraphs, num_attachments)

        # SelectMotifMlp
        pred.next_motif_distr = self._molgena.mod_select_motif_mlp(pmol_reprs, mol_reprs)  # (B, 4331)

        # TODO handle the case where all batch pmol are empty/full

        # SelectAttachmentClusters
        pred.attachment_cluster_mask = self._molgena.mod_select_attachment_clusters(
            attachment_pmol_tensor_mgraphs, cluster2_mreprs)

        # SelectAttachmentAtom(cluster1)
        pred.cluster1_attachment_mask = self._molgena.mod_select_attachment_cluster1_atom(
            cluster1_molgraphs, cluster2_reprs, attachment_tmol_reprs)

        # SelectAttachmentAtom(cluster2)
        pred.cluster2_attachment_mask = self._molgena.mod_select_attachment_cluster2_atom(
            cluster2_molgraphs, cluster1_reprs, attachment_tmol_reprs)

        # SelectAttachmentBondType
        cluster1_atom_hiddens = cluster1_molgraphs.node_hiddens[cluster1_attachment_node_indices]
        cluster2_atom_hiddens = cluster2_molgraphs.node_hiddens[cluster2_attachment_node_indices]
        pred.attachment_bond_types = self._molgena.mod_select_attachment_bond_type(
            cluster1_atom_hiddens, cluster2_atom_hiddens, attachment_tmol_reprs)

        return pred

    def _train_step(self, batch_idx: int, batch: List[str]):
        # *** ANNOTATE ***

        annotations = self._annotator.create_batched_annotations(batch)

        # *** INFERENCE ***

        torch.set_grad_enabled(True)
        self._molgena.train(True)

        self._optimizer.zero_grad()

        pred = self._run_inference(batch, annotations)

        # *** COMPUTE LOSS ***

        a1 = .1
        a2 = 1.
        a31 = 1.
        a32 = 1.
        a4 = 1.

        l1 = a1 * F.cross_entropy(pred.next_motif_distr, annotations.next_motif_ids)
        l2 = a2 * F.binary_cross_entropy(pred.attachment_cluster_mask, annotations.attachment_cluster_mask)
        l31 = a31 * F.binary_cross_entropy(pred.cluster1_attachment_mask, annotations.cluster1_attachment_mask)
        l32 = a32 * F.binary_cross_entropy(pred.cluster2_attachment_mask, annotations.cluster2_attachment_mask)
        l4 = a4 * F.cross_entropy(pred.attachment_bond_types, annotations.attachment_bond_types)

        loss = l1 + l2 + l31 + l32 + l4
        loss.backward()

        # *** STEP ***

        self._optimizer.step()

        lr = self._lr_scheduler.get_last_lr()[0]
        self._lr_scheduler.step()

        # *** LOGGING ***

        # Update tensorboard
        self._writer.add_scalars("loss", {
            "l1": l1.item(),
            "l2": l2.item(),
            "l31": l31.item(),
            "l32": l32.item(),
            "l4": l4.item()
        }, self._run_iteration)

        # Log on console
        num_batches = len(self._training_set) // self._batch_size
        logging.debug(f"Batch {batch_idx:>3}/{num_batches:>3} Inference run; "
                      f"L1: {l1.item()}, "
                      f"L2: {l2.item():.8f}, "
                      f"L31: {l31.item():.8f}, "
                      f"L32: {l32.item():.8f}, "
                      f"L4: {l4.item():.8f}, "
                      f"Total loss: {loss:.5f}, "
                      f"LR: {lr:.7f}")

    def _test_step(self):
        batch_size = self._test_batch_size

        batch = self._test_set.df.sample(n=batch_size)['smiles'].tolist()

        # *** ANNOTATE ***

        annotations = self._annotator.create_batched_annotations(batch)

        # *** INFERENCE ***

        torch.set_grad_enabled(True)
        self._molgena.train(False)

        pred = self._run_inference(batch, annotations)

        # *** METRICS ***

        pred_next_motif_ids = torch.argmax(pred.next_motif_distr, dim=1)
        m1 = (pred_next_motif_ids == annotations.next_motif_ids).sum() / batch_size
        m2 = iou(pred.attachment_cluster_mask > .5, annotations.attachment_cluster_mask > .5)
        m31 = iou(pred.cluster1_attachment_mask > .5, annotations.cluster1_attachment_mask > .5)
        m32 = iou(pred.cluster2_attachment_mask > .5, annotations.cluster2_attachment_mask > .5)
        pred_attachment_bond_types = torch.argmax(pred.attachment_bond_types, dim=1)
        m4 = (pred_attachment_bond_types == annotations.attachment_bond_types).sum() / batch_size

        # *** LOGGING ***

        # Update tensorboard
        self._writer.add_scalars("accuracy", {
            "m1": m1,
            "m2": m2,
            "m31": m31,
            "m32": m32,
            "m4": m4
        }, self._run_iteration)

        # Log on console
        logging.info(f"Test run; Accuracy: "
                     f"M1: {m1:.3f}, "
                     f"M2: {m2:.3f}, "
                     f"M31: {m31:.3f}, "
                     f"M32: {m32:.3f}, "
                     f"M4: {m4:.3f}")

    def _load_checkpoint(self, checkpoint_filepath: str):
        checkpoint = torch.load(checkpoint_filepath)
        self._epoch = checkpoint['epoch']
        self._molgena.load_state_dict(checkpoint['model'])
        self._optimizer.load_state_dict(checkpoint['optimizer'])
        self._lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])

        logging.info(f"Checkpoint loaded: {checkpoint_filepath}")

    def _save_checkpoint(self):
        checkpoint_path = path.join(self._checkpoints_dir, f"checkpoint-{self._epoch}.pt")
        torch.save({
            'epoch': self._epoch,
            'model': self._molgena,
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

            self._run_iteration += 1

            if self._run_iteration % 20 == 0:
                self._writer.flush()
                logging.debug("Flushed tensorboard writes")

            if self._run_iteration % 500 == 0:
                self._save_checkpoint()

            if self._only_first_batch:
                break

            # If "_only_first_batch", don't test
            if self._run_iteration % 100 == 0:
                self._test_step()

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
