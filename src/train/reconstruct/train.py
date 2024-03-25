from common import *
from typing import *
import torch
import pandas as pd
from torch.utils.data import DataLoader
from mol_dataset import ZincDataset
from model.molgena import Molgena
from mol_graph import create_mol_graph_from_smiles, create_tensor_graph_from_smiles_list
from tensor_graph import batch_tensor_graphs, TensorGraph
from utils.tensor_utils import exclusive_prefix_sum
from utils.chem_utils import attach_molecules


def _prepare_motif_tensor_vocab(motif_vocab: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(
        [create_mol_graph_from_smiles(motif_smiles) for motif_smiles in motif_vocab['smiles']]
    )


def collate_fn(mol_batch):
    batched_tensor_graphs = batch_tensor_graphs([tensor_graph for _, tensor_graph, _ in mol_batch])
    return (
        [(mol_smiles, mol_motif_graph) for mol_smiles, _, mol_motif_graph in mol_batch],
        batched_tensor_graphs
    )


def _make_batch_relative_indices(input_: torch.LongTensor,
                                 batch_indices: torch.LongTensor) -> Tuple[torch.LongTensor, torch.LongTensor]:
    """ Given an input tensor of indices pointing to batch_indices elements (from TensorGraph), makes every index
    relative to its batch.

    A practical use case is making returned attachment indices relative to their batch, for performing the attachment
    with rdkit.
    """
    assert input_.dim() == 1 and batch_indices.dim() == 1
    assert torch.min(input_) >= 0 and torch.max(input_) < len(batch_indices)

    # Calculate at which atom index every batch starts
    _, mol_batch_lengths = torch.unique_consecutive(batch_indices, return_counts=True)
    batch_offsets = exclusive_prefix_sum(mol_batch_lengths)

    # For every mol attachment atom, obtain its batch index
    # For example:
    # Starting from: [0, 11, 12, 23, 45, 58, 88, 89, 90, 121]
    # Obtain: [0, 0, 0, 0, 1, 1, 2, 2, 2, 3]
    batch_indices_over_input = torch.index_select(batch_indices, dim=0, index=input_)

    # For every mol attachment atom, obtain its batch offset:
    # Input (batch offsets): [0, 40, 87, 110]
    # Index: [0, 0, 0, 0, 1, 1, 2, 2, 2, 3]
    # Output: [0, 0, 0, 0, 40, 40, 87, 87, 87, 110]
    batch_offsets_over_input = torch.gather(batch_offsets, dim=0, index=batch_indices_over_input)

    # Subtract to obtain batch-relative atom indices, that we can practically use to attach mol and motif
    return input_ - batch_offsets_over_input, batch_indices_over_input


def _main():
    training_set = ZincDataset.training_set()
    validation_set = ZincDataset.validation_set()

    training_dataloader = DataLoader(training_set, batch_size=256, collate_fn=collate_fn)
    # validation_dataloader = DataLoader(validation_set, batch_size=64)

    motif_vocab = pd.read_csv(MOTIF_VOCAB_CSV)
    motif_tensor_vocab = _prepare_motif_tensor_vocab(motif_vocab)

    model = Molgena(reconstruction_mode=True)

    partial_mol_smiles_list: Optional[List[str]] = None

    for batch_num, (mol_smiles_list, mol_graphs_list, mol_graphs) in enumerate(training_dataloader):
        # mol_smiles_and_motif_graph_list: List of mol SMILES and Motif graph
        # mol_tensor_graphs: Mol TensorGraph(s) merged into a single batched TensorGraph

        if partial_mol_smiles_list is None:
            # Initialization: the partial molecule is initialized with the first selected motif
            recon_mol_graphs = model.encode(mol_graphs)
            selected_motif_distr = model.pick_next_mol_graphs(None, recon_mol_graphs)
            selected_motif_indices = torch.argmax(selected_motif_distr, dim=1).squeeze()  # (B,)
            motif_smiles_list = motif_vocab.iloc[selected_motif_indices.cpu()]['smiles']
            partial_mol_smiles_list = motif_smiles_list
            # TODO Mark the motif as used
        else:
            partial_mol_graphs = create_tensor_graph_from_smiles_list(partial_mol_smiles_list)

            # Update: attach motifs iteratively to the partial molecule
            output = model.pick_next_motif_and_attachment(partial_mol_graphs, motif_tensor_vocab, mol_graphs)

            motif_smiles_list = motif_vocab.iloc[output.selected_motif_indices.cpu()]['smiles']

            # Attachment is an array (NC, 3) where dim=1 is [mol atom index, motif atom index, bond type]
            attachment = output.attachment

            attachment_eval = eval_motif_attachment(attachment)

            attachment[:, 0], attachment_mol_batches = \
                _make_batch_relative_indices(attachment[:, 0], partial_mol_graphs.batch_indices)

            attachment[:, 1], attachment_motif_batches = \
                _make_batch_relative_indices(attachment[:, 1], output.selected_motif_graphs.batch_indices)

            assert attachment_mol_batches == attachment_motif_batches  # Bonds should be formed on the same batch!

            # Iterate over every partial molecule and attach the selected motif to it, using the predicted attachments
            batch_indices, batch_lengths = torch.unique_consecutive(attachment_mol_batches)
            batch_offsets = exclusive_prefix_sum(batch_lengths)

            for batch_idx, batch_length, batch_offset in zip(batch_indices, batch_lengths, batch_offsets):
                motif_attachment = attachment[batch_offset:batch_offset + batch_length]

                partial_mol_smiles_list[batch_idx] = attach_molecules(
                    partial_mol_smiles_list[batch_idx],
                    motif_smiles_list[batch_idx],
                    motif_attachment.tolist()
                )


if __name__ == "__main__":
    _main()
