from common import *
import pytest
from mol_graph import *
from mol_dataset import ZincDataset
from motif_vocab import MotifVocab
from model.encode_mol_mpn import EncodeMolMPN
from model.encode_mol import EncodeMol
from model.select_motif_mlp import SelectMotifMlp
from model.select_mol_attachment import SelectMolAttachment
from model.classify_mol_bond import ClassifyMolBond
from test_utils import *
from utils.tensor_utils import *


@pytest.fixture(scope="session", autouse=True)
def mol_dataset():
    dataset = ZincDataset.training_set()
    logging.info("ZINC dataset loaded")
    return dataset


@pytest.fixture(scope="session", autouse=True)
def motif_vocab():
    motif_vocab = MotifVocab.load()
    logging.info("Motif vocabulary loaded")
    return motif_vocab


# TODO ignore all inference tests as modules are subject to frequent changes (under development)
pytest.skip(allow_module_level=True)


def test_encode_mol_mpn(mol_dataset):
    BATCH_SIZE = 5

    smiles_list = mol_dataset.df.sample(n=BATCH_SIZE)['smiles'].to_list()

    mol_graph = tensorize_smiles_list(smiles_list)
    mol_graph.create_hiddens(200, 100)

    logging.info("Initializing EncodeMolMPN module...")
    module = EncodeMolMPN({
        'num_steps': 8,
        'node_features_dim': 5,
        'edge_features_dim': 1,
        'node_hidden_dim': 200,
        'edge_hidden_dim': 100
    })
    module(mol_graph)  # Inference
    logging.info("Inference done!")


def test_encode_mol(mol_dataset):
    BATCH_SIZE = 5
    MOL_REPR_DIM = 200

    smiles_list = mol_dataset.df.sample(n=BATCH_SIZE)['smiles'].to_list()

    mol_graph = tensorize_smiles_list(smiles_list)
    mol_graph.create_hiddens(MOL_REPR_DIM, 50)
    assert mol_graph.batch_size() == BATCH_SIZE

    logging.info("Initializing EncodeMol module...")
    module = EncodeMol({
        'num_steps': 8,
        'node_features_dim': 5,
        'edge_features_dim': 1,
        'node_hidden_dim': MOL_REPR_DIM,
        'edge_hidden_dim': 50
    })
    mol_repr = module(mol_graph, BATCH_SIZE)
    assert mol_repr.shape == (BATCH_SIZE, MOL_REPR_DIM)
    logging.info("Inference done!")


def test_select_motif_mlp():
    BATCH_SIZE = 100
    MOL_REPR_DIM = 200

    partial_mol_reprs = torch.randn((BATCH_SIZE, MOL_REPR_DIM))
    recon_mol_reprs = torch.randn((BATCH_SIZE, MOL_REPR_DIM))

    # Test reconstruction mode (partial_mol, recon_mol)
    module = SelectMotifMlp({
        'mol_repr_dim': MOL_REPR_DIM,
        'num_motifs': 4331 + 1,
        'reconstruction_mode': True
    })
    output = module(partial_mol_reprs, recon_mol_reprs)
    assert output.shape == (BATCH_SIZE, 4331 + 1)

    # Invalid call (partial_mol required)
    with pytest.raises(Exception) as _:
        output = module(None, recon_mol_reprs)
        assert output.shape == (BATCH_SIZE, 4331 + 1)

    # Test optimization mode (partial_mol, None)
    module = SelectMotifMlp({
        'mol_repr_dim': MOL_REPR_DIM,
        'num_motifs': 4331 + 1,
        'reconstruction_mode': False
    })
    output = module(partial_mol_reprs, None)
    assert output.shape == (BATCH_SIZE, 4331 + 1)

    # Invalid call (partial_mol required)
    with pytest.raises(Exception) as _:
        module(None, None)


def test_select_mol_attachment(mol_dataset):
    BATCH_SIZE = 5
    MOL_REPR_DIM = 200

    mol_a_reprs = torch.randn((BATCH_SIZE, MOL_REPR_DIM))

    mol_b_smiles_list = mol_dataset.df.sample(n=BATCH_SIZE)['smiles'].to_list()
    mol_b_graphs = tensorize_smiles_list(mol_b_smiles_list)
    mol_b_graphs.create_hiddens(60, 30)

    module = SelectMolAttachment({
        'num_mpn_steps': 8,
        'mol_a_repr_dim': MOL_REPR_DIM,
        'mol_b_node_features_dim': 5,
        'mol_b_edge_features_dim': 1,
        'mol_b_node_hidden_dim': 60,
        'mol_b_edge_hidden_dim': 30
    })
    output = module(mol_a_reprs, mol_b_graphs)
    assert output.shape == (mol_b_graphs.num_nodes(),)
    assert_01_tensor(output)


def test_classify_mol_bond(mol_dataset, motif_vocab):
    import random

    BATCH_SIZE = 10

    # random.seed(SEED)

    mol_smiles_list = mol_dataset.df.sample(n=BATCH_SIZE)['smiles'].tolist()
    motif_smiles_list = motif_vocab.df_id.sample(n=BATCH_SIZE)['smiles'].tolist()

    mol_graphs = tensorize_smiles_list(mol_smiles_list)
    motif_mol_graphs = tensorize_smiles_list(motif_smiles_list)

    assert mol_graphs.batch_size() == motif_mol_graphs.batch_size() == BATCH_SIZE

    # Sample proposed_bonds being careful atoms of generated pairs belong to the same batch
    def sample(n: int, max_k: Optional[int] = None):
        """ Samples *at most* K different elements from a list ranging from 0 to N - 1. """
        if max_k is None:
            max_k = n
        num_samples = random.randint(1, min(max_k, n))  # 1! Won't return an empty list
        seq = list(range(n))
        return random.sample(seq, min(num_samples, n))

    logging.info("Sampling proposed bonds...")

    _, mol_atom_counts = torch.unique_consecutive(mol_graphs.batch_indices, return_counts=True)
    mol_batch_offsets = exclusive_prefix_sum(mol_atom_counts)

    _, motif_atom_counts = torch.unique_consecutive(motif_mol_graphs.batch_indices, return_counts=True)
    motif_batch_offsets = exclusive_prefix_sum(motif_atom_counts)

    proposed_bonds = []
    for batch_idx in range(BATCH_SIZE):
        mol_atom_count = mol_atom_counts[batch_idx]
        mol_atom_indices = torch.tensor(  # Batch -relative indices
            sample(mol_atom_count, max(mol_atom_count // 2, 1)),
            dtype=torch.long)
        mol_atom_indices += mol_batch_offsets[batch_idx]  # Inter-batch indices
        assert (mol_atom_indices < mol_graphs.num_nodes()).all()  # Validate the algorithm is correct

        motif_atom_count = motif_atom_counts[batch_idx]
        motif_atom_indices = torch.tensor(  # Batch -relative indices
            sample(motif_atom_count, max(motif_atom_count // 2, 1)),
            dtype=torch.long)
        motif_atom_indices += motif_batch_offsets[batch_idx]  # Inter-batch indices
        assert (motif_atom_indices < motif_mol_graphs.num_nodes()).all()  # Validate the algorithm is correct

        proposed_bonds.append(torch.cartesian_prod(mol_atom_indices, motif_atom_indices))
    proposed_bonds = torch.cat(proposed_bonds).t()  # (2, ...)
    num_proposed_bonds = proposed_bonds.shape[1]
    logging.info(f"Sampled {num_proposed_bonds} proposed bonds")

    # Run inference
    logging.info(f"Running inference...")

    module = ClassifyMolBond({
        'num_steps': 8,
        'atom_features_dim': 5,
        'bond_features_dim': 1,
        'atom_hidden_dim': 60,
        'bond_hidden_dim': 30,
    })
    output = module(mol_graphs, motif_mol_graphs, proposed_bonds)
    assert output.shape == (num_proposed_bonds, 4)
    assert_normalized_output(output, dim=1)

    logging.info(f"Done!")


def test_node_isomorphism():
    encode_mol = EncodeMol({
        "num_steps": 40,
        "node_features_dim": 5,
        "edge_features_dim": 1,
        "node_hidden_dim": 8,
        "edge_hidden_dim": 16,
        "W1": [32],
        "W2": [32],
        "W3": [32],
        "U1": [32],
        "U2": [32]
    })

    # Encode a Carbon ring, expect all node hiddens to be equal
    smiles = "C1CCCCC1"
    mol_graph = tensorize_smiles_list([smiles])
    encode_mol(mol_graph, 1)
    node_hidden_ref = mol_graph.node_hiddens[0]
    assert torch.all(torch.all(torch.eq(mol_graph.node_hiddens, node_hidden_ref), dim=0))

    # Encode a different molecule, node hiddens should be different
    smiles = "CC(C)c1nc(N2CC[C@H](C)C2)sc1C=O"
    mol_graph = tensorize_smiles_list([smiles])
    encode_mol(mol_graph, 1)
    node_hidden_ref = mol_graph.node_hiddens[0]
    assert not torch.all(torch.all(torch.eq(mol_graph.node_hiddens, node_hidden_ref), dim=0))

