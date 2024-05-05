import pytest
from mol_dataset import ZincDataset
from model.encode_mol import EncodeMol
from motif_graph.construct_motif_graph import *
from motif_graph.tensorize_motif_graph import *
from utils.misc_utils import *


@pytest.mark.skip(reason="Slow")
def test_unique_node_features():
    motif_vocab = MotifVocab.load()
    motif_vocab_len = len(motif_vocab)

    feature_vectors = []

    dt = 0.0
    for mid in range(motif_vocab_len):
        stopwatch_ = stopwatch()
        feature_vector = create_mgraph_node_feature_vector(mid)
        dt += stopwatch_()
        feature_vectors.append(feature_vector)
    dt /= motif_vocab_len

    logging.info(f"Node features calculated for all Motif ID(s); avg: {dt_str(dt)}")

    logging.info(f"Checking pairwise equalities...")

    stopwatch_ = stopwatch()
    for mid1 in range(motif_vocab_len):
        for mid2 in range(motif_vocab_len):
            if mid1 == mid2:
                continue
            assert not (feature_vectors[mid1] == feature_vectors[mid2]).all()
        if stopwatch_() > 2.0:
            logging.info(f"Checked {mid1 + 1}/{motif_vocab_len} Motif ID(s)...")
            stopwatch_ = stopwatch()


@pytest.mark.skip(reason="Slow")
def test_node_features_repeatability():
    motif_vocab = MotifVocab.load()
    motif_vocab_len = len(motif_vocab)
    for mid in range(motif_vocab_len):
        assert (create_mgraph_node_feature_vector(mid) == create_mgraph_node_feature_vector(mid)).all()


def test_tensorize_mgraph():
    motif_vocab = MotifVocab.load()

    mol_smiles = "Cc1cc2c(c(=O)n1CCc1ccc(O)c(O)c1)[C@@H](c1ccccc1Cl)C(C#N)=C(N)O2"
    motif_graph = construct_motif_graph(mol_smiles, motif_vocab)

    stopwatch_ = stopwatch_str()

    tensor_motif_graph = tensorize_mgraph(motif_graph, motif_vocab)

    logging.info(f"Motif graph tensorized in {stopwatch_()}")

    assert tensor_motif_graph.node_features.shape[0] == len(motif_graph.nodes)
    assert tensor_motif_graph.edge_features.shape[0] == len(motif_graph.edges)
    assert tensor_motif_graph.edges.shape == (2, len(motif_graph.edges))


def test_fixed_mgraph_node_features():
    """ Tests that the algorithm used for hand-crafted node features for mgraph doesn't change. """

    ref = torch.tensor([0.4205, -0.8754, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        0.4073, -0.3797, 0.0000, 0.0000, -0.7762, 0.0000, 0.0000, 0.0000,
                        0.0000, 0.0000, 0.4518, 0.0000, 0.0000, 0.0000, 0.0000, 0.0000,
                        -0.4193, 0.0000, 0.8507, 0.1668, -0.6179, 0.0000, -0.3659, 0.0000,
                        0.2367, -0.1146, 0.0000, 0.0000, 0.9371, -0.7819, -0.1901, 0.0000,
                        -1.1100, -0.5122, 1.4346, 0.0000, 0.0000, 0.9356, 0.0000, 0.0000,
                        0.0000, -0.1100, 0.0000, -0.8135, 0.0000, 0.5159, 0.0000, 0.0000,
                        0.0000, 0.6064, 0.0000, 0.0000, 0.0000, 0.9332, -0.1689, 0.0000], dtype=torch.float32)
    assert (torch.round(create_mgraph_node_feature_vector(0), decimals=4) == ref).all()


@pytest.fixture(scope="module")
def mgraph_encoder():
    return EncodeMol({  # We use the same network for encoding molecules and mgraph(s)
        'num_steps': 8,
        'node_features_dim': 64,
        'edge_features_dim': 11,
        'node_hidden_dim': 8,
        'edge_hidden_dim': 4,
        'W1': [96],
        'W2': [96],
        'W3': [96],
        'U1': [96],
        'U2': [96],
    })


def test_mgraph_automorphism(mgraph_encoder):
    motif_vocab = MotifVocab.load()

    dataset = ZincDataset.all()  # Consider all training/validation/test sets
    dataset_len = len(dataset)

    # Number of mgraph(s) for which node_hiddens of two different nodes, after inference, results the same
    num_eq_nh_mgraphs = 0

    stopwatch_ = stopwatch()

    for i, mol_smiles in dataset:
        mgraph = construct_motif_graph(mol_smiles, motif_vocab)
        tensor_mgraph = batch_tensor_graphs([tensorize_mgraph(mgraph, motif_vocab)])
        with torch.no_grad():
            mgraph_encoder(tensor_mgraph, 1)
        assert tensor_mgraph.node_hiddens.shape[1] == 8
        assert tensor_mgraph.edge_hiddens.shape[1] == 4

        for node_hidden in tensor_mgraph.node_hiddens:
            atom_indices = \
                torch.nonzero(torch.isclose(tensor_mgraph.node_hiddens, node_hidden).all(dim=1))
            assert atom_indices.numel() >= 1
            if atom_indices.numel() > 1:
                # logging.warning(f"Found #{i} whose mgraph has at least one automorphism; "
                #                 f"SMILES: {mol_smiles}")
                num_eq_nh_mgraphs += 1
                break

        if stopwatch_() > 2.:
            logging.debug(f"Processed {i}/{dataset_len} mgraph(s) of dataset molecules...; "
                          f"Num eq NH: {num_eq_nh_mgraphs} mgraphs/{i} ({num_eq_nh_mgraphs / i * 100.:.1f}%)")
            stopwatch_ = stopwatch()


@pytest.mark.skip(reason="Visualization test")
def test_visualize_mgraph_automorphisms(mgraph_encoder):
    import matplotlib.pyplot as plt

    def int_to_color(value: int):
        colors = plt.cm.get_cmap("tab20").colors
        return colors[value % len(colors)]

    def hash_tensor(tensor):
        return hash(tuple(tensor.reshape(-1).tolist()))

    motif_vocab = MotifVocab.load()

    mol_smiles_list = [
        "C[C@H]1CC[C@@H](C)C[NH+]1CC(=O)NCCc1ccccc1",
        "COc1cc(C#N)ccc1OCC(=O)c1cc(C)c(C)c(C)c1C",
        "CC1(C)Cc2cccc(OCC(=O)NCCc3c[nH]c4ccccc34)c2O1",
        "COc1cccc(-n2cc(C(=O)[O-])c3c2[C@H](c2cc(OC)c(OC)c(OC)c2)CC(=O)N3)c1",
        "C[C@H](CS(C)(=O)=O)NC(=O)C1(c2ccc(F)cc2F)CCOCC1",
        "COc1cccc([C@@H](O)[C@H]2OC(=O)c3ccccc32)c1OC",
        "COc1ccc([N+](=O)[O-])cc1-c1ccc(C=C2C(=O)c3ccccc3C2=O)o1",
        "COc1ccc(CNc2ncnc(Nc3ccccc3OC)c2N)cc1",
        "Fc1ccc(F)c2c1CCN(Cc1nnsc1Cl)C2",
        "COC(C)(C)C[C@@H](C)[NH2+][C@H]1CCc2[nH]c3ccccc3c2C1",
        "C[NH+](C)C[C@H](O)Cc1nc(C(C)(C)C)cs1",
        "CN(c1nc(N2CCOCC2)nc(N2CCOCC2)n1)n1cccc1",
        "COc1ccc(-c2n[nH]cc2/C=C(/C#N)C(=O)C(C)(C)C)cc1OC",
        "CCN(C)C(=O)c1cc(OC)c(OCc2ccccc2)c(OC)c1",
        "Cc1nc(C(C)(C)C)nc2sc(C(=O)NC[C@@H]3CCOC3)c(C)c12",
        "CC1=NN(Cc2ccccc2Cl)C(=O)[C@]12Cc1c(C)nn(-c3ccccc3)c1N1CCCC[C@@H]12",
        "CC[NH2+][C@@H](C(C)C)[C@@H](C)C1CCCCC1",
        "CCCC(CCC)/C([O-])=N/S(=O)(=O)c1ccc2c(c1)CCCO2",
        "O=C1N=C2S[C@@H](c3ccc(F)cc3)[C@@H]3C(=O)Oc4ccccc4[C@H]3[C@@H]2S1",
        "CC1(C)C(NS(=O)(=O)CC[NH2+]C2CC2)C1(C)C",
        "CC(C)NC(=O)COC(=O)c1cc(C2CC2)nc2ccc(Cl)cc12",
        "CC(C)(COC[C@@H]1CCCO1)C(=O)NN",
        "CCC[NH+]1CCC(NC(=O)/C=C(/C)C(C)(C)C)CC1",
        "Cc1ccc(Cn2nc(C)c(CN3CCC[C@@H](c4nncn4C)C3)c2Cl)cc1",
        "Cc1ccc(-n2cnnc2SC(C)C)cc1Cl"
    ]
    num_mol_smiles = len(mol_smiles_list)

    num_cols = 6

    for i, mol_smiles in enumerate(mol_smiles_list):
        plt.subplot(ceil(num_mol_smiles / num_cols), num_cols, i + 1)

        mgraph = construct_motif_graph(mol_smiles, motif_vocab)
        tensor_mgraph = batch_tensor_graphs([tensorize_mgraph(mgraph, motif_vocab)])
        mgraph_encoder(tensor_mgraph, 1)

        rounded_node_hiddens = torch.round(tensor_mgraph.node_hiddens, decimals=7)

        # Set color node attribute based on the hash of its node_hidden
        cid = 0
        for rounded_node_hidden in rounded_node_hiddens:
            mgraph.nodes[cid]['color'] = int_to_color(hash_tensor(rounded_node_hidden))
            cid += 1

        motif_ids = nx.get_node_attributes(mgraph, 'motif_id')
        color_map = nx.get_node_attributes(mgraph, 'color')
        nx.draw(mgraph,
                node_color=[color_map[node] for node in mgraph.nodes],  # Obtain colors in networkx nodes' order
                with_labels=True,
                labels=motif_ids)
    plt.show()
