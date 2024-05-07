import networkx as nx
import pytest
from typing import *
from tensor_graph import TensorGraph
from motif_graph.construct_motif_graph import construct_motif_graph
from motif_graph.tensorize_motif_graph import tensorize_mgraph
from motif_vocab import MotifVocab
from model.encode_mol import EncodeMol
from tensor_graph import find_node_orbit


@pytest.fixture(scope="module")
def orbits_detector():
    return EncodeMol({
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


def test_find_node_orbit(orbits_detector):
    """ Tests find_node_orbit() function on some known mgraph(s). """

    # We test a list of SMILES for which we know their mgraph has certain node orbits.
    # Using `test_visualize_mgraph_automorphisms`, we have visualized the node indices and orbits, and we expect
    # find_node_orbit to correctly detect them

    motif_vocab = MotifVocab.load()

    # SMILES 0
    mol_smiles = "C[C@H]1CC[C@@H](C)C[NH+]1CC(=O)NCCc1ccccc1"
    mgraph = tensorize_mgraph(construct_motif_graph(mol_smiles, motif_vocab), motif_vocab)
    assert set(find_node_orbit(mgraph, 0, orbits_detector)) == {0, 2}
    assert set(find_node_orbit(mgraph, 3, orbits_detector)) == {3}
    assert set(find_node_orbit(mgraph, 1, orbits_detector)) == {1}

    # SMILES 1
    mol_smiles = "COc1cc(C#N)ccc1OCC(=O)c1cc(C)c(C)c(C)c1C"
    mgraph = tensorize_mgraph(construct_motif_graph(mol_smiles, motif_vocab), motif_vocab)
    assert set(find_node_orbit(mgraph, 1, orbits_detector)) == {3, 1, 4, 2}
    assert set(find_node_orbit(mgraph, 4, orbits_detector)) == {3, 1, 4, 2}
    assert set(find_node_orbit(mgraph, 8, orbits_detector)) == {8}
    assert set(find_node_orbit(mgraph, 0, orbits_detector)) == {0}

    # SMILES 2
    mol_smiles = "CC1(C)Cc2cccc(OCC(=O)NCCc3c[nH]c4ccccc34)c2O1"
    mgraph = tensorize_mgraph(construct_motif_graph(mol_smiles, motif_vocab), motif_vocab)
    assert set(find_node_orbit(mgraph, 0, orbits_detector)) == {2, 0}
    assert set(find_node_orbit(mgraph, 1, orbits_detector)) == {1}
    assert set(find_node_orbit(mgraph, 3, orbits_detector)) == {3}

    # SMILES 17
    mol_smiles = "CCCC(CCC)/C([O-])=N/S(=O)(=O)c1ccc2c(c1)CCCO2"
    mgraph = tensorize_mgraph(construct_motif_graph(mol_smiles, motif_vocab), motif_vocab)
    assert set(find_node_orbit(mgraph, 2, orbits_detector)) == {1, 2, 4, 5}  # 7?
    assert set(find_node_orbit(mgraph, 10, orbits_detector)) == {10, 11}
    assert set(find_node_orbit(mgraph, 0, orbits_detector)) == {0, 6}
    assert set(find_node_orbit(mgraph, 3, orbits_detector)) == {3}
