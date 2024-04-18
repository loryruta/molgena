from mol_graph import *


def test_empty_smiles():
    """ Tests the creation of TensorGraph or batched TensorGraph providing empty or None SMILES (valid). """

    mol_graph = tensorize_smiles("")
    assert mol_graph.num_nodes() == 0
    assert mol_graph.num_edges() == 0
    assert mol_graph.batch_indices is None

    mol_graph = tensorize_smiles(None)
    assert mol_graph.num_nodes() == 0
    assert mol_graph.num_edges() == 0
    assert mol_graph.batch_indices is None

    batched_mol_graph = tensorize_smiles_list(
        ["", "", "Brc1ccc2c(c1)[C@H]([NH2+]Cc1nncn1C1CC1)CCC2", "", None])
    assert batched_mol_graph.num_nodes() == 21
    assert batched_mol_graph.num_edges() == 24 * 2
    assert batched_mol_graph.batch_size() == 1
    batch_indices, batch_counts, batch_offsets = batched_mol_graph.batch_locations()
    assert batch_indices.tolist() == [2]
    assert batch_counts.tolist() == [21]
    assert batch_offsets.tolist() == [0]

    batched_mol_graph = tensorize_smiles_list(
        [None, "Cc1ccc([C@@H](C)N(C)C(=O)c2nnn[n-]2)s1", "CN(C)c1ccc(/N=C2/C(O)=C(c3ccccc3)c3cccc[n+]32)cc1", None, ""])
    assert batched_mol_graph.num_nodes() == 17 + 26
    assert batched_mol_graph.num_edges() == (18 + 29) * 2
    assert batched_mol_graph.batch_size() == 2
    batch_indices, batch_counts, batch_offsets = batched_mol_graph.batch_locations()
    assert batch_indices.tolist() == [1, 2]
    assert batch_counts.tolist() == [17, 26]
    assert batch_offsets.tolist() == [0, 17]
