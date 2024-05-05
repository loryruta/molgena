from random import Random
import torch
from rdkit import Chem
import networkx as nx
from motif_vocab import MotifVocab
from mol_graph import *
from tensor_graph import TensorGraph


def create_mgraph_node_feature_vector(motif_id: int) -> torch.Tensor:
    """ Creates a feature vector for a node having the given motif id. """

    feature_vector_dim = 64

    rand = Random(motif_id)
    feature_vector = torch.zeros((feature_vector_dim,), dtype=torch.float32)
    for i in range(rand.randint(10, 50)):
        val = rand.random() * 2. - 1.
        pos = rand.randint(0, feature_vector_dim - 1)
        feature_vector[pos] += val
    return feature_vector


def tensorize_mgraph(mgraph: nx.DiGraph, motif_vocab: MotifVocab) -> TensorGraph:
    tensor_graph = TensorGraph()

    tensor_graph.node_features = torch.zeros((len(mgraph.nodes),))

    node_features = []
    edge_features = []
    edges = []

    # print([node['motif_id'] for _, node in mgraph.nodes(data=True)])

    # Create node features
    i = 0
    for cid in sorted(mgraph.nodes):
        mid = mgraph.nodes[cid]['motif_id']
        node_features.append(create_mgraph_node_feature_vector(mid))
        assert i == cid  # DEBUG: paranoia check
        i += 1

    # Create edge features
    undirected_edges = [(u, v) for u, v in mgraph.edges if u < v]
    for u, v in undirected_edges:
        motif_ai1, motif_ai2, bond_type = mgraph.edges[u, v]['attachment']

        # Sanity check: the edge in the opposite direction is supposed to exist
        assert (v, u) in mgraph.edges
        assert mgraph.edges[v, u]['attachment'] == (motif_ai2, motif_ai1, bond_type)

        motif_smiles1 = motif_vocab.at_id(mgraph.nodes[u]['motif_id'])['smiles']
        motif_smiles2 = motif_vocab.at_id(mgraph.nodes[v]['motif_id'])['smiles']

        motif_a1 = Chem.MolFromSmiles(motif_smiles1).GetAtomWithIdx(motif_ai1)
        motif_a2 = Chem.MolFromSmiles(motif_smiles2).GetAtomWithIdx(motif_ai2)

        u_atom_features = torch.tensor(create_atom_features(motif_a1), dtype=torch.float32)
        v_atom_features = torch.tensor(create_atom_features(motif_a2), dtype=torch.float32)
        bond_features = create_bond_type_features(bond_type)

        edge_features.append(torch.cat([u_atom_features, bond_features, v_atom_features]))  # uv_features
        edges.append([u, v])

        edge_features.append(torch.cat([v_atom_features, bond_features, u_atom_features]))  # vu_features
        edges.append([v, u])

    tensor_graph.node_features = torch.stack(node_features)
    if edge_features:
        tensor_graph.edge_features = torch.stack(edge_features)
        tensor_graph.edges = torch.tensor(edges, dtype=torch.long).t()

    tensor_graph.validate()

    return tensor_graph
