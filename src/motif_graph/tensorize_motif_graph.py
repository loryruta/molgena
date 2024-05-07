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


def tensorize_mgraph(mgraph: nx.DiGraph,
                     motif_vocab: MotifVocab,
                     return_node_mappings: bool = False) -> Union[TensorGraph, Tuple[TensorGraph, Dict[int, int]]]:
    tensor_graph = TensorGraph()

    tensor_graph.node_features = torch.zeros((len(mgraph.nodes),))

    node_features = []
    edge_features = []
    edges = []

    # print([node['motif_id'] for _, node in mgraph.nodes(data=True)])

    # Maps:
    #   cid -> node_idx
    # To which (sequential) node index was mapped the cid
    node_mappings: Dict[int, int] = {}

    # Create node features
    for cid in mgraph.nodes:
        # IMPORTANT: don't assume nodes to be sequential (e.g. 0, 1, 2, 3, ...)!
        # For example: CID(s) in partial molecules aren't, some could be missing (e.g. 8, 4, 1, 5)!

        assert type(cid) is int
        node_idx = len(node_features)
        node_features.append(create_mgraph_node_feature_vector(mgraph.nodes[cid]['motif_id']))

        node_mappings[cid] = node_idx

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
        edges.append([node_mappings[u], node_mappings[v]])

        edge_features.append(torch.cat([v_atom_features, bond_features, u_atom_features]))  # vu_features
        edges.append([node_mappings[v], node_mappings[u]])

    tensor_graph.node_features = torch.stack(node_features)
    if edge_features:
        tensor_graph.edge_features = torch.stack(edge_features)
        tensor_graph.edges = torch.tensor(edges, dtype=torch.long).t()

    tensor_graph.validate()

    if return_node_mappings:
        return tensor_graph, node_mappings
    else:
        return tensor_graph


def tensorize_mgraphs(mgraphs: List[nx.DiGraph], motif_vocab: MotifVocab) -> TensorGraph:
    return batch_tensor_graphs([
        tensorize_mgraph(mgraph, motif_vocab) for mgraph in mgraphs
    ])
