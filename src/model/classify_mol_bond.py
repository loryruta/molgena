from common import *
import torch
from torch import nn
from typing import *
from model.encode_mol import EncodeMolMPN
from tensor_graph import TensorGraph


class ClassifyMolBond(nn.Module):
    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self._mol_mpn = EncodeMolMPN(params['mpn'])

        self._node_hidden_dim = self._mol_mpn.node_hidden_dim
        self._edge_hidden_dim = self._mol_mpn.edge_hidden_dim

        self._classify_bond_type_mlp = nn.Sequential(
            nn.Linear(
                self._mol_mpn.node_hidden_dim +
                self._mol_mpn.edge_hidden_dim +
                self._mol_mpn.node_hidden_dim,
                128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 4),
            nn.Softmax(dim=1)
        )

    @staticmethod
    def _verify_proposed_bonds(a_batch_indices: torch.LongTensor,
                               b_batch_indices: torch.LongTensor,
                               proposed_bonds: torch.LongTensor):
        assert (a_batch_indices[proposed_bonds[0]] == b_batch_indices[proposed_bonds[1]]).all()

    def forward(self, mol_a_graphs: TensorGraph, mol_b_graphs: TensorGraph, proposed_bonds: torch.LongTensor):
        """
        Given the molecular graphs of two molecules and a list of proposed bonds between the two, classifies every bond
        to be a: NONE, SINGLE, DOUBLE or TRIPLE bond.

        :param mol_a_graphs:
            The batched molecular graphs of the first molecule (e.g. the input molecule).
        :param mol_b_graphs:
            The batched molecular graphs of the second molecule (e.g. the motif).
        :param proposed_bonds:
            A long tensor of shape (2, NC) indicating the bonds between mol_a and mol_b proposed to do.
            Values are inter-batch indices that must connect atoms of the same batch. TODO could be verified
        """

        assert proposed_bonds.ndim == 2 and proposed_bonds.shape[0] == 2

        self._verify_proposed_bonds(mol_a_graphs.batch_indices, mol_b_graphs.batch_indices, proposed_bonds)

        # NC = num proposed bonds

        num_mol_a_atoms = mol_a_graphs.num_nodes()
        num_mol_a_bonds = mol_a_graphs.num_edges()
        num_mol_b_bonds = mol_b_graphs.num_edges()
        num_proposed_bonds = proposed_bonds.shape[1]  # NC

        # Create a graph where mol_a and mol_b are connected using proposed_bond

        # Since we want mol_a and mol_b to lie in the same graph, we need to offset mol_b atom indices
        proposed_bonds_1 = proposed_bonds[1] + num_mol_a_atoms
        additional_bonds = torch.cat([
            torch.stack([  # Straight edges
                proposed_bonds[0],
                proposed_bonds_1
            ]),
            torch.stack([  # Reversed edges
                proposed_bonds_1,
                proposed_bonds[0]
            ])
        ], dim=1)  # (2, N)
        num_additional_bonds = additional_bonds.shape[1]

        # Create the graph on which to run message passing: it's formed by merging the mol_a graph with the mol_b graph
        # and adding the proposed bonds
        merged_graphs: TensorGraph = TensorGraph()
        merged_graphs.node_features = torch.cat([mol_a_graphs.node_features, mol_b_graphs.node_features])
        merged_graphs.edge_features = torch.cat([
            mol_a_graphs.edge_features,
            mol_b_graphs.edge_features,
            # Use a negative value of -1000.0 for the feature, to indicate it's an artificial bond
            torch.full((num_additional_bonds, 1), -1000.0)
        ])
        merged_graphs.edges = torch.cat([
            mol_a_graphs.edges,
            mol_b_graphs.edges + num_mol_a_atoms,
            additional_bonds
        ], dim=1)
        merged_graphs.batch_indices = torch.cat([mol_a_graphs.batch_indices, mol_b_graphs.batch_indices])
        merged_graphs.create_hiddens(self._node_hidden_dim, self._edge_hidden_dim)

        # Run message passing
        self._mol_mpn(merged_graphs)

        proposed_bonds_offset = num_mol_a_bonds + num_mol_b_bonds  # Offset to retrieve proposed bonds from the concat
        proposed_bond_hiddens = \
            merged_graphs.edge_hiddens[proposed_bonds_offset:proposed_bonds_offset + num_proposed_bonds] + \
            merged_graphs.edge_hiddens[proposed_bonds_offset + num_proposed_bonds:]  # Reversed edges

        mlp_input = torch.cat([
            torch.index_select(merged_graphs.node_hiddens, 0, proposed_bonds[0]),
            torch.index_select(merged_graphs.node_hiddens[num_mol_a_atoms:], 0, proposed_bonds[1]),
            proposed_bond_hiddens
        ], dim=1)  # (NC, NH+NH+EH)
        mlp_output = self._classify_bond_type_mlp(mlp_input)  # (NC, 4)

        return mlp_output
