from __future__ import annotations
from common import *
from rdkit import Chem
import networkx as nx
import pandas as pd
from typing import *
from collections import Counter
import logging
from time import time

RDKIT_ATOM_PROPS = ['AtomMapNum', 'AtomicNum', 'ChiralTag', 'FormalCharge', 'Hybridization', 'IsAromatic', 'Isotope',
                    'MonomerInfo', 'NoImplicit', 'NumExplicitHs', 'NumRadicalElectrons', 'PDBResidueInfo']

RDKIT_BOND_PROPS = ['BondDir', 'BondType', 'IsAromatic', 'IsConjugated', 'Stereo']


def get_atom_props_as_dict(atom: Chem.Atom) -> Dict[str, Any]:
    props = {}
    for prop_name in RDKIT_ATOM_PROPS:
        prop_val = getattr(atom, f"Get{prop_name}")()
        if prop_val is not None:
            props[prop_name] = prop_val
    return props


def set_atom_props_from_dict(atom: Chem.Atom, props: Dict[str, Any]):
    for prop_name, prop_val in props.items():
        assert prop_name in RDKIT_ATOM_PROPS
        if prop_val is not None:
            getattr(atom, f"Set{prop_name}")(prop_val)


def get_bond_props_as_dict(bond: Chem.Bond) -> Dict[str, Any]:
    props = {}
    for prop_name in RDKIT_BOND_PROPS:
        prop_val = getattr(bond, f"Get{prop_name}")()
        if prop_val is not None:
            props[prop_name] = prop_val
    return props


def set_bond_props_from_dict(bond: Chem.Bond, props: Dict[str, Any]):
    for prop_name, prop_val in props.items():
        assert prop_name in RDKIT_BOND_PROPS
        if prop_val is not None:
            getattr(bond, f"Set{prop_name}")(prop_val)


def sanitize_smiles(smiles: str):
    smiles = Chem.MolToSmiles(Chem.MolFromSmiles(smiles))  # Dekekulize; took from HierVAE code
    return smiles


class MolGraph:
    smiles: str

    _mol: Optional[Chem.Mol]
    _nx_graph: Optional[nx.Graph]

    def __init__(self, smiles: str):  # Private
        if Chem.MolFromSmiles(smiles, sanitize=True) is None:
            raise Exception(f"Invalid SMILES string: \"{smiles}\"")
        self.smiles = sanitize_smiles(smiles)

        self._mol = None
        self._nx_graph = None

    @property
    def mol(self):
        if self._mol is None:
            self._mol = Chem.MolFromSmiles(self.smiles)
        # TODO Sanitize molecule, precompute properties
        return self._mol

    @property
    def nx_(self) -> nx.Graph:
        if self._nx_graph is None:
            g = nx.Graph()
            for atom in self.mol.GetAtoms():
                g.add_node(atom.GetIdx(), **get_atom_props_as_dict(atom))

            for bond in self.mol.GetBonds():
                g.add_edge(bond.GetBeginAtomIdx(), bond.GetEndAtomIdx(), **get_bond_props_as_dict(bond))

            self._nx_graph = g
        return self._nx_graph

    @staticmethod
    def from_nx(g: nx.Graph) -> MolGraph:
        """ Given a networkx graph that not always is said to be chemically valid (e.g. candidate motif graph).
         Produce a MolGraph object with a valid SMILES string. """

        edit_mol = Chem.RWMol()

        nx_to_atom_idx = {}

        for u in sorted(g.nodes):
            node = g.nodes[u]
            atom = Chem.Atom(node['AtomicNum'])
            atom.SetFormalCharge(node['FormalCharge'])
            atom_idx = edit_mol.AddAtom(atom)
            nx_to_atom_idx[u] = atom_idx

        for u, v in g.edges:
            edge = g.edges[u, v]
            edit_mol.AddBond(nx_to_atom_idx[u], nx_to_atom_idx[v], edge['BondType'])

        # Molecule sanitization:
        # https://www.rdkit.org/docs/RDKit_Book.html#molecular-sanitization

        mol = edit_mol.GetMol()

        # HierVAE code reference:
        # https://github.com/wengong-jin/hgraph2graph/blob/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767/polymers/poly_hgraph/chemutils.py#L68
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol, kekuleSmiles=True))  # Copied, don't ask
        if mol is None:
            raise Exception('Invalid input nx molecular graph')
        Chem.Kekulize(mol, clearAromaticFlags=True)
        mol = Chem.MolFromSmiles(Chem.MolToSmiles(mol))
        return MolGraph(Chem.MolToSmiles(mol))

    @staticmethod
    def from_smiles(smiles: str) -> MolGraph:
        return MolGraph(smiles)


def extract_motif_candidates(mol_graph: MolGraph) -> List[MolGraph]:
    """ Extracts motif candidates from the input molecular graph, according to the technique described in:
     "Hierarchical Generation of Molecular Graphs using Structural Motifs", Appendix A
     """
    mol = mol_graph.mol
    candidates_graph = nx.Graph()

    for bond in mol.GetBonds():
        # HierVAE code reference:
        # https://github.com/wengong-jin/hgraph2graph/blob/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767/polymers/poly_hgraph/chemutils.py#L44

        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()

        remove_bond = False
        if bond.IsInRing():
            remove_bond = False
        elif u.IsInRing() and v.IsInRing():
            remove_bond = True  # Bridge bond, remove it
        elif u.IsInRing() and v.GetDegree() > 1:
            remove_bond = True  # Bridge bond, remove it
        elif v.IsInRing() and u.GetDegree() > 1:
            remove_bond = True  # Bridge bond, remove it

        if not remove_bond:
            candidates_graph.add_node(u.GetIdx(), **get_atom_props_as_dict(u))
            candidates_graph.add_node(v.GetIdx(), **get_atom_props_as_dict(v))
            candidates_graph.add_edge(u.GetIdx(), v.GetIdx(), **get_bond_props_as_dict(bond))

    candidates = []
    for connected_nodes in nx.connected_components(candidates_graph):
        candidates += [MolGraph.from_nx(mol_graph.nx_.subgraph(connected_nodes))]
    return candidates


def decompose_to_bonds_and_rings(mol_graph: MolGraph):
    mol = mol_graph.mol

    rings_graph = nx.Graph()
    parts: List[MolGraph] = []

    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()
        u_props = get_atom_props_as_dict(u)
        v_props = get_atom_props_as_dict(v)
        if bond.IsInRing():
            # Append the ring as a Motif
            bond_props = get_bond_props_as_dict(bond)
            rings_graph.add_node(u.GetIdx(), **u_props)
            rings_graph.add_node(v.GetIdx(), **v_props)
            rings_graph.add_edge(u.GetIdx(), v.GetIdx(), **bond_props)
        else:
            # Append the standalone bond as a Motif
            bond_props = get_bond_props_as_dict(bond)
            bond_graph = nx.Graph()
            bond_graph.add_node(u.GetIdx(), **u_props)
            bond_graph.add_node(v.GetIdx(), **v_props)
            bond_graph.add_edge(u.GetIdx(), v.GetIdx(), **bond_props)
            parts.append(MolGraph.from_nx(bond_graph))

    num_bonds = len(parts)

    rings = list(nx.connected_components(rings_graph))
    num_rings = len(rings)
    for connected_nodes in rings:
        ring = MolGraph.from_nx(rings_graph.subgraph(connected_nodes))
        parts.append(ring)

    assert len(parts) == (num_rings + num_bonds)
    return parts, (num_rings, num_bonds)


def construct_motif_graph(mol_graph: MolGraph, motifs: Dict[str, int]):
    """ Given a molecular graph and the motif vocabulary, construct the motif graph.
    The motif graph is a graph where nodes are motifs: a clustering of the molecule atoms. """

    # Annotate mol_graph edges with motif indices
    mol_g = mol_graph.nx_
    candidates = extract_motif_candidates(mol_graph)
    for candidate in candidates:
        if candidate.smiles in motifs:
            # The candidate was elected to a motif (frequent enough)
            motif_idx = motifs[candidate.smiles]
            for u, v in candidate.nx_.edges:
                mol_g.edges[u, v]['motif'] = motif_idx
        else:
            # Not a motif, its parts are for sure in the motif list
            parts, _ = decompose_to_bonds_and_rings(candidate)
            for part in parts:
                assert part.smiles in motifs
                motif_idx = motifs[part.smiles]
                for u, v in part.nx_.edges:
                    mol_g.edges[u, v]['motif'] = motif_idx

    # After having annotated edges with motif indices, we can construct the motif graph
    motif_g = nx.Graph()
    for u, v, motif in mol_g.edges(data='motif'):
        motif_g.add_node(motif)

        for _, w, other_motif in mol_g.edges(u, data='motif'):
            if motif != other_motif:
                motif_g.add_edge(u, w, motif=other_motif)

        for _, w, other_motif in mol_g.edges(v, data='motif'):
            if motif != other_motif:
                motif_g.add_edge(u, w, motif=other_motif)
    return motif_g


def generate_motif_vocabulary(training_set: pd.DataFrame, min_frequency=100) -> List[Tuple[int, str, bool]]:
    candidate_counter = Counter()
    motifs: Dict[str, int] = {}  # Motif -> Motif index
    vocab_rows: List[Tuple[int, str, bool]] = []  # Motif vocabulary (id, smiles, passed_frequency_test)

    def add_motif(smiles_: str, passed_frequency_test: bool):
        if smiles_ not in motifs:
            motif_idx = len(motifs)
            motifs[smiles_] = motif_idx
            vocab_rows.append((motif_idx, smiles_, passed_frequency_test))

    started_at = time()
    logged_at = time()
    for i, smiles in enumerate(training_set['smiles']):
        mol_graph = MolGraph.from_smiles(smiles)
        candidates = extract_motif_candidates(mol_graph)
        for candidate in candidates:
            candidate_counter[candidate.smiles] += 1
        if time() - logged_at > 1.0:
            time_left = (len(training_set) - (i + 1)) / ((i + 1) / (time() - started_at))
            logging.info(f"{i}/{len(training_set)} training set SMILES processed; "
                         f"Candidate motifs: {len(candidate_counter)}, "
                         f"Time left: {time_left:.1f}s")
            logged_at = time()
        if i > 5000:
            break

    # If the candidate isn't frequent enough in training set, split it into bonds and rings
    num_candidates = len(candidate_counter)
    for i, (smiles, count) in enumerate(candidate_counter.items()):
        if count < min_frequency:
            candidate = MolGraph.from_smiles(smiles)
            parts, (num_rings, num_bonds) = decompose_to_bonds_and_rings(candidate)
            if num_rings > 0:
                logging.debug(f"Candidate motif decomposed; "
                              f"SMILES: \"{candidate.smiles}\", "
                              f"Rings: {num_rings}, "
                              f"Bonds: {num_bonds}")
            for part in parts:
                add_motif(part.smiles, False)
        else:
            add_motif(smiles, True)
        if i % 100 == 0:
            logging.info(f"{i}/{num_candidates} candidate motifs processed; Output motifs: {len(motifs)}")

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])  # Only for logging
    num_motifs = len(vocab_rows)
    num_passed_freq_test = len(df[df['passed_frequency_test']])
    logging.info(f"Motif vocabulary built; "
                 f"Size: {num_motifs}; "
                 f"Passed frequency test: {num_passed_freq_test}/{num_motifs}")
    return vocab_rows


def generate_and_save_motif_vocabulary():
    training_set = pd.read_csv(DATASET_PATH)
    vocab_rows = generate_motif_vocabulary(training_set, min_frequency=100)

    df = pd.DataFrame(vocab_rows, columns=['id', 'smiles', 'passed_frequency_test'])
    df.set_index('id', inplace=True)
    df.to_csv(MOTIF_VOCAB_CSV)


def _main():
    generate_and_save_motif_vocabulary()


if __name__ == "__main__":
    _main()
