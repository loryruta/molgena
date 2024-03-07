from __future__ import annotations
from common import *
from rdkit import Chem
import networkx as nx
import pandas as pd
from typing import *
from collections import Counter
import logging
from time import time
from utils.chem_utils import *

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


def get_bond_props_as_dict(bond: Chem.Bond) -> Dict[str, Any]:
    props = {}
    for prop_name in RDKIT_BOND_PROPS:
        prop_val = getattr(bond, f"Get{prop_name}")()
        if prop_val is not None:
            props[prop_name] = prop_val
    return props


class MolGraph:
    smiles: str

    _mol: Optional[Chem.Mol]
    _nx_graph: Optional[nx.Graph]

    def __init__(self, smiles: str):  # Private
        if Chem.MolFromSmiles(smiles, sanitize=True) is None:
            raise Exception(f"Invalid SMILES string: \"{smiles}\"")
        self.smiles = smiles

        self._mol = None
        self._nx_graph = None

    def get_mol(self):
        if self._mol is None:
            self._mol = Chem.MolFromSmiles(self.smiles)
            assert self._mol is not None
            Chem.Kekulize(self._mol)
        return self._mol

    @property  # TODO delete
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
    def from_smiles(smiles: str) -> MolGraph:
        return MolGraph(smiles)


def extract_motif_candidates(mol_graph: MolGraph) -> List[MolGraph]:
    """ Extracts motif candidates from the input molecular graph, according to the technique described in:
     "Hierarchical Generation of Molecular Graphs using Structural Motifs", Appendix A
     """

    # HierVAE code reference:
    # https://github.com/wengong-jin/hgraph2graph/blob/e396dbaf43f9d4ac2ee2568a4d5f93ad9e78b767/polymers/poly_hgraph/chemutils.py#L44

    # We perform the splitting of the input molecular graph using RWMol.
    # This class permits to remove atomic bonds while maintaining chemical validity (we can't use networkx for that!)

    mol = mol_graph.get_mol()
    new_mol = Chem.RWMol(mol)  # The molecule on which we work for splitting

    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()

        if bond.IsInRing():
            continue

        if u.IsInRing() and v.IsInRing():
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif u.IsInRing() and v.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(u))
            new_mol.AddBond(new_idx, v.GetIdx(), bond.GetBondType())
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())
        elif v.IsInRing() and u.GetDegree() > 1:
            new_idx = new_mol.AddAtom(copy_atom(v))
            new_mol.AddBond(u.GetIdx(), new_idx, bond.GetBondType())
            new_mol.RemoveBond(u.GetIdx(), v.GetIdx())

    candidates = []
    for atom_indices in Chem.GetMolFrags(new_mol):
        frag_mol = extract_mol_fragment(new_mol, atom_indices)
        candidates += [MolGraph.from_smiles(Chem.MolToSmiles(frag_mol))]
    return candidates


def decompose_to_bonds_and_rings(mol_graph: MolGraph) -> Tuple[Set[str], Tuple[int, int]]:
    mol = mol_graph.get_mol()

    bonds = set()
    rings_mol = Chem.RWMol(mol)

    # Extract all bonds that are not part of any ring
    for bond in mol.GetBonds():
        u = bond.GetBeginAtom()
        v = bond.GetEndAtom()
        if not bond.IsInRing():
            rings_mol.RemoveBond(u.GetIdx(), v.GetIdx())

            frag_mol = extract_mol_fragment(mol, {u.GetIdx(), v.GetIdx()})
            bonds.add(Chem.MolToSmiles(frag_mol))

    # Remove atoms that were left disconnected after bond removal
    rings_mol.BeginBatchEdit()
    for atom in mol.GetAtoms():
        if atom.GetDegree() == 0:
            rings_mol.RemoveAtom(atom.GetIdx())
    rings_mol.CommitBatchEdit()

    # Now only rings should be remained, create a fragment for each of them
    rings = set()
    for atom_indices in Chem.GetSymmSSSR(rings_mol):
        ring_mol = extract_mol_fragment(rings_mol, atom_indices)
        rings.add(Chem.MolToSmiles(ring_mol))

    # TODO check that the input molecule was fully decomposed (i.e. is extracting rings and bonds "enough")?

    return rings.union(bonds), (len(rings), len(bonds))


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

    # If the candidate isn't frequent enough in training set, split it into bonds and rings

    num_candidates = len(candidate_counter)
    for i, (frag_smiles, count) in enumerate(candidate_counter.items()):
        if count < min_frequency:
            candidate = MolGraph.from_smiles(frag_smiles)
            parts_smiles, _ = decompose_to_bonds_and_rings(candidate)
            for part_smiles in parts_smiles:
                add_motif(part_smiles, False)
        else:
            add_motif(frag_smiles, True)
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
