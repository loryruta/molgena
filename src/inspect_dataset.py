# This script goes through the dataset (training set, validation set, test set) and gather information that is used
# later to normalize the data.

from common import *
from mol_dataset import ZincDataset
from rdkit import Chem


def _main():
    mol_dataset = ZincDataset.all()

    atomic_number_list = []
    explicit_valence_list = []
    formal_charge_list = []
    isotope_list = []
    mass_list = []

    bond_type_list = []

    for i, mol_smiles in mol_dataset:
        mol = Chem.MolFromSmiles(mol_smiles)

        for atom in mol.GetAtoms():
            atomic_number_list.append(atom.GetAtomicNum())
            explicit_valence_list.append(atom.GetExplicitValence())
            formal_charge_list.append(atom.GetFormalCharge())
            isotope_list.append(atom.GetIsotope())
            mass_list.append(atom.GetMass())

        for bond_type in mol.GetBonds():
            bond_type_list.append(bond_type.GetBondTypeAsDouble())

        if (i + 1) % 10000 == 0:
            logging.debug(f"Processed {i}/{len(mol_dataset)} SMILES...")

    # Atom stats
    t = torch.tensor(atomic_number_list, dtype=torch.float)
    logging.info(f"Atomic number;    "
                 f"Min: {t.min():.5f}, Max: {t.max():.5f}, Mean: {t.mean():.5f}, Std: {t.std():.5f}")

    t = torch.tensor(explicit_valence_list, dtype=torch.float)
    logging.info(f"Explicit valence; "
                 f"Min: {t.min():.5f}, Max: {t.max():.5f}, Mean: {t.mean():.5f}, Std: {t.std():.5f}")

    t = torch.tensor(formal_charge_list, dtype=torch.float)
    logging.info(f"Formal charge;    "
                 f"Min: {t.min():.5f}, Max: {t.max():.5f}, Mean: {t.mean():.5f}, Std: {t.std():.5f}")

    t = torch.tensor(isotope_list, dtype=torch.float)
    logging.info(f"Isotope;          "
                 f"Min: {t.min():.5f}, Max: {t.max():.5f}, Mean: {t.mean():.5f}, Std: {t.std():.5f}")

    t = torch.tensor(mass_list, dtype=torch.float)
    logging.info(f"Mass;             "
                 f"Min: {t.min():.5f}, Max: {t.max():.5f}, Mean: {t.mean():.5f}, Std: {t.std():.5f}")

    # Bond stats
    t = torch.tensor(bond_type_list, dtype=torch.float)
    logging.info(f"Bond type;        "
                 f"Min: {t.min():.5f}, Max: {t.max():.5f}, Mean: {t.mean():.5f}, Std: {t.std():.5f}")


if __name__ == "__main__":
    _main()
