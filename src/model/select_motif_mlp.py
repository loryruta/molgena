from common import *
import torch
from torch import nn
from typing import *


# TODO Implement Motif selection with a GNN + path selection?
#   It could potentially reduce the number of parameters.

class SelectMotifMlp(nn.Module):
    """
    A layer used to select the next motif to attach to the input partial molecule.

     Can be used both for:
     - Reconstruction (see training): the input is the molecule to reconstruct and a partial molecule. The motif is
        selected such that the motif attached to the partial molecule is more equal to the molecule to reconstruct.
     - Property optimization: the molecule to reconstruct input is zero. The motif is selected such that the motif
        attached to the partial molecule contributes the improvement of a chemical property.
     """

    def __init__(self, **kwargs):
        super().__init__()

        self._mol_repr_dim = kwargs['mol_repr_dim']
        self._num_motifs = kwargs['num_motifs']
        self._reconstruction_mode = kwargs['reconstruction_mode']

        self._zero_padding_vec = torch.zeros((self._mol_repr_dim,), dtype=torch.float)  # Used on property optimization

        self._mlp = nn.Sequential(
            nn.Linear(self._mol_repr_dim + self._mol_repr_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, self._num_motifs),
            nn.Softmax(dim=1)
        )

    def forward(self, partial_mol_reprs: torch.Tensor, recon_mol_reprs: Optional[torch.Tensor] = None):
        """
        :param partial_mol_reprs:
            The vector representations for the partial molecules under reconstruction.
            Can be null for the first step of reconstruction.
        :param recon_mol_reprs:
            The vector representations for the molecules to reconstruct. Only required when reconstructing.
        """
        assert self._reconstruction_mode and recon_mol_reprs is not None
        assert (self.partial_mol_reprs is not None) or (recon_mol_reprs is not None)

        if self._reconstruction_mode is None:
            recon_mol_reprs = self._zero_padding_vec  # Zero padding

        if partial_mol_reprs is None:
            partial_mol_reprs = self._zero_padding_vec  # Zero padding

        return self._mlp(
            torch.cat([partial_mol_reprs, recon_mol_reprs], dim=1)
        )


def _main():  # TODO
    batch_size = 1024
    mol_repr_dim = 128
    num_motifs = 8522

    mol_repr_batch = torch.randn((batch_size, mol_repr_dim,))

    select_motif_mlp = SelectMotifMlp(mol_repr_dim=mol_repr_dim, num_motifs=num_motifs)

    num_params = sum(param.numel() for param in select_motif_mlp.parameters())
    print(f"Model parameters: {num_params}")

    print(f"Inference test... ", end="")
    selected_motif = select_motif_mlp(mol_repr_batch)
    assert selected_motif.shape == (batch_size, 1,)
    print(f"Done")


if __name__ == "__main__":
    _main()
