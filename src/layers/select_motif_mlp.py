from common import *
import torch
from torch import nn
from typing import *


# TODO Implement Motif selection with a GNN + path selection?
#   It could potentially reduce the number of parameters.

class SelectMotifMLP(nn.Module):
    """ MLP for Motif selection.
    Given the potential high number of motifs, this network can easily grow in size. """

    def __init__(self, **kwargs):
        super().__init__()

        self._mol_repr_dim = kwargs['mol_repr_dim']
        self._num_motifs = kwargs['num_motifs']

        self._fcn = nn.Sequential(
            nn.Linear(self._mol_repr_dim, 256),
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

    def forward(self, mol_repr):
        motif_distr = self._fcn(mol_repr)  # (B, MD,)
        return torch.argmax(motif_distr, dim=1, keepdim=True)


def _main():
    batch_size = 1024
    mol_repr_dim = 128
    num_motifs = 8522

    mol_repr_batch = torch.randn((batch_size, mol_repr_dim,))

    select_motif_mlp = SelectMotifMLP(mol_repr_dim=mol_repr_dim, num_motifs=num_motifs)

    num_params = sum(param.numel() for param in select_motif_mlp.parameters())
    print(f"Model parameters: {num_params}")

    print(f"Inference test... ", end="")
    selected_motif = select_motif_mlp(mol_repr_batch)
    assert selected_motif.shape == (batch_size, 1,)
    print(f"Done")


if __name__ == "__main__":
    _main()
