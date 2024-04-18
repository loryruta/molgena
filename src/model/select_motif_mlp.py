from common import *
import torch
from torch import nn
from typing import *
from utils.tensor_utils import *


class SelectMotifMlp(nn.Module):
    """
    A layer used to select the next motif to attach to the input partial molecule.

     Can be used both for:
       - Reconstruction (see training): the input is the molecule to reconstruct and a partial molecule. The motif is
          selected such that the motif attached to the partial molecule is more equal to the molecule to reconstruct.
       - Property optimization: the molecule to reconstruct input is zero. The motif is selected such that the motif
          attached to the partial molecule contributes the improvement of a chemical property.
     """

    mol_repr_dim: int
    num_motifs: int
    reconstruction_mode: bool

    def __init__(self, params: Dict[str, Any]):
        super().__init__()

        self.mol_repr_dim = params['mol_repr_dim']
        self.num_motifs = params['num_motifs']
        self.reconstruction_mode = params['reconstruction_mode']

        self._mlp = create_mlp(
            self.mol_repr_dim + self.mol_repr_dim,
            self.num_motifs,
            params["hidden_layers"],
            nn.LeakyReLU()
        )
        # Output logits must be _unnormalized_ to be fed to F.cross_entropy (pytorch)

    def forward(self, partial_mol_reprs: torch.Tensor, recon_mol_reprs: Optional[torch.Tensor]):
        """
        :param partial_mol_reprs:
            The vector representations for the partial molecules under reconstruction.
            Should be a zero tensor for empty partial molecules (initial reconstruction step).
        :param recon_mol_reprs:
            The vector representations for the molecules to reconstruct. Only required when reconstructing.
        """
        assert (not self.reconstruction_mode) or (recon_mol_reprs is not None)

        if not self.reconstruction_mode:
            batch_size = partial_mol_reprs.shape[0]
            recon_mol_reprs = torch.zeros((batch_size, self.mol_repr_dim), dtype=torch.float32)  # Zero padding

        return self._mlp(torch.cat([partial_mol_reprs, recon_mol_reprs], dim=1))
