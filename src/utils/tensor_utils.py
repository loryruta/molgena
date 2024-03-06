import torch
from torch import nn


def exclusive_prefix_sum(x):
    cum_sum = torch.cumsum(x, dim=0)
    return torch.cat([torch.tensor([0]), cum_sum[:-1]])


def tensor_contains_mask(a: torch.Tensor, b: torch.Tensor) -> torch.BoolTensor:
    """ Given two 1-dim tensors, returns a bool tensor of the same shape of `a`.
     The i-th bool is `True` if the i-th element is contained in `b`. """
    assert a.dim() == 1
    assert b.dim() == 1


def num_model_params(model: nn.Module) -> int:
    """ Returns the number of trainable parameters. """
    return sum(param.numel() for param in model.parameters())
