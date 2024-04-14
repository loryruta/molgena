import torch
from torch import nn


def exclusive_prefix_sum(x):
    cum_sum = torch.cumsum(x, dim=0)
    return torch.cat([torch.tensor([0]), cum_sum[:-1]])


def intersect(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    # Source: https://stackoverflow.com/a/65516153/7358682
    combined = torch.cat((a.view(-1), b.view(-1)))
    unique, counts = combined.unique(return_counts=True)
    return unique[counts > 1].reshape(-1, a.shape[1])


def cross_entropy(p: torch.Tensor, y: torch.Tensor, dim: int = 0):
    return -torch.sum(y * torch.log(p), dim=dim)


def num_model_params(model: nn.Module) -> int:
    """ Returns the number of trainable parameters. """
    return sum(param.numel() for param in model.parameters())


def iou(a: torch.Tensor, b: torch.Tensor, dim: int = 0):
    """ Calculates intersection Over Union (Jaccard index) of the two input tensors. """
    assert a.shape == b.shape
    assert a.dtype == b.dtype == torch.bool
    return (a & b).sum(dim=dim) / (a | b).sum(dim=dim)
