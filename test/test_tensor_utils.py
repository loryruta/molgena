from utils.tensor_utils import *


def test_exclusive_prefix_sum():
    x = torch.tensor([5, 1, 2, 5, 6])
    y = exclusive_prefix_sum(x)
    assert (y == torch.tensor([0, 5, 6, 8, 13])).all()

    x = torch.tensor([1, 1, 1, 1, 1, 1, 1])
    y = exclusive_prefix_sum(x)
    assert (y == torch.tensor([0, 1, 2, 3, 4, 5, 6])).all()
