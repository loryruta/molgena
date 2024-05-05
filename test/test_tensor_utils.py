from utils.tensor_utils import *


def test_exclusive_prefix_sum():
    x = torch.tensor([5, 1, 2, 5, 6])
    y = exclusive_prefix_sum(x)
    assert (y == torch.tensor([0, 5, 6, 8, 13])).all()

    x = torch.tensor([1, 1, 1, 1, 1, 1, 1])
    y = exclusive_prefix_sum(x)
    assert (y == torch.tensor([0, 1, 2, 3, 4, 5, 6])).all()


def test_deterministic_index_add():
    """ Tests that the result of index_add is reproducible:
    https://pytorch.org/docs/stable/generated/torch.Tensor.index_add_.html#torch.Tensor.index_add_
    """

    index = torch.tensor([4, 2, 2, 2, 3, 0, 1, 4], dtype=torch.long)
    src = torch.rand((8, 10), dtype=torch.float32)

    t1 = torch.zeros((5, 10), dtype=torch.float32)
    t1 = t1.index_add(0, index, src)

    t2 = torch.zeros((5, 10), dtype=torch.float32)
    t2 = t2.index_add(0, index, src)

    assert (t1 == t2).all()
