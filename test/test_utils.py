import torch


# TODO change name (test_utils seems that we're testing utils while these are utils for tests)


def assert_normalized_output(input_: torch.Tensor, dim: int = 0):
    """ Checks that the input is normalized at the given dim; i.e. it must sum to 1. """
    assert (torch.abs(torch.sum(input_, dim=dim) - 1.0) <= 1e-3).all()


def assert_01_tensor(input_: torch.Tensor):
    """ Checks that input values are [0, 1] (e.g. validates the output of a sigmoid function). """
    assert torch.min(input_).item() >= 0.0
    assert torch.max(input_).item() <= 1.0
