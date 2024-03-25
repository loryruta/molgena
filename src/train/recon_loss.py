import torch
from utils.tensor_utils import intersect, cross_entropy
from typing import *


def eval_attachment_similarity(attachment_a: torch.LongTensor, attachment_b: torch.LongTensor) -> float:
    """ Evaluates the similarity between attachment_a and attachment_b using IoU (or Jaccard index).
    Practical use case: attachment_a is the prediction and attachment_b is the ground truth.
    """
    attachment_a = attachment_a[attachment_a[:, 2] != 0]
    attachment_b = attachment_b[attachment_b[:, 2] != 0]  # TODO Paranoia, the GT shouldn't have NONE bonds (remove)

    intersection = intersect(attachment_a, attachment_b)
    union = torch.cat([attachment_a, attachment_b]).unique()

    num_intersection = intersection.shape[0]
    num_union = union.shape[0]

    assert num_intersection <= num_union
    return num_intersection / num_union


class LossParams:
    pred_motif_distr: torch.FloatTensor
    pred_attachment: Optional[torch.LongTensor]  # Optional because on first reconstruction step there's no attachment
    true_motif_distr: torch.FloatTensor
    true_attachment: Optional[torch.LongTensor]


def eval_(**kwargs: Unpack[LossParams]) -> float:
    params: LossParams = LossParams()
    params.__dict__.update(kwargs)

    assert params.pred_motif_distr.shape == params.true_motif_distr.shape
    assert (params.pred_attachment is None) == (params.true_attachment is None)

    batch_size = params.pred_motif_distr.shape[0]

    c1 = 1.0  # Coefficient for motif selection component
    c2 = 1.0  # Coefficient for attachment component

    l1 = c1 * cross_entropy(params.pred_motif_distr, params.true_motif_distr) / batch_size

    l2 = 0.0
    if params.pred_attachment is not None:
        l2 = c2 * eval_attachment_similarity(params.pred_attachment, params.true_attachment)

    return l1 + l2
