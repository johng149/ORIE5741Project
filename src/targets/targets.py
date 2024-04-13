import torch
from torch import Tensor
from torch.nested import nested_tensor, to_padded_tensor
from torch import roll
from typing import List


def create_targets(raw: List[Tensor]) -> Tensor:
    """
    Given list of vectors describing sequence, where -1
    is sequence position we do not need to make predictions
    for, create targets tensor to be used for loss function

    @param raw: List[Tensor] - list of vectors describing sequence
    @return Tensor - tensor of targets

    For the targets to work correctly, raw must have a -1 as the
    first element for each sequence.

    For example, if raw is:
    [
        [-1,-1,2,-1,-1,3],
        [-1,8,-1,5]
    ]

    result is:
    [
        [-1,2,-1,-1,3,-1]
        [8,-1,5,-1,-1,-1]
    ]
    """
    for i, t in enumerate(raw):
        raw[i] = roll(t, -1)
        raw[i][-1] = -1
    return to_padded_tensor(nested_tensor(raw), -1)
