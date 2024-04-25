import torch
from torch import Tensor
from torch.nested import nested_tensor, to_padded_tensor
from torch import roll
from typing import List, Tuple


def create_targets(raw: List[Tensor] | Tuple[Tensor]) -> Tensor:
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
    processed = []
    for i, t in enumerate(raw):
        t = roll(t, -1)
        t[-1] = -1
        processed.append(t)
    return to_padded_tensor(nested_tensor(processed), -1)


def positional_indices(targets: List[Tensor] | Tuple[Tensor]) -> Tensor:
    """
    Given list of vectors describing target vector, where -1
    is sequence position we do not need to make predictions
    for, create positional indices tensor to be used for
    positional embeddings

    @param targets: List[Tensor] - list of vectors describing target vector
    @return Tensor - tensor of positional indices

    It is similar to create_targets, but instead of shifting,
    we just pad and then increment all by 1

    For example, if raw is:
    [
        [-1,-1,2,-1,-1,3],
        [-1,8,-1,5]
    ],

    which produces targets of
    [
        [-1,2,-1,-1,3,-1],
        [8,-1,5,-1,-1,-1]
    ]

    then result is:
    [
        [0,1,0,0,1,0],
        [1,0,1,0,0,0]
    ]
    """
    return (targets >= 0).int()
