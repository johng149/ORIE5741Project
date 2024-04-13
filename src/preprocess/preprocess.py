import torch
from torch import Tensor
from torch.nn.functional import pad
from typing import List, Tuple


def preprocess(raw: List[Tensor]) -> Tuple[Tensor, int, List[int]]:
    """
    Preprocess takes in a list of 2D tensors. Each tensor may have
    different lengths, but the same number of dimensions. We pad
    the tensors to the same shape, as well as returning the
    longest sequence length and the original sequence lengths.

    @param raw: List[Tensor] - list of 2D tensors to preprocess
    @return Tuple[Tensor, int, List[int]] - padded tensor, longest
        sequence length, and original sequence lengths
    """
    seq_lens = [t.size(0) for t in raw]
    longest = max(seq_lens)
    padded = []
    for t in raw:
        padded.append(pad(t, (0, 0, 0, longest - t.size(0))))
    return torch.stack(padded), longest, seq_lens
