import torch
from torch import Tensor
from torch import ones, tril
from typing import List


def create_masks(seq_lens: List[int], longest_seq_len: int) -> Tensor:
    attn_matrices = []
    for i in range(len(seq_lens)):
        seq = seq_lens[i]
        mask = tril(ones(longest_seq_len, longest_seq_len), diagonal=0)
        mask[:, seq:] = 0
        attn_matrices.append(mask.bool())
    return torch.stack(attn_matrices)
