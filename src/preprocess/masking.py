import torch
from torch import Tensor
from torch import ones, tril
from typing import List


def create_masks(seq_lens: List[int], longest_seq_len: int) -> Tensor:
    """
    Given list of sequence lengths and the longest sequence length,
    create a mask for each sequence where it is False for padding /
    future tokens. It is assumed that multihead attention is not
    being used

    @param seq_lens: List of sequence lengths
    @param longest_seq_len: Length of longest sequence
    @return: Tensor of shape (batch_size, longest_seq_len, longest_seq_len)
    """
    attn_matrices = []
    for i in range(len(seq_lens)):
        seq = seq_lens[i]
        mask = tril(ones(longest_seq_len, longest_seq_len), diagonal=0)
        mask[:, seq:] = 0
        attn_matrices.append(mask.bool())
    return torch.stack(attn_matrices)


def multiheadify_masks(masks: Tensor) -> Tensor:
    """
    Given the output of create_masks, modifies the masks such that
    it is compatible with multihead attention. It is assumed
    that each head uses the same mask pattern (which is reasonable
    since padding / future tokens are still the same regardless
    of which head is being considered)

    @param masks: Tensor of shape (batch_size, longest_seq_len, longest_seq_len)
    @return: Tensor of shape (batch_size, 1, longest_seq_len, longest_seq_len)
    """
    return masks.unsqueeze(1)


def multiheadify_with_num_heads(masks: Tensor, num_heads: int) -> Tensor:
    """
    Given the output of create_masks, modifies the masks such that
    it is compatible with multihead attention from
    `nn.MultiheadAttention`. This differs from `multiheadify_masks`
    because `multiheadify_masks` changes the number of dimensions to
    4, while this function keeps the number of dimensions the same
    but repeats the mask as many times as there are heads

    @param masks: Tensor of shape (batch_size, longest_seq_len, longest_seq_len)
    @param num_heads: Number of heads
    @return: Tensor of shape (batch_size * num_heads, longest_seq_len, longest_seq_len)
    """
    return masks.repeat(num_heads, 1, 1)
