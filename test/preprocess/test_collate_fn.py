from src.preprocess.collate_fn import mat_collate_fn
import torch


def test_mat_collate_fn():
    seq_len1 = 3
    seq_len2 = 4
    emb_dim = 5
    emb1 = torch.randn(seq_len1, emb_dim)
    emb2 = torch.randn(seq_len2, emb_dim)
    target1 = torch.tensor([0, 1, 0])
    target2 = torch.tensor([1, 0, 1, 0])
    batch = [(emb1, target1), (emb2, target2)]
    emb, pos_indices, targets, masks = mat_collate_fn(batch)

    longest_seq_len = max(seq_len1, seq_len2)
    assert emb.shape == (2, longest_seq_len, emb_dim)
    assert pos_indices.shape == (2, longest_seq_len)
    assert targets.shape == (2, longest_seq_len)
    assert masks.shape == (2, longest_seq_len, longest_seq_len)
