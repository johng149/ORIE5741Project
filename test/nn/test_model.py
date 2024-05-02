from src.nn.model import DummyModel
import torch


def test_dummy_model():
    vocab_size = 2
    emb_dim = 3
    tokens = torch.tensor([0, 1, 0, 1], dtype=torch.long)
    dm = DummyModel(vocab_size, emb_dim)
    output = dm(tokens)
    assert output.shape == (4, emb_dim)


def test_dummy_model_batched():
    vocab_size = 2
    emb_dim = 3
    tokens = torch.tensor([[0, 1, 0, 1], [1, 0, 1, 0]], dtype=torch.long)
    dm = DummyModel(vocab_size, emb_dim)
    output = dm(tokens)
    assert output.shape == (2, 4, emb_dim)
