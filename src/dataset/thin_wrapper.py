from torch.utils.data import Dataset
from datasets import Dataset as HuggingFaceDataset
import random
from torch import Tensor
from typing import List
from src.common.extract import mat_extract


class ThinWrapperDataset(Dataset):
    def __init__(self, hf_dataset: HuggingFaceDataset, max_seq_len: int):
        self.hf_dataset = hf_dataset
        self.max_seq_len = max_seq_len

    def __len__(self):
        return len(self.hf_dataset)

    def __getitem__(self, idx):
        item = self.hf_dataset.__getitem__(idx)
        embs, targets = mat_extract([item])
        emb, target = embs[0], targets[0]
        seq_len, emb_dim = emb.shape
        start = random.randint(0, seq_len - self.max_seq_len)
        emb = emb[start : start + self.max_seq_len, :]
        target = target[start : start + self.max_seq_len]
        return emb, target
