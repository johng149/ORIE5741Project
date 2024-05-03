from torch.utils.data import Dataset
import torch
import random


class SparkifyDataset(Dataset):
    def __init__(self, data_file, max_seq_len=128, map_device="cpu"):
        self.data_file = data_file
        self.max_seq_len = max_seq_len
        self.map_device = map_device
        self.kwargs = {
            "data_file": data_file,
            "max_seq_len": max_seq_len,
            "map_device": map_device,
        }
        self.data = torch.load(data_file, map_location=map_device)
        self.tensors = self.data["tensors"]
        self.targets = self.data["targets"]

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        tensor = self.tensors[idx]
        target = self.targets[idx]
        seq_len, emb_dim = tensor.shape
        start = random.randint(0, max(0, seq_len - self.max_seq_len))
        emb = tensor[start : start + self.max_seq_len, :]
        target = target[start : start + self.max_seq_len]
        return emb, target
