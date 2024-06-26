import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from src.preprocess.masking import multiheadify_with_num_heads

"""
Adds positional encoding to the input embeddings using sin/cosine functions

Dimensions:
    Input: (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, d_model)
"""


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=5000):
        super(PositionalEncoding, self).__init__()
        pe_matrix = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe_matrix)

    def forward(self, x):
        seq_len = x.size(1)
        x += self.pe[:seq_len, :]
        return x


"""
Applies masking to input embeddings

Dimensions:
    Input: (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, seq_len)
"""

# class Mask(nn.Module):
#     def __init__(self):
#         super(Mask, self).__init__()

#     def forward(self, x, mask):
#         return x.masked_fill(mask == 0, float('-inf'))


"""
Transformer encoder with multi-head self-attention and layer normalization

Dimensions:
    Input: (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, seq_len)

"""


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(
            d_model, nhead, dropout=dropout, batch_first=True
        )
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=mask, need_weights=False)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src


"""
Stacks transformer encoder layers
"""


class TransformerEncoder(nn.Module):
    def __init__(
        self,
        num_layers,
        d_model,
        nhead,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=5000,
    ):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [
                TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
                for _ in range(num_layers)
            ]
        )

        # self.dropout = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        for layer in self.layers:
            src = layer(src, mask)
        return src


"""
Integrates embedding layer and encoders
"""


class TransformerModel(nn.Module):
    def __init__(
        self,
        num_classes,
        nhead,
        num_layers,
        dim_feedforward=2048,
        dropout=0.1,
        max_length=5000,
        emb_dim=384,
    ):
        super(TransformerModel, self).__init__()
        self.kwargs = {
            "num_classes": num_classes,
            "nhead": nhead,
            "num_layers": num_layers,
            "dim_feedforward": dim_feedforward,
            "dropout": dropout,
            "max_length": max_length,
            "emb_dim": emb_dim,
        }
        self.nhead = nhead
        self.pos_emb = nn.Embedding(2, emb_dim)
        self.encoder = TransformerEncoder(
            num_layers, emb_dim, nhead, dim_feedforward, dropout, max_length
        )
        self.linear = nn.Linear(emb_dim, num_classes)

    def forward(self, emb, pos_indices, mask=None):
        """

        Args:
            emb (Tensor): Input shape batch_size x seq_len x emb_dim
            pos_indices (Tensor): Input shape batch_size x seq_len, binary tensor
            mask (Tensor, optional): Input shape batch_size x seq_len x seq_len. Defaults to None.

        Returns:
            _type_: _description_
        """
        batch_size, seq_len, emb_dim = emb.shape
        expected_mask_batch = batch_size * self.nhead
        if mask is not None and mask.shape[0] != expected_mask_batch:
            mask = multiheadify_with_num_heads(mask, self.nhead)
        pos = self.pos_emb(pos_indices)
        emb = emb + pos
        output = self.encoder(emb, mask)
        output = self.linear(output)
        return output
