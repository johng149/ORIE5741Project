import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from sentence_transformers import SentenceTransformer


'''
Adds positional encoding to the input embeddings using sin/cosine functions

Dimensions:
    Input: (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, d_model)
'''
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, seq_len=5000):
        super(PositionalEncoding, self).__init__()
        pe_matrix = torch.zeros(seq_len, d_model)
        position = torch.arange(0, seq_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe_matrix[:, 0::2] = torch.sin(position * div_term)
        pe_matrix[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe_matrix)

    def forward(self, x):
        seq_len = x.size(1)
        x += self.pe[:seq_len, :]
        return x
    
'''
Applies masking to input embeddings

Dimensions:
    Input: (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, seq_len)
'''

class Mask(nn.Module):
    def __init__(self):
        super(Mask, self).__init__()

    def forward(self, x, mask):
        return x.masked_fill(mask == 0, float('-inf'))


'''
Transformer encoder with multi-head self-attention and layer normalization

Dimensions:
    Input: (batch_size, seq_len, d_model)
    Output: (batch_size, seq_len, seq_len)

'''

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attn = nn.MultiheadAttention(d_model, nhead, dropout=dropout)
        self.linear1 = nn.Linear(d_model, dim_feedforward)
        self.dropout = nn.Dropout(dropout)
        self.linear2 = nn.Linear(dim_feedforward, d_model)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, src, mask=None):
        src2, _ = self.self_attn(src, src, src, attn_mask=mask)
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(F.relu(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src



'''
Stacks transformer encoder layers
'''

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, nhead, dim_feedforward=2048, dropout=0.1):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout) for _ in range(num_layers)])
        self.pe = PositionalEncoding(d_model)
        self.mask = Mask()

    def forward(self, src, mask=None):
        src = self.pe(src)
        if mask is not None:
            src = self.mask(src, mask)
        for layer in self.layers:
            src = layer(src)
        return src


'''
Integrates embedding layer and encoders
''' 
class TransformerModel(nn.Module):
    def __init__(self, model_name, d_model, nhead, num_layers, dim_feedforward=2048, dropout=0.1):
        super(TransformerModel, self).__init__()
        self.embedding = SentenceTransformer(model_name)
        self.encoder = TransformerEncoder(num_layers, d_model, nhead, dim_feedforward, dropout)

    def forward(self, src, mask=None):
        embedded_src = self.embedding.encode(src)
        output = self.encoder(embedded_src, mask)
        return output
