from typing import List, Tuple
from torch import Tensor
import torch

def produce_embedding(raw: dict) -> Tensor:
    """
    Given a dict containing the input embeddings as a flat list of floats
    in a "values" key as well as "numRows" and "numCols", produce the
    corresponding embedding tensor.

    @param raw: Dict containing the input embeddings as a flat list of floats
        in a "values" key as well as "numRows" and "numCols"
    @return Tensor representing the input embeddings
    """
    return torch.tensor(raw["values"]).reshape(raw["numRows"], raw["numCols"])

def mat_extract(batch: List[dict]) -> Tuple[List[Tensor], List[Tensor]]:
    """
    Given batch from materialized dataset (not lazy loaded),
    extract the input embeddings and positional indices.

    @param batch: List of dict containing an "embeddings" and "target" key,
        the dict's "embeddings" key contains the input embeddings as a flat
        list of floats in a "values" key as well as "numRows" and "numCols",
        while the "target" key contains the target indices as a flat list of
        integers in a "values" key
    @return Tuple of List of input embeddings and List of positional indices,
        these are stored in lists since their sequence lengths may differ
    """
    embs = [produce_embedding(d["embeddings"]) for d in batch]
    targets = [torch.tensor(d["target"]["values"]) for d in batch]
    return embs, targets