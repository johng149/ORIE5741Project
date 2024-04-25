import torch
from torch import Tensor
from typing import List, Tuple
from src.common.extract import mat_extract
from src.preprocess.emb_process import emb_process
from src.preprocess.targets import create_targets, positional_indices
from src.preprocess.masking import create_masks


def mat_collate_fn_dicts(batch: List[dict]) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given batch from materialized dataset (not lazy loaded), collate
    the input embeddings, positional indices, target indices, and
    attention masks

    @param batch: List of dict containing an "embeddings" and "target" key,
        the dict's "embeddings" key contains the input embeddings as a flat
        list of floats in a "values" key as well as "numRows" and "numCols",
        while the "target" key contains the target indices as a flat list of
        integers in a "values" key
    @return Tuple of input embeddings, positional indices, target indices,
        and attention masks
    """
    embs, targets = mat_extract(batch)
    targets = create_targets(targets)
    pos_indices = positional_indices(targets)
    emb, longest, seq_lens = emb_process(embs)
    masks = create_masks(seq_lens, longest)
    return emb, pos_indices, targets, masks


def mat_collate_fn(
    batch: List[Tuple[Tensor, Tensor]]
) -> Tuple[Tensor, Tensor, Tensor, Tensor]:
    """
    Given batch from materialized dataset (not lazy loaded), collate
    the input embeddings, positional indices, target indices, and
    attention masks

    @param batch: List of Tuple containing the input embeddings and target indices
    @return Tuple of input embeddings, positional indices, target indices,
        and attention masks
    """
    embs, targets = zip(*batch)
    targets = create_targets(targets)
    pos_indices = positional_indices(targets)
    emb, longest, seq_lens = emb_process(embs)
    masks = create_masks(seq_lens, longest)
    return emb, pos_indices, targets, masks
