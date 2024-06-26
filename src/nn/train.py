import torch
from torch import Tensor
from typing import Union, Optional
from src.preprocess.collate_fn import mat_collate_fn
from src.dataset.dataset import SparkifyDataset
from torch import nn
from torch.nn import functional as F
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import os


def save_checkpoint(
    model,
    optm,
    epoch,
    model_kwargs,
    train_data,
    test_data,
    writer,
    checkpoint_path,
    ignore_index,
    cross_entropy_weights=None,
):
    save_data = {
        "model": model.state_dict(),
        "optimizer": optm.state_dict(),
        "epoch": epoch,
        "model_kwargs": model.kwargs,
        "train_kwargs": train_data.kwargs,
        "test_kwargs": test_data.kwargs if test_data else None,
        "writer_dir": writer.get_logdir(),
        "checkpoint_path": checkpoint_path,
        "ignore_index": ignore_index,
        "cross_entropy_weights": cross_entropy_weights,
    }
    torch.save(save_data, os.path.join(checkpoint_path, "checkpoint.pth"))


def accuracy(pred: Tensor, targets: Tensor, ignore_idx: int = -1) -> float:
    """
    Compute the accuracy of the model's predictions.

    @param pred: Predicted labels, shape batch_size x seq_len
    @param targets: True labels, shape batch_size x seq_len,
        if a target is equal to ignore_idx, it is ignored
    @param ignore_idx: Index to ignore in the targets
    @return: Accuracy of the model's predictions
    """
    with torch.no_grad():
        acc = (pred == targets).sum().item() / (targets != ignore_idx).sum().item()
    return acc


def train(
    train_data: SparkifyDataset,
    test_data: Optional[SparkifyDataset],
    model: nn.Module,
    target_epochs: int,
    seen_epochs: int,
    batch_size: int,
    test_batch_size: int,
    optm: optim.Optimizer,
    writer: Union[SummaryWriter, str],
    checkpoint_path: str,
    ignore_index: int = -1,
    test_every_n: int = 100,
    save_every_n: int = 100,
    device: str = "cpu",
    checkpoint_name: str = "checkpoint.pth",
    cross_entropy_weights: Optional[Tensor] = None,
):
    """
    Train the model on the training data.

    @param train_data: Training data
    @param test_data: Test data
    @param model: Model to train
    @param epochs: Number of epochs to train for
    @param seen_epochs: Number of epochs model has already been trained for
    @param batch_size: Batch size for training
    @param test_batch_size: Batch size for testing
    @param optm: Optimizer to use for training
    @param writer: SummaryWriter for logging, if a string is passed, it is assumed to be a path
        and will be used to create a SummaryWriter
    @param checkpoint_path: Path to save model checkpoints
    @param ignore_index: Index to ignore in the targets
    @param test_every_n: Test the model every n iterations
    @param save_every_n: Save the model every n iterations
    @param device: Device to use for training
    @param checkpoint_name: Name of the checkpoint file
    @param cross_entropy_weights: Weights for the cross entropy loss, useful for
        imbalanced datasets
    """
    if cross_entropy_weights is not None:
        cross_entropy_weights = cross_entropy_weights.to(device)
    os.makedirs(checkpoint_path, exist_ok=True)
    if isinstance(writer, str):
        os.makedirs(writer, exist_ok=True)
    else:
        os.makedirs(writer.get_logdir(), exist_ok=True)
    train_dl = DataLoader(
        train_data, batch_size=batch_size, shuffle=True, collate_fn=mat_collate_fn
    )
    test_dl = (
        DataLoader(
            test_data,
            batch_size=test_batch_size,
            shuffle=False,
            collate_fn=mat_collate_fn,
        )
        if test_data
        else None
    )
    if isinstance(writer, str):
        writer = SummaryWriter(writer)
    train_iter = iter(train_dl)
    test_iter = iter(test_dl) if test_dl else None
    try:
        for epoch in tqdm(range(seen_epochs, target_epochs)):
            model.train()
            try:
                emb, pos_indices, targets, masks = next(train_iter)
            except StopIteration:
                train_iter = iter(train_dl)
                emb, pos_indices, targets, masks = next(train_iter)
            batch_size, seq_len, _ = emb.shape
            emb, pos_indices, targets, masks = (
                emb.to(device),
                pos_indices.to(device),
                targets.to(device),
                masks.to(device),
            )
            outputs = model(emb, pos_indices, mask=masks)
            with torch.no_grad():
                acc = accuracy(outputs.argmax(-1), targets)
                writer.add_scalar("Accuracy/train", acc, epoch)
            loss = F.cross_entropy(
                outputs.reshape(-1, outputs.size(-1)),
                targets.reshape(-1),
                ignore_index=ignore_index,
                weight=cross_entropy_weights,
            )
            optm.zero_grad()
            loss.backward()
            optm.step()
            writer.add_scalar("Loss/train", loss.item() / batch_size, epoch)
            if epoch % test_every_n == 0 and test_data:
                with torch.no_grad():
                    model.eval()
                    try:
                        emb, pos_indices, targets, masks = next(test_iter)
                    except StopIteration:
                        test_iter = iter(test_dl)
                        emb, pos_indices, targets, masks = next(test_iter)
                    batch_size, seq_len, _ = emb.shape
                    emb, pos_indices, targets, masks = (
                        emb.to(device),
                        pos_indices.to(device),
                        targets.to(device),
                        masks.to(device),
                    )
                    outputs = model(emb, pos_indices, mask=masks)
                    acc = accuracy(outputs.argmax(-1), targets)
                    writer.add_scalar("Accuracy/test", acc, epoch)
                    loss = F.cross_entropy(
                        outputs.reshape(-1, outputs.size(-1)),
                        targets.reshape(-1),
                        ignore_index=ignore_index,
                        weight=cross_entropy_weights,
                    )
                    writer.add_scalar("Loss/test", loss.item() / batch_size, epoch)
            if epoch % save_every_n == 0:
                save_checkpoint(
                    model,
                    optm,
                    epoch,
                    model.kwargs,
                    train_data,
                    test_data,
                    writer,
                    checkpoint_path,
                    ignore_index,
                    cross_entropy_weights,
                )
        save_checkpoint(
            model,
            optm,
            epoch,
            model.kwargs,
            train_data,
            test_data,
            writer,
            checkpoint_path,
            ignore_index,
            cross_entropy_weights,
        )
    except KeyboardInterrupt:
        save_checkpoint(
            model,
            optm,
            epoch,
            model.kwargs,
            train_data,
            test_data,
            writer,
            checkpoint_path,
            ignore_index,
            cross_entropy_weights,
        )


def load_checkpoint(
    checkpoint_path: str,
    model_class: nn.Module,
    optm_class: optim.Optimizer,
    checkpoint_file: str = "checkpoint.pth",
    device: str = "cpu",
):
    """
    Load a model checkpoint.

    @param checkpoint_path: Path to the checkpoint
    @param model_class: Model class to load
    @param optm_class: Optimizer class to load
    @return: Model, optimizer, epoch, train data, test data (if available), writer directory,
        checkpoint path, ignore index, and cross entropy weights
    """
    checkpoint = torch.load(os.path.join(checkpoint_path, checkpoint_file))
    model = model_class(**checkpoint["model_kwargs"])
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    optm = optm_class(model.parameters())
    optm.load_state_dict(checkpoint["optimizer"])
    train_ds = SparkifyDataset(**checkpoint["train_kwargs"])
    test_ds = (
        SparkifyDataset(**checkpoint["test_kwargs"])
        if checkpoint["test_kwargs"]
        else None
    )
    writer = SummaryWriter(checkpoint["writer_dir"])
    return (
        model,
        optm,
        checkpoint["epoch"],
        train_ds,
        test_ds,
        writer,
        checkpoint["checkpoint_path"],
        checkpoint["ignore_index"],
        checkpoint["cross_entropy_weights"],
    )
