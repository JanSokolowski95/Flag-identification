import pathlib
import lightning as L
from flags_dataset import FlagsDataset
import torch
from torch.utils.data import random_split, DataLoader
from torchvision import transforms
import os


class FlagsDataModule(L.LightningDataModule):
    def __init__(self, path: str, transform: torch.nn.Module = None, **kwargs):
        super().__init__()

        self.path = pathlib.Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True, mode=0o770)

        self.kwargs = kwargs
        self.transform = transform

    def setup(self, stage: str):
        # Using same transforms as otherwise we would be
        # training, validating and testing on EXACTLY same data
        # Introduces randomness, but hard to avoid if we are basing
        # whole data on one source with one image per example

        self.flags_training, self.flags_test = random_split(
            FlagsDataset(self.path, transform=self.transform),
            [0.9, 0.1],
        )
        self.flags_train, self.flags_val = random_split(
            self.flags_training,
            [0.8, 0.2],
        )

    def train_dataloader(self):
        return DataLoader(self.flags_train, **self.kwargs)

    def val_dataloader(self):
        return DataLoader(self.flags_val, **self.kwargs)

    def test_dataloader(self):
        return DataLoader(self.flags_test, **self.kwargs)
