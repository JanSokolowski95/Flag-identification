import pathlib
import lightning as L
from flags_dataset import FlagsDataset
import torch
from torch.utils.data import random_split, DataLoader
import os


class FlagsDataModule(L.LightningDataModule):
    """DataModule for Flags dataset.

    This class is responsible for preparing the data for the model
    and "zipping" everything together.

    Attributes:
        path (pathlib.Path):
            Path to the directory where the data will be stored.
        transform (torch.nn.Module):
            Transform to be applied to the data.
            `None` (no transformations) by default.
        kwargs:
            Additional arguments to be passed to DataLoader,
            e.g. batch_size, num_workers, etc.

    """

    def __init__(self, path: str, transform: torch.nn.Module = None, **kwargs) -> None:
        """Initializes FlagsDataModule.

        Args:
            path (str):
                Path to the directory where the data will be stored.
            transform (torch.nn.Module):
                Transform to be applied to the data.
                `None` (no transformations) by default.
            **kwargs:
                Additional arguments to be passed to DataLoader,
                e.g. batch_size, num_workers, etc.
        """
        super().__init__()

        self.path = pathlib.Path(path).resolve()
        self.path.mkdir(parents=True, exist_ok=True, mode=0o770)

        self.kwargs = kwargs
        self.transform = transform

    def setup(self, stage: str):
        """Setups the data for training, validation and testing.

        Args:
            _:
                Stage for which the data should be prepared.
                Unused, API compatibility only.
        """
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
        """DataLoader for training data.

        Returns:
            Training DataLoader.

        """
        return DataLoader(self.flags_train, **self.kwargs)

    def val_dataloader(self):
        """DataLoader for validation data.

        Returns:
            Validation DataLoader.

        """
        return DataLoader(self.flags_val, **self.kwargs)

    def test_dataloader(self):
        """DataLoader for testing data.

        Returns:
            Testing DataLoader.

        """
        return DataLoader(self.flags_test, **self.kwargs)
