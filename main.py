# from scrapper import Scrapper
# from augmentator import Augmentator
# from data_splitter import DataSplitter
# from flags_dataset import FlagsDataset

import sys

from torch.utils.data import random_split
import torchvision
import os
import pathlib
import torch
from torchvision import transforms
import lightning as L
import argparse

import parser
import command


def main():
    """Main function of the program."""
    L.seed_everything(seed=42)
    args = parser.args()
    sys.exit(getattr(command, args.commands)(args))
    return 0


if __name__ == "__main__":
    main()
