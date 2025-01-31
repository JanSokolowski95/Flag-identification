import argparse

import lightning as L

from scrapper import Scrapper
from augmentator import Augmentator
from lightning.pytorch.callbacks import ModelCheckpoint
from flags_data_module import FlagsDataModule
from model import Classifier
import torch
import glob
from PIL import Image
import pathlib
from torchvision.transforms.functional import pil_to_tensor


def download(args: argparse.Namespace):
    scrapper = Scrapper()
    scrapper.scrape(args.path, args.url)


def train(args: argparse.Namespace):
    data_module = FlagsDataModule(path=args.path, transform=None)
    trainer = L.Trainer(
        check_val_every_n_epoch=1,
        max_epochs=100,
        precision="16-mixed",
        benchmark=True,
        callbacks=[
            ModelCheckpoint(dirpath=str(args.model_path)),
        ],
    )
    model = Classifier()
    trainer.fit(model, data_module)


def predict(args: argparse.Namespace):
    img_labels = {
        idx: flag for idx, flag in enumerate(pathlib.Path(args.data_path).iterdir())
    }

    checkpoints = args.model_path.glob("*.ckpt")

    last_checkpoint = max(
        [f for f in checkpoints], key=lambda item: item.stat().st_ctime
    )

    model = Classifier.load_from_checkpoint(last_checkpoint)

    img = Image.open(args.path)  # torch.randn(1, 3, 224, 224).to(device="cuda")

    tensor = pil_to_tensor(img).to(device="cuda").float()
    tensor = tensor[None, :, :, :]

    print("--------------------------------------------")
    print(tensor.size())
    print("--------------------------------------------")

    output = model.model(tensor)
    label = torch.argmax(output, dim=-1).flatten().item()
    country = str(img_labels[label]).split("\\")[-1]
    print(f"This is the flag of {country}")


def pipeline(args: argparse.Namespace):
    download(args)
    train(args)
    # predict(args)
