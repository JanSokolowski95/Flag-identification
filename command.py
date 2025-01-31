import argparse

import lightning as L

from scrapper import Scrapper
from lightning.pytorch.callbacks import ModelCheckpoint
from flags_data_module import FlagsDataModule
from model import Classifier
import torch
import glob
from PIL import Image
import pathlib
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import v2
from scrapper import preprocess


def download(args: argparse.Namespace):
    """Download images from the given URL.

    Args:
        args:
            Arguments passed to the command.
            See `parser.py:download` for more details
            about the arguments.

    """
    scrapper = Scrapper()
    scrapper.scrape(args.path, args.url)


def train(args: argparse.Namespace):
    """Train the model.

    Args:
        args:
            Arguments passed to the command.
            See `parser.py:train` for more details
            about the arguments.

    """
    data_module = FlagsDataModule(
        path=args.path,
        transform=v2.Compose(
            [
                v2.ColorJitter(0.2, 0.2, 0.2),
                v2.RandomApply(
                    [
                        v2.RandomRotation(degrees=(-5, 5)),
                        v2.GaussianBlur(kernel_size=(7, 7)),
                        v2.ElasticTransform(alpha=20.0),
                    ],
                    p=0.3,
                ),
            ]
        ),
    )

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
    """Predict the country of the flag.

    Args:
        args:
            Arguments passed to the command.
            See `parser.py:predict` for more details
            about the arguments.

    """
    img_labels = {
        idx: flag for idx, flag in enumerate(pathlib.Path(args.data_path).iterdir())
    }

    checkpoints = args.model_path.glob("*.ckpt")

    last_checkpoint = max(
        [f for f in checkpoints], key=lambda item: item.stat().st_ctime
    )

    model = Classifier.load_from_checkpoint(last_checkpoint)

    img = Image.open(args.path)
    new_img = preprocess(img)

    tensor = pil_to_tensor(new_img).to(device="cuda").float()
    tensor = tensor[None, :, :, :]

    print("--------------------------------------------")
    print(tensor.size())
    print("--------------------------------------------")

    output = model.model(tensor)
    label = torch.argmax(output, dim=-1).flatten().item()
    country = str(img_labels[label]).split("\\")[-1]
    print(f"This is the flag of {country}")


def pipeline(args: argparse.Namespace):
    """Run the whole pipeline.

    Args:
        args:
            Arguments passed to the command.
            See `parser.py:pipeline` for more details
            about the arguments.

    """
    download(args)
    train(args)
    # predict(args)
