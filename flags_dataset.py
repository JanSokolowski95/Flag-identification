import os
from torch.utils.data import Dataset
from torchvision.io import read_image
import torch
from torchvision.transforms import v2


transforms = v2.Compose(
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
)


class FlagsDataset(Dataset):
    def __init__(self, img_dir, transform=transforms):
        print(img_dir)
        self.img_labels = [
            (idx, flag / img_name)
            for idx, flag in enumerate(img_dir.iterdir())
            for img_name in (img_dir / flag).iterdir()
        ]
        to_tensor = v2.Compose([v2.ToImage(), v2.ToDtype(torch.float32, scale=True)])
        img_dir = str(img_dir)
        self.images = [
            to_tensor(read_image((os.path.join(img_dir, f"{flag}/{img_name}"))))
            for flag in os.listdir((img_dir))
            for img_name in os.listdir((os.path.join(img_dir, flag)))
        ]

        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        label = self.img_labels[idx][0]
        image = self.images[idx]
        if self.transform:
            image = self.transform(image)
        return image, label
