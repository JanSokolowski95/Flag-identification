import torch
from torchvision.transforms import v2
from PIL import Image
import os
from tqdm import tqdm


class Augmentator:
    def __init__(self, iterations):
        self.iterations = iterations

    def _augment(self, img_path, save_path, transforms: int = 100):
        # print(path)
        img = Image.open(img_path)
        new_img = transforms(img)
        new_img.save(save_path)

    def augment_data(self, path=os.getcwd()):
        images = [
            os.path.join(d, x)
            for d, dirs, files in os.walk(path)
            for x in files
            if x.endswith(".jpg")
        ]
        files = {
            d.split("\\")[-1]: {"path": d, "count": len(os.listdir(d))}
            for d, dirs, files in os.walk(path)
        }

        new_files = files

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
        for img_path in tqdm(images):
            country_name = img_path.split("\\")[-2].split(".")[0]
            path = files[country_name]["path"]
            for i in range(self.iterations):
                self._augment(
                    img_path,
                    "{path}/{count}.jpg".format(
                        path=path, count=new_files[country_name]["count"]
                    ),
                    transforms,
                )
                new_files[country_name]["count"] += 1
