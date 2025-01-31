import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO, BytesIO
from pathlib import Path
import re
import os
from PIL import Image
from tqdm import tqdm
import pathlib


# def resize_images(resizer: ImageResizer):
#     images = [
#         os.path.join(d, x)
#         for d, dirs, files in os.walk(os.getcwd())
#         for x in files
#         if x.endswith(".jpg")
#     ]

# for img_path in images:
#     resizer.resize_image(img_path)


def preprocess(img):
    fill_color = (179, 0, 255)
    rgb_im = img.convert("RGB")
    img_w, img_h = rgb_im.size
    size = max(img_w, img_h)
    offset = ((size - img_w) // 2, (size - img_h) // 2)
    new_img = Image.new("RGB", (size, size), fill_color)
    new_img.paste(rgb_im, offset)
    return new_img


class Scrapper:
    def _get_images(self, table):
        return ["https:" + e.get("src") for e in table.select('img[src*=".svg.png"]')]

    def _get_countries(self, table):
        countries_df = pd.read_html(StringIO(str(table)))
        return [re.sub("\[.\]", "", country) for country in countries_df[0]["State"]]

    def scrape(
        self,
        path: str | pathlib.Path,
        url="https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags",
    ):
        path = pathlib.Path(path)

        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        table = soup.find("table")
        countries = dict()
        images = self._get_images(table)
        for country, image in tqdm(
            zip(self._get_countries(table), images),
            total=len(images),
        ):
            # Chad and Romania have pretty much the same flag, so I'm making only one label for them
            if country == "Chad":
                country = "either Chad or Romania"
            elif country == "Romania":
                country = "either Chad or Romania"
            current_path = path / country
            if not country in countries:
                countries[country] = 0
            else:
                countries[country] += 1

            current_path.mkdir(parents=True, exist_ok=True, mode=0o770)
            img = Image.open(BytesIO(requests.get(image).content))

            new_img = preprocess(img)
            new_img.save((current_path / str(countries[country])).with_suffix(".jpg"))

            # with (
            #     (current_path / str(countries[country]))
            #     .with_suffix(".jpg")
            #     .open("xb") as f
            # ):
            #     f.write(requests.get(image).content)
