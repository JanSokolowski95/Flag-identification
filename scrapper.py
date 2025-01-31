import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO, BytesIO
import re
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
    """Preprocess the image to make it square and fill the background with a color.

    Args:
        img (Image):
            The image to be preprocessed.

    """
    # Purple is the least common color used in flags, so I'm padding using purple, instead of standard black
    fill_color = (179, 0, 255)
    rgb_im = img.convert("RGB")
    img_w, img_h = rgb_im.size
    size = max(img_w, img_h)
    offset = ((size - img_w) // 2, (size - img_h) // 2)
    new_img = Image.new("RGB", (size, size), fill_color)
    new_img.paste(rgb_im, offset)
    return new_img


class Scrapper:
    """Scrapper class to download images of flags from Wikipedia and preprocess them."""

    def _get_images(self, table) -> list[str]:
        """Get the images from the table.

        Args:
            table:
                The table containing the images.

        Returns:
            list:
                A list of URLs of the images.

        """
        return ["https:" + e.get("src") for e in table.select('img[src*=".svg.png"]')]

    def _get_countries(self, table) -> list[str]:
        """Get the countries from the table.

        Args:
            table:
                The table containing the countries.

        Returns:
            list:
                A list of countries.

        """
        countries_df = pd.read_html(StringIO(str(table)))
        return [re.sub("\[.\]", "", country) for country in countries_df[0]["State"]]

    def scrape(
        self,
        path: str | pathlib.Path,
        url="https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags",
    ) -> None:
        """Scrape the images of flags from Wikipedia and preprocess them.

        Args:
            path (str or Path):
                The path where the images will be saved.
            url (str):
                The URL of the Wikipedia page with the flags.

        """
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
            if country not in countries:
                countries[country] = 0
            else:
                countries[country] += 1

            current_path.mkdir(parents=True, exist_ok=True, mode=0o770)
            img = Image.open(BytesIO(requests.get(image).content))

            new_img = preprocess(img)
            new_img.save((current_path / str(countries[country])).with_suffix(".jpg"))
