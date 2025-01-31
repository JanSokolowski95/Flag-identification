import argparse
import pathlib


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="commands")
    download(subparsers)
    train(subparsers)
    predict(subparsers)
    pipeline(subparsers)

    return parser.parse_args()


def download(subparsers) -> None:
    # Download data
    download_parser = subparsers.add_parser("download")
    download_parser.add_argument(
        "--path",
        help="Path to save the downloaded data",
        default=pathlib.Path.cwd() / "data",
    )
    download_parser.add_argument(
        "--url",
        help="URL to download the data from",
        default="https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags",
    )


def train(subparsers) -> None:
    # Train model
    train_parser = subparsers.add_parser("train")
    train_parser.add_argument(
        "--path",
        help="Path to the downloaded data",
        default=pathlib.Path.cwd() / "data",
    )
    train_parser.add_argument(
        "--model-path",
        help="Path to save the trained model",
        default=pathlib.Path.cwd() / "model",
    )


def predict(subparsers) -> None:
    # Predict
    predict_parser = subparsers.add_parser("predict")
    predict_parser.add_argument(
        "--path",
        help="Path to the image",
    )
    predict_parser.add_argument(
        "--model-path",
        help="Path to save the trained model",
        default=pathlib.Path.cwd() / "model",
    )
    predict_parser.add_argument(
        "--data-path",
        help="Path to the downloaded data",
        default=pathlib.Path.cwd() / "data",
    )


def pipeline(subparsers) -> None:
    # Whole pipeline
    pipeline_parser = subparsers.add_parser("pipeline")
    pipeline_parser.add_argument(
        "--path",
        help="Path to save the downloaded data",
        default=pathlib.Path.cwd() / "data",
    )
    pipeline_parser.add_argument(
        "--model-path",
        help="Path to save the trained model",
        default=pathlib.Path.cwd() / "model",
    )
    pipeline_parser.add_argument(
        "--url",
        help="URL to download the data from",
        default="https://en.wikipedia.org/wiki/Gallery_of_sovereign_state_flags",
    )
