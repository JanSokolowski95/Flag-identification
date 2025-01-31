import argparse
import pathlib


def args() -> argparse.Namespace:
    """Parse the command line arguments.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="commands")
    download(subparsers)
    train(subparsers)
    predict(subparsers)
    pipeline(subparsers)

    return parser.parse_args()


def download(subparsers) -> None:
    """Add the download subparser.

    Args:
        subparsers:
            The subparsers to which the download
            subparser will be added.
    """
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
    """Add the train subparser.

    Args:
        subparsers:
            The subparsers to which the train
            subparser will be added.
    """
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
    """Add the predict subparser.

    Args:
        subparsers:
            The subparsers to which the predict
            subparser will be added.
    """
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
    """Add the pipeline subparser.

    Args:
        subparsers:
            The subparsers to which the pipeline
            subparser will be added.
    """
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
