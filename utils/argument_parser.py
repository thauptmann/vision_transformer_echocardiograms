import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Train a model to predict ejection fraction from echocardiograms."
    )
    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the model architecture that will be trained.",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        help="Number of samples in a batch.",
        required=False,
        default=10,
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        help="Number of frames in a sample.",
        required=False,
        default=100,
    )

    parser.add_argument(
        "-d",
        "--id_device",
        type=int,
        help="ID of the device.",
        required=False,
        default=0,
    )
    args = parser.parse_args()
    return args
