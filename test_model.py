import argparse

import lightning as L
from data_loader import EchoDataModule
from pathlib import Path

from models.swin3d import LitVideoSwinTransformer
from models.threeD_resnet import LitMC3_18
from models.video_vision_transformer import LitVideoVisionTransformer


def main(checkpoint_path, datamodule, model):

    trainer = L.Trainer(
        devices=1,
    )
    trainer.test(ckpt_path=checkpoint_path, datamodule=datamodule, model=model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Test a trained model to predict ejection fraction from echocardiograms."
    )
    parser.add_argument(
        "-c",
        "--checkpoint_path",
        type=str,
        help="Path to the trained model",
        default="",
    )

    parser.add_argument(
        "-m",
        "--model",
        type=str,
        help="Name of the model architecture that will be trained.",
    )
    parser.add_argument(
        "-f",
        "--frames",
        type=int,
        help="Number of frames in a sample.",
        required=False,
        default=100,
    )

    args = parser.parse_args()
    echonet_dynamic = EchoDataModule(data_dir="data", n_frames=args.frames)

    if args.model == "mc3_18":
        model = LitMC3_18()
    elif args.model == "vivit":
        model = LitVideoVisionTransformer(frames=args.frames)
    elif args.model == "video_swin":
        model = LitVideoSwinTransformer()

    if Path(args.checkpoint_path).exists():
        main(args.checkpoint_path, echonet_dynamic, model)
    else:
        print("No valid checkpoint path given.")
