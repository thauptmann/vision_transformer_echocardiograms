import lightning as L
from models.threeD_resnet import LitMC3_18
from models.video_vision_transformer import LitVideoVisionTransformer
from lightning import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from data_loader import EchoDataModule
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor

from utils.argument_parser import parse_arguments

seed_everything(5, workers=True)


def main(
    model,
    datamodule,
    id_device,
):

    # saves top-K checkpoints based on "val_loss" metric
    checkpoint_callback = ModelCheckpoint(
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        filename="echonet-dynamic-{epoch:02d}-{val_loss:.2f}",
    )

    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=20)
    lr_monitor = LearningRateMonitor(logging_interval="step")

    trainer = L.Trainer(
        max_epochs=200,
        devices=[id_device],
        precision="16-mixed",
        deterministic=True,
        callbacks=[early_stopping_callback, checkpoint_callback, lr_monitor],
    )

    tuner = Tuner(trainer)
    tuner.lr_find(model, datamodule=datamodule)
    trainer.fit(model=model, datamodule=datamodule)


if __name__ == "__main__":
    args = parse_arguments()

    if args.model == "mc3_18":
        model = LitMC3_18()
    elif args.model == "vivit":
        model = LitVideoVisionTransformer(frames=args.frames)
    elif args.model == "video_swin":
        model = LitVideoSwinTransformer()

    echonet_dynamic = EchoDataModule(
        data_dir="data", batch_size=args.batch_size, n_frames=args.frames
    )
    main(model, echonet_dynamic, args.id_device)
