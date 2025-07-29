import lightning as L
from models.threeD_resnet import MC3_18
from lightning import seed_everything
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.tuner import Tuner
from data_loader import EchoDataModule

seed_everything(5, workers=True)


def main(model, datamodule):
    trainer = L.Trainer(
        max_epochs=200,
        devices=1,
        precision="16-mixed",
        deterministic=True,
        callbacks=[EarlyStopping(monitor="val_loss", mode="min", patience=10)],
        fast_dev_run=False,
    )

    tuner = Tuner(trainer)
    lr_finder = tuner.lr_find(model, datamodule=datamodule)
    if lr_finder:
        new_lr = lr_finder.suggestion()
        model.hparams.lr = new_lr

    trainer.fit(model=model, datamodule=datamodule)
    trainer.test(ckpt_path="best", datamodule=datamodule)


if __name__ == "__main__":
    echonet_dynamic = EchoDataModule(data_dir="data")
    model = MC3_18()
    main(model, echonet_dynamic)
