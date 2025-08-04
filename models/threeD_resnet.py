from torchvision.models.video import mc3_18
from lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch


class LitMC3_18(LightningModule):
    def __init__(self, lr=1e-4):
        super().__init__()
        self.lr = lr
        self.model = mc3_18()
        self.model.fc = Linear(self.model.fc.in_features, 1)
        old_input_layer = self.model.stem[0]
        self.model.stem[0] = torch.nn.Conv3d(
            in_channels=1,
            out_channels=old_input_layer.out_channels,
            kernel_size=old_input_layer.kernel_size,
            stride=old_input_layer.stride,
            padding=old_input_layer.padding,
            dilation=old_input_layer.dilation,
            groups=old_input_layer.groups,
            bias=old_input_layer.bias,
            padding_mode=old_input_layer.padding_mode,
        )
        self.save_hyperparameters()

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()

        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_sub, y = batch
        y_hat = torch.mean(self.model(x_sub))
        loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", loss)
        self.log("mae_loss", mae_loss)

    def test_step(self, batch, batch_idx):
        x_sub, y = batch
        y_hat = torch.mean(self.model(x_sub))
        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        self.log("test_mae", mae_loss)
        self.log("test_mse", mse_loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": CosineAnnealingWarmRestarts(
                    optimizer=optimizer, T_0=5, T_mult=2
                ),
                "monitor": "val_loss",
                "frequency": 1,
                "interval": "epoch",
            },
        }
