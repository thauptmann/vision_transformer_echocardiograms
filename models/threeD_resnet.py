from torchvision.models.video import mc3_18
from lightning import LightningModule
import torch.nn.functional as F
from torch.nn import Linear, Conv3d
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch


class MC3_18(LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.model = mc3_18()
        self.model.fc = Linear(self.model.fc.in_features, 1)
        old_stem = self.model.stem[0]
        self.model.stem[0] = Conv3d(
            in_channels=1,
            out_channels=old_stem.out_channels,
            kernel_size=old_stem.kernel_size,
            stride=old_stem.stride,
            padding=old_stem.padding,
            dilation=old_stem.dilation,
            groups=old_stem.groups,
            bias=old_stem.bias is not None,
            padding_mode=old_stem.padding_mode,
        )
        self.model = torch.compile(self.model)
        self.lr = lr

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze().float()

        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze().float()

        loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        self.log("val_loss", loss)
        self.log("mae_loss", mae_loss)

    def test_step(self, batch, batch_idx):
        x, y = batch
        if x.shape[2] < 100:
            return
        x_sub = torch.cat(
            [x[:, :, a : a + 100, :, :] for a in range(0, x.shape[2] - 99, 50)], dim=0
        )

        y_hat = torch.mean(self.model(x_sub)).float()

        mse_loss = F.mse_loss(y_hat, y)
        mae_loss = F.l1_loss(y_hat, y)
        self.log("test_mae", mae_loss)
        self.log("test_mse", mse_loss)

    def configure_optimizers(self):
        optimizer = AdamW(self.parameters(), lr=self.lr)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": ReduceLROnPlateau(optimizer=optimizer),
                "monitor": "val_loss",
                "frequency": 5,
            },
        }
