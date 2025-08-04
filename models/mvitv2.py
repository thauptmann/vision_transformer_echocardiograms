from lightning import LightningModule
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts
import torch


class LitVideoVisionTransformer(LightningModule):
    def __init__(
        self,
        lr=1e-3,
        embedding_size=256,
        resolution=112,
        t=5,
        h=14,
        w=14,
        frames=100,
    ):
        super().__init__()
        self.lr = lr
        self.model = VideoVisionTransformer(embedding_size, resolution, t, h, w, frames)
        self.n = embedding_size

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.model(x).squeeze()

        loss = F.mse_loss(y_hat, y)
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        x_sub, y = batch
        subvideo_y = self.model(x_sub)
        y_hat = torch.mean(subvideo_y)

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


class VideoVisionTransformer(torch.nn.Module):
    def __init__(self, embedding_size, resolution, t, h, w, frames):
        super().__init__()
        self.n = 1e4
        self.sequence_length = int((frames / t) * (resolution / h) * (resolution / w))
        self.embedding_size = embedding_size
        self.patch_to_linear = torch.nn.Conv2d(
            kernel_size=(h, w), in_channels=t, out_channels=embedding_size
        )
        self.register_buffer("positional_encoding", self._compute_positional_encoding())
        self.t = t
        self.h = h
        self.w = w
        encoder_layer = torch.nn.TransformerEncoderLayer(
            d_model=embedding_size, nhead=8, dim_feedforward=4 * embedding_size
        )
        self.transformer_encoder = torch.nn.TransformerEncoder(
            encoder_layer, num_layers=8
        )
        self.regression_layer = torch.nn.Linear(embedding_size, 1)
        self.cls_token = torch.nn.Parameter(torch.randn(1, 1, embedding_size))

    def forward(self, x):
        x_tubelet_patches = (
            x.unfold(2, self.t, self.t)
            .unfold(3, self.h, self.h)
            .unfold(4, self.w, self.w)
        )
        x_tubelet_patches = x_tubelet_patches.contiguous().view(
            -1, self.t, self.h, self.w
        )
        x_tubelet_embeddings = self.patch_to_linear(x_tubelet_patches)
        x_tubelet_embeddings = x_tubelet_embeddings.view(
            -1, self.sequence_length, self.embedding_size
        )
        x_tubelet_embeddings += self.positional_encoding
        cls_token_batch = self.cls_token.expand(x.shape[0], -1, -1)
        x_tubelet_embeddings = torch.concat(
            (cls_token_batch, x_tubelet_embeddings), dim=1
        )
        embedding = self.transformer_encoder(x_tubelet_embeddings)[:, 0]
        return self.regression_layer(embedding)

    def _compute_positional_encoding(self):
        positional_encoding = torch.zeros((self.sequence_length, self.embedding_size))
        for k in range(self.sequence_length):
            for i in torch.arange(int(self.embedding_size / 2)):
                denominator = torch.pow(self.n, 2 * i / self.embedding_size)
                if k % 2 == 0:
                    positional_encoding[k, 2 * i] = torch.sin(k / denominator)
                else:
                    positional_encoding[k, 2 * i + 1] = torch.cos(k / denominator)
        return positional_encoding
