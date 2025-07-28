import os
import random
import torch
import pandas as pd

from torch.utils.data import Dataset
from torchcodec.decoders import VideoDecoder
from lightning import LightningDataModule
from torchvision.tv_tensors import Video
from torchvision.transforms import v2
from torch.utils.data import DataLoader


class EchoDataModule(LightningDataModule):
    def __init__(self, data_dir: str = "", batch_size: int = 10):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.transforms = v2.RandomApply(
            [
                v2.RandomAffine(degrees=25, translate=(0.05, 0.05), scale=(0.9, 1.0)),
                v2.ColorJitter(),
                v2.GaussianBlur(5),
                v2.GaussianNoise(),
                v2.RandomAdjustSharpness(0),
            ]
        )

    def setup(self, stage):
        if stage == "fit":
            self.train_dataset = EchoDataset(
                data_dir=self.data_dir,
                split="TRAIN",
                transform=self.transforms,
            )
            self.validation_dataset = EchoDataset(
                data_dir=self.data_dir,
                split="VAL",
            )
        if stage == "test":
            self.test_dataset = EchoDataset(
                data_dir=self.data_dir,
                split="TEST",
                use_full_video=True,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=4,
            drop_last=True,
            shuffle=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.validation_dataset, batch_size=self.batch_size * 2, num_workers=4
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=1, num_workers=4)


class EchoDataset(Dataset):
    def __init__(
        self,
        data_dir,
        split="TRAIN",
        transform=None,
        use_full_video=False,
        n_frames=100,
        resolution=112,
    ):
        super().__init__()
        self.video_directory = data_dir + "/Videos"
        all_labels = pd.read_csv(data_dir + "/FileList.csv")
        self.video_labels = all_labels[all_labels["Split"] == split].reset_index()
        self.transform = transform
        self.use_full_video = use_full_video
        self.n_frames = n_frames
        self.resolution = resolution

    def __len__(self):
        return len(self.video_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(
            self.video_directory,
            self.video_labels.at[idx, "FileName"] + ".avi",
        )
        video = VideoDecoder(img_path)[:][:, 0:1, :, :]
        video_frame_number = (
            video.shape[0] if video.shape[0] < self.n_frames else self.n_frames
        )

        if self.use_full_video:
            padded_video = video
        else:
            padded_video = torch.zeros(
                [self.n_frames, 1, self.resolution, self.resolution]
            )
            if self.transform:
                n_frames = video.shape[0]
                if n_frames > self.n_frames:
                    start = random.randint(0, n_frames - self.n_frames)
                else:
                    start = 0
            else:
                start = 0

            padded_video[:video_frame_number, :, :, :] += video[
                start : start + video_frame_number, :, :, :
            ]

        if self.transform:
            padded_video = self.transform(Video(padded_video))

        padded_video = torch.permute(padded_video, [1, 0, 2, 3])
        label = self.video_labels.at[idx, "EF"].astype("float32")

        return padded_video.float() / 255, label
