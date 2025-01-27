import torch
import torch.nn as nn
import torchvision.transforms as transforms
from datasets import Dataset
from PIL import Image
from torch.utils.data import Dataset
from transformers import Trainer

BUFFER_SIZE = 400
BATCH_SIZE = 1
IMG_WIDTH = 256
IMG_HEIGHT = 256
OUTPUT_CHANNELS = 3
LAMBDA = 100


class Pix2PixDataset(Dataset):
    def __init__(self, image_paths, split="train"):
        self.image_paths = image_paths
        self.split = split
        self.transform = self._get_transforms()

    def _get_transforms(self):
        if self.split == "train":
            return transforms.Compose(
                [
                    transforms.Resize((286, 286)),
                    transforms.RandomCrop((256, 256)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )
        else:
            return transforms.Compose(
                [
                    transforms.Resize((256, 256)),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
                ]
            )

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path)
        # Split the image into input and target (assuming they're side by side)
        w = image.size[0] // 2
        input_image = image.crop((w, 0, w * 2, image.size[1]))
        target_image = image.crop((0, 0, w, image.size[1]))

        if self.transform:
            input_image = self.transform(input_image)
            target_image = self.transform(target_image)

        return {"input_images": input_image, "real_images": target_image}


class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        # Encoder
        self.encoder = nn.ModuleList(
            [
                self._downsample(3, 64, normalize=False),  # (b, 64, 128, 128)
                self._downsample(64, 128),  # (b, 128, 64, 64)
                self._downsample(128, 256),  # (b, 256, 32, 32)
                self._downsample(256, 512),  # (b, 512, 16, 16)
                self._downsample(512, 512),  # (b, 512, 8, 8)
                self._downsample(512, 512),  # (b, 512, 4, 4)
                self._downsample(512, 512),  # (b, 512, 2, 2)
                self._downsample(512, 512),  # (b, 512, 1, 1)
            ]
        )

        # Decoder
        self.decoder = nn.ModuleList(
            [
                self._upsample(512, 512, dropout=True),  # (b, 1024, 2, 2)
                self._upsample(1024, 512, dropout=True),  # (b, 1024, 4, 4)
                self._upsample(1024, 512, dropout=True),  # (b, 1024, 8, 8)
                self._upsample(1024, 512),  # (b, 1024, 16, 16)
                self._upsample(1024, 256),  # (b, 512, 32, 32)
                self._upsample(512, 128),  # (b, 256, 64, 64)
                self._upsample(256, 64),  # (b, 128, 128, 128)
            ]
        )

        self.final = nn.Sequential(
            nn.ConvTranspose2d(128, OUTPUT_CHANNELS, 4, 2, 1), nn.Tanh()
        )

    def _downsample(self, in_channels, out_channels, normalize=True):
        layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.LeakyReLU(0.2))
        return nn.Sequential(*layers)

    def _upsample(self, in_channels, out_channels, dropout=False):
        layers = [
            nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
        ]
        if dropout:
            layers.append(nn.Dropout(0.5))
        return nn.Sequential(*layers)

    def forward(self, x):
        # Encoder
        skips = []
        for encoder in self.encoder:
            x = encoder(x)
            skips.append(x)

        # Decoder
        skips = skips[:-1][::-1]  # reverse skip connections
        for decoder, skip in zip(self.decoder, skips):
            x = decoder(x)
            x = torch.cat([x, skip], dim=1)

        x = self.final(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        def discriminator_block(in_channels, out_channels, normalize=True):
            layers = [nn.Conv2d(in_channels, out_channels, 4, 2, 1)]
            if normalize:
                layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.LeakyReLU(0.2))
            return layers

        self.model = nn.Sequential(
            *discriminator_block(OUTPUT_CHANNELS * 2, 64, normalize=False),
            *discriminator_block(64, 128),
            *discriminator_block(128, 256),
            *discriminator_block(256, 512),
            nn.ZeroPad2d(1),
            nn.Conv2d(512, 1, 4, padding=1)
        )

    def forward(self, x, y):
        return self.model(torch.cat([x, y], dim=1))


class Pix2PixTrainer(Trainer):
    def __init__(self, generator, discriminator, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.generator = generator
        self.discriminator = discriminator
        self.gen_criterion = nn.BCEWithLogitsLoss()
        self.l1_criterion = nn.L1Loss()

    def compute_loss(self, model, inputs, return_outputs=False):
        real_A = inputs["input_images"]
        real_B = inputs["real_images"]

        # Generate fake image
        fake_B = self.generator(real_A)

        # Discriminator losses
        fake_AB = torch.cat([real_A, fake_B], 1)
        pred_fake = self.discriminator(fake_B.detach(), real_A)
        loss_D_fake = self.gen_criterion(pred_fake, torch.zeros_like(pred_fake))

        real_AB = torch.cat([real_A, real_B], 1)
        pred_real = self.discriminator(real_B, real_A)
        loss_D_real = self.gen_criterion(pred_real, torch.ones_like(pred_real))

        loss_D = (loss_D_fake + loss_D_real) * 0.5

        # Generator losses
        pred_fake = self.discriminator(fake_B, real_A)
        loss_G_GAN = self.gen_criterion(pred_fake, torch.ones_like(pred_fake))
        loss_G_L1 = self.l1_criterion(fake_B, real_B) * LAMBDA
        loss_G = loss_G_GAN + loss_G_L1

        total_loss = loss_G + loss_D

        if return_outputs:
            return (
                total_loss,
                {
                    "loss_G": loss_G.item(),
                    "loss_D": loss_D.item(),
                    "loss_G_GAN": loss_G_GAN.item(),
                    "loss_G_L1": loss_G_L1.item(),
                },
            )
        return total_loss
