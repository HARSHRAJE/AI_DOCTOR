from pathlib import Path

import pandas as pd
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset
from torchvision import transforms


# ============================================================
# Configuration
# ============================================================

IMAGE_SIZE = 224


# ============================================================
# Aspect-ratio preserving resize + padding
# ============================================================

class ResizeWithPadding:

    def __init__(
        self,
        size=224,
        fill=0
    ):
        self.size = size
        self.fill = fill

    def __call__(self, image):

        # Preserve original aspect ratio
        image.thumbnail(
            (self.size, self.size),
            Image.Resampling.BILINEAR
        )

        # Calculate padding
        width, height = image.size

        pad_left = (self.size - width) // 2
        pad_right = self.size - width - pad_left

        pad_top = (self.size - height) // 2
        pad_bottom = self.size - height - pad_top

        # Add padding
        image = ImageOps.expand(
            image,
            border=(
                pad_left,
                pad_top,
                pad_right,
                pad_bottom
            ),
            fill=self.fill
        )

        return image


# ============================================================
# Dataset
# ============================================================

class ChestXrayDataset(Dataset):

    def __init__(
        self,
        csv_file,
        root_dir,
        transform=None
    ):

        self.data = pd.read_csv(csv_file)

        self.root_dir = Path(
            root_dir
        )

        self.transform = transform

    def __len__(self):

        return len(self.data)

    def __getitem__(self, index):

        row = self.data.iloc[index]

        image_path = (
            self.root_dir
            / row["image_path"]
        )

        # ----------------------------------------------------
        # Load image
        # ----------------------------------------------------

        image = Image.open(
            image_path
        ).convert("RGB")

        label = int(
            row["label"]
        )

        # ----------------------------------------------------
        # Transform
        # ----------------------------------------------------

        if self.transform:

            image = self.transform(
                image
            )

        return image, label


# ============================================================
# ImageNet normalization
# ============================================================

IMAGENET_MEAN = [
    0.485,
    0.456,
    0.406
]

IMAGENET_STD = [
    0.229,
    0.224,
    0.225
]


# ============================================================
# Training transformations
# ============================================================

train_transform = transforms.Compose([

    # Keep anatomical proportions
    ResizeWithPadding(
        IMAGE_SIZE
    ),

    # --------------------------------------------------------
    # Conservative augmentation
    # --------------------------------------------------------

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ColorJitter(
        brightness=0.12,
        contrast=0.12
    ),

    # --------------------------------------------------------
    # Convert to tensor
    # --------------------------------------------------------

    transforms.ToTensor(),

    # --------------------------------------------------------
    # ImageNet normalization
    # --------------------------------------------------------

    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])


# ============================================================
# Validation / Test / Inference
# ============================================================

val_test_transform = transforms.Compose([

    # Preserve aspect ratio
    ResizeWithPadding(
        IMAGE_SIZE
    ),

    # No random augmentation
    transforms.ToTensor(),

    # Same normalization
    transforms.Normalize(
        mean=IMAGENET_MEAN,
        std=IMAGENET_STD
    )
])