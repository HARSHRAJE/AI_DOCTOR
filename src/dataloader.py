from pathlib import Path

from torch.utils.data import DataLoader

from dataset import ChestXrayDataset
from dataset import train_transform
from dataset import val_test_transform


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

CSV_DIR = PROCESSED_DIR

# --------------------------------------------------
# Datasets
# --------------------------------------------------

train_dataset = ChestXrayDataset(
    csv_file=CSV_DIR / "train.csv",
    root_dir=PROJECT_ROOT,
    transform=train_transform
)

val_dataset = ChestXrayDataset(
    csv_file=CSV_DIR / "val.csv",
    root_dir=PROJECT_ROOT,
    transform=val_test_transform
)

test_dataset = ChestXrayDataset(
    csv_file=CSV_DIR / "test.csv",
    root_dir=PROJECT_ROOT,
    transform=val_test_transform
)


# --------------------------------------------------
# DataLoaders
# --------------------------------------------------

train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)

test_loader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=0
)


print("Dataset sizes")
print("=" * 30)

print(f"Train:      {len(train_dataset)}")
print(f"Validation: {len(val_dataset)}")
print(f"Test:       {len(test_dataset)}")

print("\nDataLoader batches")

print(f"Train batches:      {len(train_loader)}")
print(f"Validation batches: {len(val_loader)}")
print(f"Test batches:       {len(test_loader)}")