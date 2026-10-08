from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


# --------------------------------------------------
# Paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "chest_xray"
OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------
# Class mapping
# --------------------------------------------------

CLASS_MAPPING = {
    "COVID": 0,
    "NORMAL": 1,
    "PNEUMONIA": 2
}


# --------------------------------------------------
# Create metadata
# --------------------------------------------------

records = []

for class_name, label in CLASS_MAPPING.items():

    class_dir = DATA_DIR / class_name

    images = [
        path
        for path in class_dir.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    ]

    for image_path in images:

        records.append({
            "image_path": str(image_path.relative_to(PROJECT_ROOT)),
            "class_name": class_name,
            "label": label
        })


df = pd.DataFrame(records)

print(f"Total images: {len(df)}")
print("\nClass distribution:")
print(df["class_name"].value_counts())


# --------------------------------------------------
# Train / temporary split
# 70% train
# 30% temporary
# --------------------------------------------------

train_df, temp_df = train_test_split(
    df,
    test_size=0.30,
    stratify=df["label"],
    random_state=42
)


# --------------------------------------------------
# Validation / test split
# 15% validation
# 15% test
# --------------------------------------------------

val_df, test_df = train_test_split(
    temp_df,
    test_size=0.50,
    stratify=temp_df["label"],
    random_state=42
)


# --------------------------------------------------
# Save
# --------------------------------------------------

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_df = val_df.sample(frac=1, random_state=42).reset_index(drop=True)
test_df = test_df.sample(frac=1, random_state=42).reset_index(drop=True)


train_df.to_csv(OUTPUT_DIR / "train.csv", index=False)
val_df.to_csv(OUTPUT_DIR / "val.csv", index=False)
test_df.to_csv(OUTPUT_DIR / "test.csv", index=False)


# --------------------------------------------------
# Print results
# --------------------------------------------------

print("\nSplit results:")
print(f"Train:      {len(train_df)}")
print(f"Validation: {len(val_df)}")
print(f"Test:       {len(test_df)}")

print("\nTrain distribution:")
print(train_df["class_name"].value_counts())

print("\nValidation distribution:")
print(val_df["class_name"].value_counts())

print("\nTest distribution:")
print(test_df["class_name"].value_counts())

print("\n✅ Dataset split completed.")