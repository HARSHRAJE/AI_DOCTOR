import torch
import torch.nn as nn

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

from tqdm import tqdm

from dataloader import test_loader
from model import create_model


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 3

MODEL_PATH = "models/best_robust_model.pth"

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Create model
# ============================================================

model = create_model(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# Load trained model
# ============================================================

checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

model.load_state_dict(
    checkpoint["model_state_dict"]
)

print()
print("Best checkpoint loaded")
print(
    f"Checkpoint epoch: "
    f"{checkpoint['epoch']}"
)

print(
    f"Validation accuracy: "
    f"{checkpoint['val_accuracy']:.4f}"
)


# ============================================================
# Evaluation mode
# ============================================================

model.eval()


# ============================================================
# Predictions
# ============================================================

all_labels = []
all_predictions = []


test_progress = tqdm(
    test_loader,
    desc="Testing",
    unit="batch"
)


with torch.no_grad():

    for images, labels in test_progress:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Forward pass
        outputs = model(images)

        # Predicted class
        _, predictions = torch.max(
            outputs,
            1
        )

        # Store results
        all_labels.extend(
            labels.cpu().numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )


# ============================================================
# Accuracy
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)


# ============================================================
# Results
# ============================================================

print()
print("=" * 70)
print("TEST RESULTS")
print("=" * 70)

print(
    f"Test Accuracy: {accuracy:.4f}"
)

print(
    f"Test Accuracy: {accuracy * 100:.2f}%"
)


# ============================================================
# Classification Report
# ============================================================

print()
print("=" * 70)
print("CLASSIFICATION REPORT")
print("=" * 70)

print(
    classification_report(
        all_labels,
        all_predictions,
        target_names=CLASS_NAMES,
        digits=4
    )
)


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    all_labels,
    all_predictions
)

print()
print("=" * 70)
print("CONFUSION MATRIX")
print("=" * 70)

print()

print("             Predicted")

print(
    f"{'':15}"
    f"{'COVID':>12}"
    f"{'NORMAL':>12}"
    f"{'PNEUMONIA':>12}"
)

for i, class_name in enumerate(CLASS_NAMES):

    print(
        f"{class_name:<15}"
        f"{cm[i][0]:>12}"
        f"{cm[i][1]:>12}"
        f"{cm[i][2]:>12}"
    )


print()
print("Evaluation completed.")