import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm import tqdm
from pathlib import Path

from dataloader import train_loader, val_loader
from model import create_model


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 3

EPOCHS = 5

# Small learning rate for full fine-tuning
LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", DEVICE)


# ============================================================
# Model directory
# ============================================================

MODEL_DIR = Path("models")

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True
)

BEST_MODEL_PATH = (
    MODEL_DIR / "best_robust_model.pth"
)


# ============================================================
# Create model
# ============================================================

model = create_model(
    num_classes=NUM_CLASSES
)

model = model.to(DEVICE)


# ============================================================
# Parameter information
# ============================================================

total_parameters = sum(
    param.numel()
    for param in model.parameters()
)

trainable_parameters = sum(
    param.numel()
    for param in model.parameters()
    if param.requires_grad
)

print()
print("=" * 70)
print("MODEL INFORMATION")
print("=" * 70)

print(
    f"Total parameters:      "
    f"{total_parameters:,}"
)

print(
    f"Trainable parameters:  "
    f"{trainable_parameters:,}"
)

print("=" * 70)


# ============================================================
# Loss
# ============================================================

criterion = nn.CrossEntropyLoss()


# ============================================================
# Optimizer
# ============================================================

optimizer = AdamW(
    model.parameters(),
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY
)


# ============================================================
# Best validation accuracy
# ============================================================

best_val_accuracy = 0.0


# ============================================================
# Training loop
# ============================================================

for epoch in range(EPOCHS):

    print()
    print("=" * 70)
    print(f"FULL FINE-TUNING — Epoch {epoch + 1}/{EPOCHS}")
    print("=" * 70)

    # --------------------------------------------------------
    # TRAINING
    # --------------------------------------------------------

    model.train()

    running_loss = 0.0

    correct = 0
    total = 0

    batch_count = 0

    progress_bar = tqdm(
        train_loader,
        desc="Training",
        unit="batch"
    )

    for images, labels in progress_bar:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)

        # Loss
        loss = criterion(
            outputs,
            labels
        )

        # Backpropagation
        loss.backward()

        # Update ALL model parameters
        optimizer.step()

        # ----------------------------------------------------
        # Statistics
        # ----------------------------------------------------

        running_loss += loss.item()

        batch_count += 1

        _, predicted = torch.max(
            outputs,
            1
        )

        total += labels.size(0)

        correct += (
            predicted == labels
        ).sum().item()

        current_loss = (
            running_loss / batch_count
        )

        current_accuracy = (
            correct / total
        )

        progress_bar.set_postfix(
            loss=f"{current_loss:.4f}",
            acc=f"{current_accuracy:.4f}"
        )

    train_loss = (
        running_loss / batch_count
    )

    train_accuracy = (
        correct / total
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    val_running_loss = 0.0

    val_correct = 0
    val_total = 0

    val_batch_count = 0

    val_progress = tqdm(
        val_loader,
        desc="Validation",
        unit="batch"
    )

    with torch.no_grad():

        for images, labels in val_progress:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            # Forward pass
            outputs = model(images)

            # Validation loss
            loss = criterion(
                outputs,
                labels
            )

            val_running_loss += loss.item()

            val_batch_count += 1

            # Predictions
            _, predicted = torch.max(
                outputs,
                1
            )

            val_total += labels.size(0)

            val_correct += (
                predicted == labels
            ).sum().item()

            current_val_loss = (
                val_running_loss /
                val_batch_count
            )

            current_val_accuracy = (
                val_correct /
                val_total
            )

            val_progress.set_postfix(
                loss=f"{current_val_loss:.4f}",
                acc=f"{current_val_accuracy:.4f}"
            )


    val_loss = (
        val_running_loss /
        val_batch_count
    )

    val_accuracy = (
        val_correct /
        val_total
    )


    # --------------------------------------------------------
    # Epoch summary
    # --------------------------------------------------------

    print()
    print("-" * 70)

    print(
        f"Train Loss:       {train_loss:.4f}"
    )

    print(
        f"Train Accuracy:   {train_accuracy:.4f}"
    )

    print(
        f"Validation Loss:  {val_loss:.4f}"
    )

    print(
        f"Validation Acc:   {val_accuracy:.4f}"
    )

    print("-" * 70)


    # --------------------------------------------------------
    # Save best model
    # --------------------------------------------------------

    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "epoch": epoch + 1,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "train_loss":
                    train_loss,

                "train_accuracy":
                    train_accuracy,

                "val_loss":
                    val_loss,

                "val_accuracy":
                    val_accuracy,
            },
            BEST_MODEL_PATH
        )

        print(
            f"✅ Best full fine-tuned model saved to:"
            f" {BEST_MODEL_PATH}"
        )

    else:

        print(
            "No validation improvement."
        )

        print(
            f"Best validation accuracy: "
            f"{best_val_accuracy:.4f}"
        )


# ============================================================
# Completed
# ============================================================

print()
print("=" * 70)
print("FULL FINE-TUNING COMPLETED")
print("=" * 70)

print(
    f"Best Validation Accuracy: "
    f"{best_val_accuracy:.4f}"
)

print(
    f"Best model: "
    f"{BEST_MODEL_PATH}"
)