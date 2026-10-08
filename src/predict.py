import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image

from model import create_model
from dataset import val_test_transform


# ============================================================
# Configuration
# ============================================================

NUM_CLASSES = 3

CLASS_NAMES = [
    "COVID",
    "NORMAL",
    "PNEUMONIA"
]

PROJECT_ROOT = Path(
    __file__
).resolve().parent.parent

MODEL_PATH = (
    PROJECT_ROOT
    / "models"
    / "best_robust_model.pth"
)


# ============================================================
# Device
# ============================================================

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# Load model
# ============================================================

def load_model():

    print(
        f"Loading model from: {MODEL_PATH}"
    )

    model = create_model(
        num_classes=NUM_CLASSES
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(DEVICE)

    model.eval()

    print("Model loaded successfully.")

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Checkpoint epoch: "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Validation accuracy: "
        f"{checkpoint['val_accuracy']:.4f}"
    )

    return model


# ============================================================
# Predict image
# ============================================================

def predict_image(
    image_path,
    model
):

    image_path = Path(image_path)

    if not image_path.exists():

        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = Image.open(
        image_path
    ).convert("RGB")

    # --------------------------------------------------------
    # IMPORTANT:
    # Use EXACTLY the same transform used
    # for validation and testing.
    # --------------------------------------------------------

    image_tensor = val_test_transform(
        image
    )

    # Add batch dimension
    image_tensor = image_tensor.unsqueeze(0)

    image_tensor = image_tensor.to(
        DEVICE
    )

    # --------------------------------------------------------
    # Inference
    # --------------------------------------------------------

    with torch.no_grad():

        outputs = model(
            image_tensor
        )

        probabilities = F.softmax(
            outputs,
            dim=1
        )[0]

    # --------------------------------------------------------
    # Prediction
    # --------------------------------------------------------

    predicted_index = torch.argmax(
        probabilities
    ).item()

    predicted_class = CLASS_NAMES[
        predicted_index
    ]

    confidence = probabilities[
        predicted_index
    ].item()

    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    probability_dict = {}

    for index, class_name in enumerate(
        CLASS_NAMES
    ):

        probability_dict[class_name] = (
            probabilities[index].item()
        )

    return {
        "prediction": predicted_class,
        "confidence": confidence,
        "probabilities": probability_dict
    }


# ============================================================
# Main
# ============================================================

if __name__ == "__main__":

    if len(sys.argv) != 2:

        print()
        print(
            "Usage:"
        )

        print(
            "python src\\predict.py "
            "<image_path>"
        )

        print()

        print(
            "Example:"
        )

        print(
            "python src\\predict.py "
            "data\\sample\\xray.jpg"
        )

        sys.exit(1)

    image_path = sys.argv[1]

    print()
    print("=" * 70)
    print("AI DOCTOR - X-RAY PREDICTION")
    print("=" * 70)

    print(
        f"Device: {DEVICE}"
    )

    print(
        f"Model: {MODEL_PATH}"
    )

    print(
        f"Image: {image_path}"
    )

    print(
        "Transform: val_test_transform"
    )

    print("=" * 70)

    try:

        model = load_model()

        result = predict_image(
            image_path,
            model
        )

        print()
        print(
            "=" * 70
        )

        print(
            f"Prediction: "
            f"{result['prediction']}"
        )

        print(
            f"Confidence: "
            f"{result['confidence'] * 100:.2f}%"
        )

        print()
        print("Probabilities:")

        for (
            class_name,
            probability
        ) in result["probabilities"].items():

            print(
                f"  {class_name:<12}"
                f"{probability * 100:.2f}%"
            )

        print(
            "=" * 70
        )

    except Exception as e:

        print()
        print(
            f"ERROR: {e}"
        )

        sys.exit(1)