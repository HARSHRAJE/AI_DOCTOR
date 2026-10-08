import torch.nn as nn
from torchvision.models import densenet121, DenseNet121_Weights


def create_model(num_classes=3):

    # Load DenseNet-121 with ImageNet pretrained weights
    model = densenet121(
        weights=DenseNet121_Weights.DEFAULT
    )

    # --------------------------------------------------
    # Full fine-tuning
    # --------------------------------------------------

    # Make ALL DenseNet parameters trainable
    for param in model.parameters():
        param.requires_grad = True

    # --------------------------------------------------
    # Replace ImageNet classifier
    # --------------------------------------------------

    num_features = model.classifier.in_features

    model.classifier = nn.Linear(
        num_features,
        num_classes
    )

    return model