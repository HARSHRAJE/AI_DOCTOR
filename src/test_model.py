import torch

from model import create_model


# Device
device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print("Device:", device)


# Create model
model = create_model(num_classes=3)

model = model.to(device)

print("\nModel created successfully.")

# Test input
x = torch.randn(4, 3, 224, 224).to(device)

# Forward pass
output = model(x)

print("\nInput shape:")
print(x.shape)

print("\nOutput shape:")
print(output.shape)

print("\nOutput:")
print(output)