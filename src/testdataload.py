import torch

from dataloader import train_loader


images, labels = next(iter(train_loader))

print("Images shape:")
print(images.shape)

print("\nLabels shape:")
print(labels.shape)

print("\nLabels:")
print(labels)

print("\nImage datatype:")
print(images.dtype)

print("\nLabel datatype:")
print(labels.dtype)