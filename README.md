# Chest X-ray Classification

A research prototype that classifies chest X-ray images into **COVID**, **NORMAL**, or **PNEUMONIA** using a fine-tuned DenseNet-121.

The reported internal test accuracy is 97.18%. External generalization is unreliable; this project is for research and educational use only, not for diagnosis or clinical decision-making.

## Requirements

- Python 3.10 or newer
- `pip`
- Internet access the first time the model is created, to download DenseNet-121 ImageNet weights
- A CUDA-capable GPU is optional; the code uses a GPU when available and otherwise runs on CPU

From the project root, create and activate a virtual environment, then install dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install torch torchvision pandas scikit-learn pillow tqdm
```

For a CUDA-enabled PyTorch installation, use the install command recommended by the [official PyTorch selector](https://pytorch.org/get-started/locally/).

## Prepare your dataset

Place your image files locally in this layout:

```text
data/
└── raw/
    └── chest_xray/
        ├── COVID/
        │   └── *.jpg, *.jpeg, or *.png
        ├── NORMAL/
        │   └── *.jpg, *.jpeg, or *.png
        └── PNEUMONIA/
            └── *.jpg, *.jpeg, or *.png
```

Images can also be in subfolders within each class directory. The class folder names must match exactly. Then generate stratified train, validation, and test CSV files (70% / 15% / 15%):

```powershell
python src\split_dataset.py
```

The script writes the CSV files to `data/processed/`. Keep the image files at the same paths when training; the CSVs store image paths relative to the project root.

## Train and evaluate

Train DenseNet-121 using the training and validation splits:

```powershell
python src\train.py
```

The best validation checkpoint is saved locally as `models/best_robust_model.pth`. Evaluate that checkpoint on the test split with:

```powershell
python src\test.py
```

## Predict one image

After training, pass the path to an image:

```powershell
python src\predict.py data\sample\xray.jpg
```

The script prints the predicted class, confidence, and class probabilities. It loads `models/best_robust_model.pth`; this checkpoint is created by training and is not included in the GitHub repository.

## Keep data and model files out of GitHub

Dataset images and trained model checkpoints are not embedded in or pushed with this repository. They can be large and may contain sensitive medical data. The `.gitignore` excludes `data/`, `models/`, `outputs/`, and `.venv/` from Git by default. Keep datasets and checkpoints on your machine, and do not force-add sensitive or licensed data to Git. If collaborators need access, use an authorized, access-controlled storage location and follow the dataset's license and privacy requirements.
