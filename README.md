# Scene Image Classification with Wide ResNet-50-2

A reproducible, modular version of the scene-classification workflow from the original Kaggle notebook. It uses the Intel Image Classification dataset and a pretrained TorchVision Wide ResNet-50-2 model to classify six scene categories.

## Classes
`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`

## What has been improved
- Correct `requires_grad` handling and an explicit option to freeze the backbone.
- Separate training and validation transforms (no random augmentation during validation).
- Reproducible, stratified train/validation split.
- Automatic device selection: Apple Silicon MPS, NVIDIA CUDA, or CPU.
- Batch-based evaluation with accuracy, macro precision/recall/F1, per-class report, and confusion matrix.
- Best-checkpoint selection using validation macro-F1.
- Saves metrics and class names alongside the model.
- Command-line options for paths, epochs, batch size, learning rate, and fine-tuning.
- No hard-coded Kaggle/Colab paths and no CUDA-only calls.

## Dataset setup

Download the **Intel Image Classification** dataset from Kaggle:
https://www.kaggle.com/datasets/puneet6060/intel-image-classification

Extract it so that the folders look like this (the extra nested folder is common in the Kaggle archive):

```text
data/
  seg_train/
    seg_train/
      buildings/
      forest/
      glacier/
      mountain/
      sea/
      street/
  seg_test/
    seg_test/
      buildings/
      forest/
      glacier/
      mountain/
      sea/
      street/
```

Do not commit the dataset, model weights, or generated outputs to GitHub.

## Environment setup

Python 3.10 or 3.11 is recommended.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS, use the Python environment you created for this project. PyTorch's MPS backend will be used automatically if available.

## Train and evaluate

From the project root:

```bash
python -m src.train \
  --train-dir data/seg_train/seg_train \
  --test-dir data/seg_test/seg_test \
  --epochs 15
```

For a faster head-only transfer-learning baseline, freeze the pretrained backbone:

```bash
python -m src.train \
  --train-dir data/seg_train/seg_train \
  --test-dir data/seg_test/seg_test \
  --epochs 5 \
  --freeze-backbone
```

By default, the model is fine-tuned end-to-end, which is closer to the behavior of the original notebook: its intended freezing statement contains a typo (`required_grad` instead of `requires_grad`), so it does not actually freeze the pretrained layers.

Outputs are written to `outputs/`:
- `best_model.pt` — checkpoint with the highest validation macro-F1
- `metrics.json` — run configuration, validation/test metrics, per-class report and confusion matrix

## Evaluation and reporting

The uploaded notebook's saved output shows one training epoch (training loss 0.495 and validation loss 0.236). It does **not** contain a final test-accuracy result, despite `epochs = 7` being set in the code. Therefore, this repository does not claim that 90%+ accuracy has been verified. Run the pipeline and report the values written to `outputs/metrics.json`.

The held-out `seg_test` set is used only for final evaluation, not checkpoint selection. The validation set is split from `seg_train`.

## Results

### Experimental setup

- **Dataset:** Intel Image Classification
- **Task:** Six-class scene image classification (`buildings`, `forest`, `glacier`, `mountain`, `sea`, `street`)
- **Architecture:** Pretrained Wide ResNet-50-2
- **Approach:** Transfer learning with end-to-end fine-tuning
- **Input image size:** 150 × 150 pixels
- **Optimizer:** Stochastic Gradient Descent (SGD) with momentum
- **Loss function:** Cross-entropy loss
- **Hardware:** Apple Silicon GPU using PyTorch MPS
- **Training duration:** 15 epochs
- **Model selection:** Best checkpoint selected using validation macro-F1

### Held-out test performance

| Metric | Score |
|---|---:|
| Accuracy | 90.67% |
| Macro precision | 90.89% |
| Macro recall | 90.90% |
| Macro-F1 | 90.88% |

The model achieved **90.67% test accuracy** and a **90.88% macro-F1 score** in the 15-epoch experiment. Macro-averaged metrics give equal weight to each class, providing a complementary view of performance beyond overall accuracy.

### Validation performance

The best validation macro-F1 observed during training was **90.93% at epoch 12**. The checkpoint with the highest validation macro-F1 was saved and subsequently evaluated on the test set.

### Reproducibility and limitations

The training pipeline uses a fixed random seed, separate training and evaluation transforms, a stratified validation split, and automatic device selection for CUDA, Apple Silicon MPS, or CPU.

These results are from a recreated and improved implementation inspired by the original Kaggle notebook; they should not be interpreted as a verified reproduction of its original results. The test set was evaluated across multiple training experiments during development, so the reported test score is an observed result rather than a completely untouched final benchmark. Results may vary with the software environment, hardware, and training configuration.

## Project structure

```text
.
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   ├── __init__.py
│   └── train.py
└── tests/
    └── test_project.py
```

## Limitations
- Results depend on the exact dataset files, hardware, random seed, and training configuration.
- A single run is not evidence of generalisation across other image datasets.
- The default full fine-tuning run may be slow on CPU; use `--freeze-backbone` for a quicker baseline.
