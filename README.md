# AI-Enhanced Mpox Diagnosis

Research code repository for **“A New AI-Enhanced Public Health Framework for Monkeypox Diagnosis”**.

This repository is intended to document and support reproducibility of a six-class skin-lesion image classification study using a ConvNeXt-Base backbone enhanced with Squeeze-and-Excitation (SE) attention.

> **Reproducibility note:** This repository is a structured research-code release. Before claiming exact reproduction of the paper, integrate and validate the original experiment scripts, checkpoint files, dataset version, split manifests, and software environment. The templates included here should not be represented as the exact source of reported results until that validation is complete.

## Method overview

The research pipeline is organized into the following stages:

1. Dataset inspection and preprocessing.
2. Class-count reporting and duplicate-image screening using MD5 file hashes.
3. Stratified five-fold cross-validation.
4. Training-time augmentation according to the experiment's class-specific policy.
5. Two-stage transfer learning with ImageNet-1K-pretrained ConvNeXt-Base.
6. Squeeze-and-Excitation attention and an advanced classification head.
7. Independent test-set evaluation.
8. Test-time augmentation (TTA), where enabled in the corresponding experiment.
9. Grad-CAM visual explanations.

## Target classes

| Code | Class |
|---|---|
| CHP | Chickenpox |
| CWP | Cowpox |
| HEALTHY | Healthy |
| HFMD | Hand-foot-mouth disease |
| MSL | Measles |
| MKP | Monkeypox |

The class-to-index mapping used by a training run must be saved with that run. Do not assume alphabetical ordering if the original experiment used a different mapping.

## Reported results

The following values were supplied for the paper's current results. They are recorded here as reported values, not independently reproduced by this repository template.

| Protocol | Metric | Reported result |
|---|---|---:|
| Five-fold cross-validation | Macro-F1 | 84.23 ± 2.60% |
| Independent test set | Accuracy | 88.50% |
| Independent test set | Macro-F1 | 86.21% |
| Independent test set | Macro ROC-AUC | 98.28% |

Add the per-fold results, confidence intervals (if calculated), class-wise metrics, and evaluation scripts before describing these numbers as independently reproducible.

## Repository layout

```text
AI-Enhanced-Mpox-Diagnosis/
├── README.md
├── LICENSE
├── CITATION.cff
├── requirements.txt
├── .gitignore
├── configs/
│   └── config.yaml
├── data/
│   └── README.md
├── notebooks/
│   └── README.md
├── src/
│   ├── data/
│   │   ├── duplicate_detection.py
│   │   └── inspect_dataset.py
│   ├── models/
│   │   └── convnext_se.py
│   ├── training/
│   │   └── README.md
│   ├── evaluation/
│   │   └── README.md
│   └── explainability/
│       └── README.md
├── results/
│   ├── figures/
│   └── tables/
└── docs/
    ├── methodology.md
    └── experimental_setup.md
```

## Environment

Python 3.10 or 3.11 is recommended. Install PyTorch and torchvision using the command appropriate for your operating system and CUDA version from the official PyTorch installation selector. Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

Check the installation:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__); print('CUDA:', torch.cuda.is_available())"
```

## Dataset

The study uses the **Mpox Skin Lesion Dataset Version 2.0 (MSLD v2.0)**. Dataset access and licensing are governed by the original dataset host and its terms. This repository does not redistribute the images.

Read [`data/README.md`](data/README.md) before preparing the data. Keep the independent test set separate from model fitting, model selection, and augmentation.

## Model

The model template in `src/models/convnext_se.py` implements the high-level design:

- ConvNeXt-Base feature extractor.
- SE attention with reduction ratio 16.
- Global average pooling.
- LayerNorm and dropout-based classifier head.
- GELU activations.

The discussed classifier-head configuration is:
`LayerNorm → Dropout(0.35) → Linear(1024, 1024) → GELU → Dropout(0.30) → Linear(1024, 512) → GELU → Dropout(0.25) → Linear(512, number_of_classes)`.

The two-stage training schedule (freeze backbone, then unfreeze the last two feature stages) must be implemented and checked against the final experiment script before use for reproduction. See `docs/experimental_setup.md`.

## Data quality checks

Run the dataset inspection and duplicate checks:

```bash
python -m src.data.inspect_dataset --root /path/to/dataset
python -m src.data.duplicate_detection --root /path/to/dataset
```

The duplicate detector uses MD5 hashes to identify byte-identical files, even if their filenames differ. MD5 is used here as a file-identity screening mechanism, not as a security or authenticity guarantee. It will not detect visually identical images that have been re-encoded or resized.

## Training and evaluation

The training, cross-validation, independent-test, TTA, and Grad-CAM code from the final experiments should be added to the corresponding `src/` directories. The README files in those directories specify the expected command-line behavior and outputs.

Do not run cross-validation over a dataset that already contains a held-out independent test set. The test set must remain untouched until the final evaluation.

## Results and artifacts

Place paper figures in `results/figures/` and CSV/Markdown result tables in `results/tables/`. Do not commit large checkpoints, private data, or raw image datasets unless distribution is permitted and repository size/licensing have been considered. Consider Git LFS or a DOI-backed research archive for large model weights.

## Citation

When the paper is published, update `CITATION.cff` with the final author list, venue, year, DOI, and publication URL. Until then, cite the manuscript using its current bibliographic status.

## Limitations and responsible use

This is a research framework for image classification, not a clinically validated diagnostic device. Predictions must not be used as a substitute for assessment by qualified healthcare professionals. Performance on the research dataset does not establish generalization to other populations, devices, image acquisition conditions, or clinical settings.

## License

The code license in this repository does not change the license or terms governing the external dataset. Review `LICENSE` and the dataset's terms before redistribution.
