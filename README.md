# Solar Filament Segmentation 2026

A two-stage instance-segmentation pipeline for the **Solar Filament Segmentation 2026** Kaggle competition. It detects individual solar filaments with Mask R-CNN and refines each detected instance with a U-Net-style model using a ResNet-18 encoder.

Competition: https://www.kaggle.com/competitions/filament-segmentation-2026

## Overview

The notebook `filament_segmentation_v4_final.ipynb` implements the following workflow:

1. Downloads and extracts the competition data using the Kaggle CLI.
2. Parses the instance-level filament annotations.
3. Splits the training data into training and validation sets, grouped by month.
4. Trains a **Mask R-CNN ResNet-50-FPN detector** to generate filament candidates and coarse masks.
5. Trains a **U-Net refiner with a ResNet-18 encoder** to improve the masks of individual candidates.
6. Selects predictions using score filtering, mask-IoU non-maximum suppression (NMS), and minimum-area filtering.
7. Evaluates configurations using Panoptic Quality (PQ), where a match requires IoU > 0.5.
8. Runs inference on the test set and creates a Kaggle submission CSV.

The notebook also includes optional experiments such as a second detector, YOLO11-seg, test-time augmentation (TTA), candidate re-scoring, and mask fusion.

## Repository contents

```text
.
├── filament_segmentation_v4_final.ipynb
├── README.md
└── requirements.txt
```

Training checkpoints, cached predictions, configuration files, and generated submissions are written to the working directory (`filament_work/`) or Google Drive when running in Colab. They are not required to be committed to GitHub.

## Environment

**Recommended environment:** Google Colab with a GPU runtime (for example, NVIDIA T4).

The notebook was designed around Colab paths, Kaggle competition downloads, and optional Google Drive persistence. A GPU is strongly recommended; CPU execution may be very slow and may not be practical for full training.

Python 3.10 or 3.11 is recommended. Install a PyTorch and torchvision pair compatible with the selected Python version and CUDA runtime.

## Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd <YOUR_REPOSITORY_FOLDER>
```

### 2. Install dependencies

In a Colab notebook cell or a compatible Python environment:

```bash
pip install -r requirements.txt
```

Install the correct PyTorch/torchvision build for your environment if the versions installed by pip are not compatible with your CUDA runtime. Use the official selector: https://pytorch.org/get-started/locally/

### 3. Configure Kaggle access

The notebook downloads the competition dataset using the Kaggle CLI.

1. Accept the competition rules on Kaggle: https://www.kaggle.com/competitions/filament-segmentation-2026
2. Create or obtain your Kaggle API credentials.
3. In Google Colab, add your Kaggle API token as a Colab Secret named `KAGGLE_API_TOKEN` (recommended). The notebook requests credentials without displaying the token.
4. Alternatively, configure the Kaggle CLI credentials using Kaggle's documented method: https://github.com/Kaggle/kaggle-api

**Never commit Kaggle API tokens, `kaggle.json`, or other credentials to GitHub.**

### 4. Open and run the notebook

Open `filament_segmentation_v4_final.ipynb` in Google Colab and select a GPU runtime.

Run the notebook cells from top to bottom. For an initial end-to-end check, set:

```python
SMOKE_TEST = True
```

After confirming that the pipeline runs, change it to:

```python
SMOKE_TEST = False
```

Then run the full training and evaluation workflow.

## Configuration

The main configuration is in the notebook's **Config** section. Important settings include:

| Setting | Purpose |
|---|---|
| `PREPROC` | Image preprocessing mode (`clahe`, `raw`, `msr`, or `msr_clahe`) |
| `SMOKE_TEST` | Enables a small, faster test run |
| `DET_EPOCHS` | Number of detector training epochs |
| `DET_BATCH` | Detector batch size |
| `REF_EPOCHS` | Number of refiner training epochs |
| `REF_SIZE` | Refiner input resolution |
| `VAL_SUBSET_N` | Number of validation images used for monitoring and tuning |
| `RETRAIN_DET` | Forces detector training to restart |
| `RETRAIN_REF` | Forces refiner training to restart |
| `RECOMPUTE_CACHE` | Recomputes cached detections and refinements |

Reduce batch sizes if GPU memory is insufficient.

## Checkpoints and recovery

The notebook saves detector/refiner checkpoints and prediction caches so work can resume after a Colab reconnect. When Google Drive is available, the working directory is stored under:

```text
/content/drive/MyDrive/filament_work/
```

Without Google Drive, files are stored locally in:

```text
./filament_work/
```

After reconnecting, rerun the setup and configuration cells. Follow the notebook's restore instructions before continuing with tuning or inference.

## Optional components

- **YOLO11-seg:** The optional YOLO experiment requires `ultralytics`. It is included in `requirements.txt`, but the YOLO branch is disabled by default (`USE_YOLO = False`).
- **Second detector:** An optional second Mask R-CNN model can be trained and combined with the main detector.
- **TTA and candidate re-scoring:** Optional experiments evaluate predictions from flipped image passes and use candidate-level features to improve selection.

These experiments are optional; the main Mask R-CNN + refiner pipeline does not require enabling them.

## Output

The notebook generates a Kaggle submission file named:

```text
submission.csv
```

Depending on the optional experiments selected, it may also generate files such as `submission_rescored.csv` or `submission_tta.csv`. Use the notebook's final sanity-check and download cells before submitting to Kaggle.

## Evaluation

The notebook evaluates instance predictions using **Panoptic Quality (PQ)**. A predicted filament and a ground-truth filament are considered a match when their Intersection over Union (IoU) is greater than 0.5. The pipeline also checks that final predicted masks within an image do not overlap, as required by the competition submission format.

## Notes

- The notebook trains models from public COCO/ImageNet pretrained weights; it does not load the separate `final_detector.pt` or `final_refiner.pt` files described in its introductory notes.
- Full training is computationally expensive and intended for a GPU runtime.
- Results can vary with the runtime, dataset version, random seed, and installed library versions.
- Check the competition page for the latest rules, dataset details, and submission requirements.
