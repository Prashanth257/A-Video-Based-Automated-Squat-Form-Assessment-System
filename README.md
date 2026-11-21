# A Video-Based Automated Squat Form Assessment System for Self-Guided and Unsupervised Training Using Markerless Augmented Feedback with Knowledge of Results and Knowledge of Performance


A hybrid deep learning pipeline for classifying squat exercise form as **Good** or **Bad**
from raw video, using MediaPipe pose estimation, channel-wise topology refinement graph
convolution (CTR-GCN), a Transformer encoder, and an LSTM head.

---

## Repository

**GitHub:** [Prashanth257/A-Video-Based-Automated-Squat-Form-Assessment-System](https://github.com/Prashanth257/A-Video-Based-Automated-Squat-Form-Assessment-System)

---

## Overview

This repository contains the training and evaluation code for a markerless, single-camera
squat form assessment system. The pipeline:

1. Extracts 33 body landmarks per frame using **MediaPipe Pose** (BlazePose).
2. Computes biomechanical features: hip/knee joint angles, torso lean, range of motion,
   angular velocity, and smoothness.
3. Trains a hybrid **CTR-GCN → Transformer Encoder → LSTM** classifier on 45-frame windows.
4. Applies a rule-based biomechanical module alongside the classifier to flag specific
   technique errors (shallow depth, excessive trunk lean, knee valgus, asymmetry).
5. Evaluates on a held-out validation split using a frame-gap splitting strategy to
   prevent temporal leakage.

---

## Repository Structure

```
.
├── Code1.py                   # Main training + evaluation script
├── Code2.py                   # Supplementary analysis / utility script
├── LICENSE                    # MIT License
├── README.md                  # This file
```

> **Note:** The repository currently contains two Python scripts (`Code1.py` and `Code2.py`)
> covering MV7-FIT training and validation. REHAB26 evaluation and the CSV-based error
> annotation validation are described in the manuscript; see [Known Gaps](#known-gaps) below.

---

## Datasets

### MV7-FIT (training and validation)

A Multi-View Raw Video Dataset of Seven Fitness Exercises with Good/Bad Form Labels.

- **Source:** Mendeley Data
- **DOI:** [10.17632/kgbb3yn47p.3](https://doi.org/10.17632/kgbb3yn47p.3)
- **Subset used:** Squat exercise only
- **Composition:** 25 participants × 3 camera views × Good/Bad labels = 150 videos
- **Split:** 75 Good + 75 Bad

### REHAB26-ViewAngles (external evaluation)

A multi-view dataset for studying camera viewpoint in rehabilitation exercises.

- **Source:** Zenodo
- **Version:** v1, published 28 September 2026
- **DOI (related work):** [10.5281/zenodo.13305825](https://doi.org/10.5281/zenodo.13305825)
- **Subset used:** Squat (Exercise code `Ex6`)
- **Provided:** RGB videos from three synchronized cameras, 2D and 3D MediaPipe skeleton
  coordinates (33 joints), repetition segmentation, exercise-correctness labels, and
  error-type annotations [reference:2][reference:3]

**Dataset contents relevant to this work:**

| File | Size | Description |
|---|---|---|
| `videos.zip` | 8.4 GB | RGB videos, organised as `videos/<exercise>/<participant>/<participant>_cam<1\|2\|3>.mp4` |
| `jointSequences-mediapipe.zip` | 686.0 MB | MediaPipe Pose 2D/3D skeleton arrays, shape `(frames, 33, 4)` |
| `Error_types.csv` | 934 Bytes | Error codes for incorrect repetitions |
| `joint_names.txt` | 511 Bytes | Joint labels for the 33 MediaPipe landmarks |

**Recording conditions:** Three synchronized iPhone 16e smartphones (30 FPS, 12 MP,
f/1.6) at relative angles of 0° (cam1), 45° (cam2), and 90° (cam3), with participants
gradually rotating from 0° to 90° in 10° steps to yield 29 unique viewing angles [reference:4].
Skeletons were estimated with MediaPipe Pose v0.10.20 (Full model) and stored as NumPy
arrays with no filtering or smoothing [reference:5].

Download both datasets and place them in a local directory. Update the dataset paths in
`Code1.py` before running.

---

## Installation

```bash
git clone https://github.com/Prashanth257/A-Video-Based-Automated-Squat-Form-Assessment-System.git
cd A-Video-Based-Automated-Squat-Form-Assessment-System
pip install -r requirements.txt
```

`requirements.txt`:

```
torch>=2.0
torchvision>=0.15
mediapipe>=0.10
opencv-python>=4.8
numpy>=1.23
pandas>=1.5
scikit-learn>=1.2
matplotlib>=3.7
seaborn>=0.12
tqdm>=4.65
```

---

## Usage

1. Open `Code1.py` in your Python environment (or convert to a notebook if preferred).
2. Set `DATASET_ROOT` to the directory containing the MV7-FIT squat videos.
3. (Optional) Set `REHAB26_ROOT` to the extracted REHAB26-ViewAngles directory.
4. Run `Code1.py` sequentially. Training takes approximately **30–60 minutes on CPU**
   for the full 50 epochs; a CUDA GPU is recommended.

### Expected runtime on CPU

| Stage | Time |
|---|---|
| Landmark extraction (75 Good + 75 Bad videos) | ~28 min |
| Training (50 epochs) | ~10–20 min |
| Validation + result export | ~1 min |

---

## Model Architecture

| Stage | Component | Details |
|---|---|---|
| 1 | CTR-GCN Block 1 | 12 → 32 channels, temporal kernel (9,1), dropout 0.3 |
| 2 | CTR-GCN Block 2 | 32 → 64 channels, temporal kernel (9,1), dropout 0.3 |
| 3 | Premutation | Learnable joint + time embeddings, normalised |
| 4 | Transformer Encoder | `d_model=64`, 8 attention heads, FFN dim 256, GELU, pre-LN |
| 5 | Frame Projector MLP | 1088 → 256 → 128, ReLU, dropout 0.4 |
| 6 | LSTM | 2 layers, hidden 128, bidirectional, temporal mean pooling |
| 7 | Classifier Head | 256 → 256 → 2 logits, ReLU, dropout |

**Input tensor:** `(B, 12, 45, 17)` — batch × (x, y, z for 4 joints) × 45 frames × 17 joints

---

## Training Configuration

| Hyperparameter | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate (peak) | `1e-3` |
| Weight decay | `1e-2` |
| β | (0.9, 0.999) |
| Scheduler | OneCycleLR, 30% warm-up, cosine decay |
| Batch size | 32 |
| Epochs | 50 |
| Loss | Cross-entropy with label smoothing ε = 0.1 |
| Class weights | Inverse class frequency |
| Gradient clipping | max norm 1.0 |
| Early stopping | Patience 10 on validation weighted F1 |
| Mixup | α = 1.0 |
| Augmentation | Temporal warp (0.8–1.2×), Gaussian noise (σ ∈ [0.01, 0.05]), temporal dropout (5–15% frames), random scaling (0.9–1.1×) |
| Seed | Fixed (`torch.manual_seed(42)`) |

---

## Data Splitting Strategy

A **frame-gap split** is used to avoid temporal leakage while maximising the number of
training windows from a limited video set:

- **Training:** first 65% of frames from each video
- **Gap:** 15 frames discarded (not used in either split)
- **Validation:** final 20% of frames from each video

---

## Outputs

After running `Code1.py`, the following files are written to `outputs/`:

- `classification_report.txt` — per-class precision, recall, F1
- `confusion_matrix.png` — validation confusion matrix
- `roc_curve.png` — ROC curve and AUC
- `training_curves.png` — train/val loss, accuracy, and F1 per epoch
- Console log — full training history table with per-epoch learning rate and overfit gap

---

## Known Gaps

The following analyses described in the manuscript are **not implemented in the current
repository** and must be reproduced separately or added before submission:

- **REHAB26 external evaluation:** the code contains only MV7-FIT training and
  validation. REHAB26 videos are not loaded, and no cross-dataset evaluation is run.
- **CSV-based error annotation validation:** the agreement between system-detected
  errors and the REHAB26 error-type annotations is not computed in this repository.
- **Real-time visual feedback loop:** the overlay rendering and live skeleton feedback
  described in the manuscript are not implemented here.
- **Post-exercise analytics output:** ROM, smoothness, and symmetry are computed as
  features but are not exported as a per-video report.

To claim these results, add the corresponding code or a supplementary script, and
update this README with the actual numbers.

---

## Reproducibility Notes

- All random seeds are fixed at the top of `Code1.py`.
- Deterministic cuDNN is enabled; exact numeric reproduction may still differ slightly
  across GPU architectures.
- The MediaPipe version affects landmark output; pin `mediapipe` to the version used
  for the reported results.

---

## Anonymity for Double-Blind Review

If you are linking this repository from a manuscript under double-anonymous review:

1. Host the repository under an anonymous account.
2. Remove all author names, emails, institution names, and any identifying commit
   messages (`git log --format='%an <%ae>' | sort -u` to check).
3. Do not link the GitHub profile to ORCID or institutional pages.
4. Use an anonymous DOI service (e.g., anonymous.4open.science) if the journal
   requires a citable archived snapshot.
5. Strip any output files that may contain local file paths with your username.

After acceptance, replace the anonymised link with the permanent repository.

---

## Citation

If you use this code, please cite:

```bibtex
@article{key,
  title   = {A Video-Based Automated Squat Form Assessment System for Monocular-Unsupervised
             Training Using Markerless Augmented Feedback with Knowledge of Results and
             Knowledge of Performance},
  author  = {Authors withheld for review},
  journal = {Journal of King Saud University Computer and Information Sciences},
  year    = {2026},
  note    = {Under review}
}
```

If you use the REHAB26-ViewAngles dataset, cite the related paper as required by its
license [reference:7]:

```bibtex
@article{janosova2026rehab26,
  title   = {Impact of Patient Orientation in Single- and Multi-View Camera Environments
             for AI-based Rehabilitation Monitoring},
  author  = {Jánošová, M. and Lang, A. and Budikova, P. and Sedmidubsky, J.},
  journal = {arXiv},
  year    = {2026},
  doi     = {10.48550/arXiv.2609.35726}
}
```

---

## License

This project is released under the MIT License. See [LICENSE](https://github.com/Prashanth257/A-Video-Based-Automated-Squat-Form-Assessment-System/blob/main/LICENSE) for details.

The MV7-FIT dataset is governed by its own terms on Mendeley Data. The REHAB26-ViewAngles
dataset is restricted to academic, non-profit, non-commercial research use and requires
citation of the related paper [reference:8]. Refer to the original dataset pages before
redistribution.

---

## Acknowledgements

- [MediaPipe](https://github.com/google/mediapipe) for pose estimation
- [MV7-FIT](https://doi.org/10.17632/kgbb3yn47p.3) dataset authors
- [REHAB26-ViewAngles](https://zenodo.org/records/23018128) dataset authors
- The CTR-GCN, Transformer, and LSTM implementations are adapted from their respective
  public reference implementations (see manuscript references)
