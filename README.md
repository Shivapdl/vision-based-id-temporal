# TARA: Temporal Adaptive Recognition Architecture

### A Non-Invasive Alternative to RFID — Self-Sustaining 3D Identification of Group-Housed Livestock
![TARA conceptual framework](assets/figure1_framework.png)



[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-orange.svg)](https://www.tensorflow.org/)

Official implementation of **TARA**, a non-invasive, vision-based framework for identifying group-housed sows from 3D point cloud sequences captured at commercial electronic feeding stations (EFS). TARA reframes identification as a **temporal consensus problem**, combining frame-level deep learning with visit-level majority voting and an autonomous self-recalibration loop, reaching **100% visit-level identification accuracy** without RFID tags.

> Shiva Paudel, TsungCheng Tsai, Dongyi Wang — University of Arkansas, Fayetteville, AR, USA

---

## Quickstart

```bash
# 1. Clone
git clone https://github.com/Shivapdl/vision-based-id-temporal.git
cd vision-based-id-temporal

# 2. Set up an isolated environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 3. Get data
#    - Dataset available on request (see docs/DATA.md), OR
#    - Drop your own point clouds into data/raw/ (same layout)

# 4. Configure
#    Edit configs/default.yaml — set data.num_classes and any paths/thresholds

# 5. Run
jupyter notebook
```

Notebooks read all settings from `configs/default.yaml` via `src/config.py`, so **no hardcoded paths** — it runs the same on any machine.

### Run order

1. `Segment_with_individual_calibration.ipynb` — preprocessing
2. `PointNet_1.5K_V5.ipynb` — train the Day 1 baseline
3. `ID_And Visit_Match.ipynb` — visit-level consensus
4. `Next_Day_Validation.ipynb` — cross-day evaluation
5. `PointNet_1.5K_V5_D2.ipynb` — autonomous re-calibration

---

## Overview

Current industry standards rely on RFID ear tags, which are invasive, prone to loss, and limited by antenna range. TARA replaces them with a self-sufficient, semi-supervised vision pipeline built on three ideas:

- **Temporal Majority Voting** — identity is assumed invariant across a feeding visit, so frame-level predictions are aggregated into a single robust ID, filtering postural noise and sensor artifacts.
- **Confidence Gating** — a strict per-frame threshold (τ = 0.99) rejects degraded point clouds before they corrupt the visit-level decision.
- **Autonomous Re-calibration** — high-confidence visits are retroactively pseudo-labeled and added to a fine-tuning pool, letting the model adapt to morphological drift (growth, pregnancy) with no human intervention.

The backbone is a **PointNet** classifier (TensorFlow/Keras) with dual T-Net spatial/feature transforms, operating on 1,500-point dorsal surface manifolds.

---

## Key Results

Longitudinal dataset of 9 sows, 89,944 RGB-D frames over 96 hours, collected in an operational commercial barn (IACUC #25059).

| Model State            | Test Set | Frame % | Visit %  | Conv. % |
| ---------------------- | -------- | ------- | -------- | ------- |
| Base Model (Day 1)     | Day 2    | 93.94   | 96.94    | 82.78   |
| Base Model (Day 1)     | Day 3    | 94.17   | 96.30    | 77.14   |
| Re-calib. 1 (Pseudo D2)| Day 3    | 96.77   | 99.13    | 80.42   |
| Re-calib. 1 (Pseudo D2)| Day 4 †  | 97.03   | 100.00   | 68.00   |
| Re-calib. 2            | Day 4 †  | 97.73   | 100.00   | 79.54   |

† Reduced cohort due to session termination.

**Peak frame-level accuracy: 97.73% — Visit-level identification rate: 100%.**

### Component Ablation

| Variant                         | τ    | Vote | Recalib. | D3 Visit % | D4 Visit % |
| ------------------------------- | ---- | ---- | -------- | ---------- | ---------- |
| A. Naive frame-level            | 0.00 | –    | –        | 72.67      | 73.06      |
| B. + Confidence gating          | 0.99 | –    | –        | 96.09      | 97.83      |
| C. + Majority voting            | 0.99 | ✓    | –        | 96.30      | 100.00     |
| D. + Recalibration (full TARA)  | 0.99 | ✓    | ✓        | 99.15      | 100.00     |

---

## Method

```
Depth Stream ─▶ Visit Labeling ─▶ PointNet (fθ) ─▶ Frame ID ŷv,i
                                                        │
                                                  Majority Voting
                                                        │
                                                  Pseudo-Label Ŷv ─▶ Autonomous Re-calibration
```

A final visit identity is assigned only when a visit meets minimum duration (K = 10 confident frames) and consensus (γ = 0.50); otherwise it is rejected to prevent ID flickering and identity contamination.

---

## Repository Structure

```
vision-based-id-temporal/
├── configs/
│   └── default.yaml          # all paths & hyperparameters live here
├── src/
│   └── config.py             # config loader / path helper for notebooks
├── data/
│   ├── raw/                  # your input captures (gitignored)
│   └── processed/            # preprocessing output (gitignored)
├── checkpoints/              # saved weights (gitignored)
├── results/                  # metrics, figures (gitignored)
├── docs/
│   └── DATA.md               # dataset access & layout
├── *.ipynb                   # pipeline notebooks (see table below)
├── requirements.txt
└── README.md
```

| Notebook | Stage | Description |
| -------- | ----- | ----------- |
| `Segment_with_individual_calibration.ipynb` | Preprocessing | Segmentation, voxel downsampling, ROI crop, dorsal isolation |
| `PointNet_1.5K_V5.ipynb` | Training | PointNet on Day 1 (80/20 split), 1,500-point sampling |
| `PointNet_1.5K_V5_D2.ipynb` | Re-calibration | Fine-tune on Day 2 pseudo-labels |
| `Next_Day_Validation.ipynb` | Evaluation | Cross-day validation, drift analysis |
| `ID_And Visit_Match.ipynb` | Consensus | Visit-level majority voting / ID matching |
| `Conversion_%.ipynb` | Analysis | Visit-to-ID conversion rate |
| `Test Point Clouds.ipynb` | Utilities | Test-set inference |
| `Visualize Point Clods.ipynb` | Visualization | Point cloud rendering |

---

## Data

The dataset is **available on request** — see [`docs/DATA.md`](docs/DATA.md) for how to request access and the expected folder layout. You can also run TARA on your **own** point clouds without the original dataset.

## Hardware (data collection)

3 × Intel RealSense D435 on Gestal ESF feeding stations, Jetson Orin Nano edge modules. Capture triggered at Δt = 2 s when a subject is within 0.6 m.

---

## Citation

```bibtex
@article{paudel2026tara,
  title   = {A Non-Invasive Alternative to RFID: Self-Sustaining 3D Identification of Group-Housed Livestock},
  author  = {Paudel, Shiva and Tsai, TsungCheng and Wang, Dongyi},
  year    = {2026},
  institution = {University of Arkansas}
}
```

## License

MIT License — see [`LICENSE`](LICENSE).

## Contact

Shiva Paudel — `shivap@uark.edu` · Dongyi Wang — `dongyiw@uark.edu`
