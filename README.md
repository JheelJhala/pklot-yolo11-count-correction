# pklot-yolo11-count-correction
Correcting YOLO11 parking-lot counts on PKLot with classical ML. ES 678 project.
# Can Classical ML Correct YOLO11 Parking-Lot Counts?

ES 678 Machine Learning course project, IIT Gandhinagar.
**Author:** Your Name (Roll Number)

Pretrained YOLO11 detectors count cars without any lot-specific training, but their counts of occupied parking spaces can be systematically wrong. This project trains Ridge, a decision tree and an RBF SVR on summary features of YOLO11 outputs (detect, segment and oriented boxes) to predict the true occupied count on [PKLot](https://web.inf.ufpr.br/vri/databases/parking-lot-database/). It compares them with raw YOLO, a tuned confidence threshold and linear calibration, across weather conditions and on a held-out lot, using a day-level bootstrap.

## Main result

Test MAE in cars, with 95% day-level bootstrap intervals. This is Table 1 of the paper.

| Method | Sunny | Cloudy | Rainy | New lot |
|---|---|---|---|---|
| Raw YOLO (T = 0.25) | 18.0 | 17.3 | 38.5 | 14.5 |
| Raw + tuned T = 0.65 | 17.1 | 16.4 | 49.5 | 4.2 |
| Linear calibration | 22.7 | 27.4 | 36.9 | 12.7 |
| Ridge | 8.1 | 8.3 | 22.8 | 5.9 |
| Decision tree | 6.3 | 7.2 | 26.6 | 7.3 |
| **SVR (RBF)** | **5.2** | **3.9** | **17.0** | 4.3 |

All three models beat raw YOLO and linear calibration in every condition. On the new lot, the SVR ties the tuned threshold on MAE and has lower RMSE. The tuned threshold itself is unstable: it moves to 0.15 when the training days change.

## Repository layout

```
README.md
requirements.txt
ml_project.ipynb          full pipeline, run on Kaggle (outputs kept)
data/
  labels.csv              one row per image: lot, weather, occupied count, path
  splits.csv              train / test_sunny / test_cloudy / test_rainy / test_newlot
  features.csv            per-image feature vectors at T = 0.25 (model input)
  det/
    chunk_*_det.csv       every YOLO11 detection at conf >= 0.05 (25 chunks of 500 images)
    chunk_*_img.csv       per-image brightness and contrast
results/
  baseline_results.csv    raw, tuned-threshold and linear-calibration baselines
  cv_tuning.csv           chosen hyperparameters (GroupKFold over days)
  ml_results_vs_raw.csv   model MAE and paired differences vs raw YOLO
  ml_results_vs_tuned.csv ... vs the tuned threshold
  ml_results_vs_linear.csv ... vs linear calibration
  ablation.csv            feature-group ablation (Table 2)
  sensitivity.csv         MAE vs threshold per lot and weather
  worst_frames.csv        largest errors, used for the failure audit
  sensitivity.png         Figure 2
  audit_panel.pdf         Figure 1
```

## Data access

The PKLot images are **not** included because the dataset is several GB.

- **Official source:** PKLot, Almeida et al. (2015), from the [UFPR VRI database page](https://web.inf.ufpr.br/vri/databases/parking-lot-database/). The licence is CC BY 4.0.
- **Copy used for this project:** the Kaggle dataset [`blanderbuss/parking-lot-dataset`](https://www.kaggle.com/datasets/blanderbuss/parking-lot-dataset).

The notebook expects this layout:
```
<ROOT>/PKLot/PKLot/{PUCPR,UFPR04,UFPR05}/{Sunny,Cloudy,Rainy}/<date>/<image>.jpg + .xml
```
Set `ROOT` in the first data cell. On Kaggle, it is `/kaggle/input/datasets/blanderbuss/parking-lot-dataset`.

You need the images only to rerun detection or the visual failure audit. **Every table and figure except the audit panel can be reproduced from `data/` alone.**

## Environment

- Kaggle notebook, Python 3, with a GPU (needed for YOLO inference only).
- Packages are listed in `requirements.txt`: `ultralytics`, `scikit-learn`, `pandas`, `numpy`, `opencv-python`, `matplotlib`.
- Models: `yolo11n.pt`, `yolo11n-seg.pt` and `yolo11n-obb.pt`. Ultralytics downloads them automatically.

## How to reproduce

### Option A: from saved detections (fast, CPU only)

1. Upload the `data/` folder as a Kaggle dataset, or use it locally.
2. Copy it to the working directory the notebook reads from:
   ```
   !mkdir -p /kaggle/working && cp -r /kaggle/input/<your-dataset>/data/* /kaggle/working/
   ```
3. Run the analysis cells, starting at the feature-building cell. The YOLO inference cell can be skipped.

This reproduces all baselines, model results, ablations, bootstrap intervals and the sensitivity figure.

### Option B: full pipeline (needs a GPU and PKLot)

| Step | Notebook cells | Output |
|---|---|---|
| Parse XML annotations | 1–2 | `labels.csv` |
| Install Ultralytics | 3 | |
| Build splits (seed 42, 20% of sunny days per lot held out) | 6 | `splits.csv` |
| YOLO11 detect / segment / OBB inference at conf 0.05, imgsz 1280 | 7* | `det/chunk_*` |
| Features at T = 0.25 | 8–9 | `features.csv` |
| Raw error summary and label-mismatch check | 10–13 | printed |
| Baselines (raw, tuned T, linear calibration) | 14 | `baseline_results.csv` |
| Ridge / tree / SVR with GroupKFold tuning, bootstrap, ablation | 15–16 | `cv_tuning.csv`, `ml_results_*.csv`, `ablation.csv` |
| Purged run, RMSE bootstrap, date-folder checks | 17–23 | printed |
| Threshold sensitivity | 24 | `sensitivity.csv`, `sensitivity.png` |
| Failure audit and OBB analysis | 26–33 | `worst_frames.csv`, `audit_panel.pdf` |

\*The inference code in cell 7 is stored as a Markdown cell so it doesn't rerun by accident. It takes several hours on a GPU. To run it, change the cell type to Code.

The notebook was developed interactively on Kaggle. If you run it top to bottom, follow the step order in the table rather than re-executing cells 4, 5 and 25, which are backup and path-check helpers.

## Reproducibility notes

- **Seeds:** split `default_rng(42)`; bootstrap `default_rng(0)` with 2,000 resamples of days; decision tree `random_state=0`; audit samples `random_state=0/1`.
- **No test data is used for tuning.** Hyperparameters and the tuned threshold are chosen on train only, with 5-fold GroupKFold over days. Feature scaling is fit inside the sklearn `Pipeline`.
- **Thresholds are applied after inference.** Detections are saved once at conf 0.05, so any threshold can be evaluated without rerunning YOLO.
- **Known data issues:**
  - One PUCPR image (2012-11-06 18:48:46) has no XML annotation and is excluded, leaving 12,416 images.
  - Two date folders (PUCPR 2012-09-16, UFPR05 2013-14-16) don't match their file dates. Grouping uses the filename date; this left every split unchanged.
  - 15 training days also contribute cloudy or rainy test frames. The purged rerun is in the notebook.
- Per-lot breakdowns were added after seeing the raw results. The intervals for sunny sub-rows (4–5 days) are descriptive only.

## Citation

If you use this code, please also cite PKLot:

> P. R. L. de Almeida, L. S. Oliveira, A. S. Britto Jr., E. J. Silva Jr., A. L. Koerich. PKLot: a robust dataset for parking lot classification. *Expert Systems with Applications*, 42(11):4937–4949, 2015.
