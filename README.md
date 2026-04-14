# Predicting SEPTA Regional Rail On-Time Performance
**CSC 382 — Machine Learning | Dr. Amir | Spring 2026**
---

## Project Overview

This project applies supervised machine learning to predict the number of minutes a SEPTA Regional Rail train will arrive late, using route, station, and temporal features from 1.88 million individual trip records. A Random Forest Regressor is the primary model (R² = 0.853), compared against a Linear Regression baseline (R² = 0.067) and a Logistic Regression classifier (F1 = 0.030 vs Random Forest F1 = 0.822).

---

## Repository Structure

```
CSC382_Project/
├── septa_model.py        # Full ML pipeline — load, clean, engineer, train, evaluate
├── README.md             # This file
└── otp.csv               # ⚠️ NOT INCLUDED — see download instructions below
```

---

## ⚠️ Dataset Not Included

`otp.csv` is **not included** in this repository because the file exceeds GitHub's file size limit (~230 MB).

**To download the dataset:**

1. Go to: [https://www.kaggle.com/datasets/septa/on-time-performance](https://www.kaggle.com/datasets/septa/on-time-performance)
2. Sign in or create a free Kaggle account
3. Click **Download** and extract the zip file
4. Place `otp.csv` in the same directory as `septa_model.py`

The dataset contains 1,882,015 individual SEPTA Regional Rail arrival records from March–November 2016 across 7 columns: `train_id`, `direction`, `origin`, `next_station`, `date`, `status`, `timeStamp`.

---

## Environment Setup

Python 3.10 or later is required.

Install all dependencies with:

```bash
pip install scikit-learn pandas numpy matplotlib
```

Tested library versions:

| Library | Version |
|---|---|
| scikit-learn | 1.4 |
| pandas | 2.1 |
| numpy | 1.26 |
| matplotlib | 3.8 |

---

## How to Run

Once `otp.csv` is in the project directory:

```bash
python septa_model.py
```

The script will run the full pipeline automatically:

1. Load and explore the dataset
2. Clean data and parse delay minutes from the status column
3. Engineer features (cyclical encoding, rush hour flag, weekend flag, etc.)
4. Encode categoricals and split 80/20 train/test
5. Train Linear Regression, Random Forest, and Logistic Regression
6. Print all evaluation metrics to the console
7. Save three plots to the working directory:
   - `feature_importance.png`
   - `delay_by_day.png`
   - `delay_by_station.png`

Expected runtime on a standard laptop: **3–6 minutes** (dominated by Random Forest training on 1.5M rows).

---

## Configuration

Two settings at the top of `septa_model.py` can be adjusted:

| Variable | Default | Description |
|---|---|---|
| `RANDOM_STATE` | `42` | Fixed seed for all splitting and model training |
| `DELAY_THRESHOLD` | `6` | Minutes late cutoff for binary delayed/on-time label |
| `TUNE_RF` | `False` | Set to `True` in `main()` to run GridSearchCV hyperparameter tuning (slower, extra credit) |

---

## Results Summary

| Model | RMSE | MAE | R² |
|---|---|---|---|
| Linear Regression (Baseline) | 6.30 min | 3.98 min | 0.067 |
| Random Forest Regressor | 2.50 min | 1.26 min | **0.853** |

| Model | Accuracy | Precision | Recall | F1 |
|---|---|---|---|---|
| Logistic Regression | 0.755 | 0.438 | 0.016 | 0.030 |
| Random Forest (≥6 min) | 0.919 | 0.890 | 0.763 | **0.822** |

---

## Notes on Dataset Acquisition

The original project proposal planned to build a custom multi-year dataset by parsing SEPTA's annual Route Statistics PDFs (2020–2025). A custom PDF parser (`parse_septa_all.py`) was built and successfully extracted 713 route-level records across five years. However, this approach produced only monthly route-level aggregates, which was insufficient in size for meaningful model training. The Kaggle dataset was adopted as a replacement while keeping the same research framing — predicting SEPTA on-time performance from route and temporal features.
