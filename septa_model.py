# =============================================================================
# CSC 382 — Machine Learning Project
# Predicting SEPTA Regional Rail On-Time Performance
# Dataset: SEPTA On-Time Performance (Kaggle)
#          https://www.kaggle.com/datasets/septa/on-time-performance
# Author:  Garrett Crowner  |  Student ID: 0998499
# =============================================================================
#
# ACTUAL CSV COLUMNS (otp.csv):
#   train_id     - train number identifier
#   direction    - Northbound / Southbound
#   origin       - departure station
#   next_station - next stop (used as destination proxy)
#   date         - service date
#   status       - "On Time", "2 min", "Arrived", "Canceled", etc.
#   timeStamp    - datetime of the record
#
# NOTE: There is no raw `late` column — minutes late is parsed from `status`.
#       "On Time" -> 0,  "5 min" -> 5,  "Arrived" -> 0,  "Canceled" -> dropped
# =============================================================================

import re
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score,
                             accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

# ── reproducibility ───────────────────────────────────────────────────────────
RANDOM_STATE = 42

# ── classification threshold ─────────────────────────────────────────────────
# SEPTA's official OTP definition: on time = arrives within 5 min 59 sec.
# We use 6 minutes as the binary cutoff: late >= 6 → delayed (is_delayed = 1)
DELAY_THRESHOLD = 6

# ── path to CSV ───────────────────────────────────────────────────────────────
DATA_PATH = "otp.csv"


# =============================================================================
# HELPER — parse minutes late from status string
# =============================================================================

def parse_late_minutes(status: str) -> float:
    """
    Convert the status string to a numeric minutes-late value.

    Examples:
        "On Time"  -> 0
        "Arrived"  -> 0
        "2 min"    -> 2
        "15 min"   -> 15
        "Canceled" -> NaN  (these rows get dropped in clean_data)
    """
    if not isinstance(status, str):
        return 0.0
    s = status.strip().lower()
    if s in ("on time", "arrived", ""):
        return 0.0
    if "cancel" in s:
        return np.nan          # flagged for removal
    m = re.search(r"(\d+)\s*min", s)
    if m:
        return float(m.group(1))
    # Any other non-empty status treated as 0 (on time)
    return 0.0


# =============================================================================
# 1. DATA LOADING & EXPLORATION
# =============================================================================

def load_data(filepath: str) -> pd.DataFrame:
    """Load otp.csv, print summary stats, parse late minutes from status."""
    df = pd.read_csv(filepath, low_memory=False)

    print("=== 1. DATA EXPLORATION ===")
    print(f"Raw shape: {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Columns  : {list(df.columns)}\n")

    # Sanity check — confirm expected columns exist
    required = ["train_id", "origin", "next_station", "status", "date"]
    missing  = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"\nMissing columns: {missing}\n"
            f"Available      : {list(df.columns)}\n"
            f"Check that you are using the correct otp.csv file."
        )

    # Standardise column names to internal names used throughout
    df = df.rename(columns={
        "train_id":     "trainno",
        "next_station": "dest",
        "timeStamp":    "timestamp",
    })

    # Parse minutes late from status string
    df["late"] = df["status"].apply(parse_late_minutes)

    # Quick sanity check
    print(f"--- SANITY CHECK ---")
    print(f"Shape : {df.shape[0]:,} rows x {df.shape[1]} columns")
    print(f"Sample:\n{df[['trainno','origin','dest','status','late','date']].head(3).to_string()}")
    print(f"--------------------\n")

    print(df["late"].describe().to_string())
    print(f"\nStatus value counts (top 10):\n{df['status'].value_counts().head(10)}")
    print(f"\nUnique origins      : {df['origin'].nunique()}")
    print(f"Unique destinations : {df['dest'].nunique()}")
    print(f"Unique trains       : {df['trainno'].nunique()}")
    print(f"\nMissing values:\n{df.isnull().sum()}")
    return df


# =============================================================================
# 2. DATA CLEANING
# =============================================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    - Drop canceled trips (late = NaN from parse_late_minutes)
    - Parse date column
    - Cap extreme outliers (>120 min)
    - Drop duplicate rows
    """
    print("\n=== 2. DATA CLEANING ===")
    original = len(df)

    # Drop canceled rows (NaN late values)
    canceled = df["late"].isna()
    df = df[~canceled].copy()
    print(f"Removed {canceled.sum():,} canceled records")

    # Parse date
    df["date"] = pd.to_datetime(df["date"], infer_datetime_format=True, errors="coerce")
    bad_dates = df["date"].isna().sum()
    if bad_dates:
        df = df.dropna(subset=["date"])
        print(f"Dropped {bad_dates:,} rows with unparseable dates")

    # Cap extreme outliers — same protective pattern as HW1
    cap_mask = df["late"] > 120
    if cap_mask.sum():
        med = df.loc[~cap_mask, "late"].median()
        df.loc[cap_mask, "late"] = med
        print(f"Capped {cap_mask.sum():,} extreme values (>120 min) with median ({med:.1f})")

    # Fill missing origin/dest with placeholder so encoding doesn't break
    missing_origin = df["origin"].isna().sum()
    missing_dest   = df["dest"].isna().sum()
    if missing_origin:
        df["origin"] = df["origin"].fillna("Unknown")
        print(f"Filled {missing_origin:,} missing origin values with 'Unknown'")
    if missing_dest:
        df["dest"] = df["dest"].fillna("Unknown")
        print(f"Filled {missing_dest:,} missing dest values with 'Unknown'")

    # Drop duplicates (same as HW1)
    df = df.drop_duplicates()
    print(f"Removed {original - len(df):,} total rows")
    print(f"Clean dataset: {len(df):,} rows remaining")
    return df


# =============================================================================
# 3. FEATURE ENGINEERING
# =============================================================================

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    month / day_of_week / day_of_year / hour
        Temporal features from date and timestamp.

    month_sin / month_cos
        Cyclical encoding — December and January treated as adjacent.
        sin(2*pi*month/12). Same technique from the project proposal.

    is_rush_hour
        1 if AM peak (7-9) or PM peak (16-19).
        Commute windows drive the most scheduling pressure on SEPTA rail.

    is_weekend
        1 if Saturday or Sunday. Weekend rail runs on reduced frequency
        and shows different delay patterns from weekday commuter service.

    direction_enc
        Binary encoding of Northbound (1) vs Southbound (0).
        Inbound morning / outbound evening trains behave differently.

    is_delayed  (binary classification target)
        1 if late >= 6 min, matching SEPTA's official OTP definition.
        Derived directly from the regression target.
    """
    print("\n=== 3. FEATURE ENGINEERING ===")

    # Temporal features from date
    df["month"]       = df["date"].dt.month
    df["day_of_week"] = df["date"].dt.dayofweek    # 0=Mon, 6=Sun
    df["day_of_year"] = df["date"].dt.dayofyear

    # Cyclical month encoding (same as project proposal)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    print("Created month_sin / month_cos (cyclical encoding)")
    print("  Reason: Treats Dec and Jan as adjacent, not opposite ends of scale")

    # Hour from timestamp
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
        df["hour"] = df["timestamp"].dt.hour.fillna(8).astype(int)
    else:
        df["hour"] = 8
    print("Created hour (extracted from timeStamp)")

    # Rush hour flag
    df["is_rush_hour"] = (
        ((df["hour"] >= 7)  & (df["hour"] <= 9)) |
        ((df["hour"] >= 16) & (df["hour"] <= 19))
    ).astype(int)
    print("Created is_rush_hour  (AM: 7-9, PM: 16-19)")
    print("  Reason: Peak windows concentrate scheduling pressure and delays")

    # Weekend flag
    df["is_weekend"] = (df["day_of_week"] >= 5).astype(int)
    print("Created is_weekend")
    print("  Reason: Weekend service differs structurally from weekday commuter")

    # Direction encoding (bonus feature from the actual dataset)
    if "direction" in df.columns:
        df["direction_enc"] = (
            df["direction"].str.lower().str.contains("north", na=False)
        ).astype(int)
        print("Created direction_enc (1=Northbound, 0=Southbound)")
        print("  Reason: Inbound/outbound trains have different loading patterns")

    # Binary classification target
    df["is_delayed"] = (df["late"] >= DELAY_THRESHOLD).astype(int)
    pct = df["is_delayed"].mean() * 100
    print(f"\nCreated is_delayed (threshold = {DELAY_THRESHOLD} min)")
    print(f"  {pct:.1f}% of trips delayed  |  {100-pct:.1f}% on time")

    return df


# =============================================================================
# 4. ENCODING & TRAIN/TEST SPLIT
# =============================================================================

def encode_and_split(df: pd.DataFrame):
    """
    Label-encode categoricals, build feature matrix,
    split 80/20 with random_state=42 (same as all HW2 assignments).
    Returns both unscaled (for RF) and scaled (for linear models).
    """
    print("\n=== 4. ENCODING & TRAIN/TEST SPLIT ===")

    le_origin = LabelEncoder()
    le_dest   = LabelEncoder()
    le_train  = LabelEncoder()

    df["origin_enc"] = le_origin.fit_transform(df["origin"].astype(str))
    df["dest_enc"]   = le_dest.fit_transform(df["dest"].astype(str))
    df["train_enc"]  = le_train.fit_transform(df["trainno"].astype(str))

    feature_cols = [
        "train_enc",       # which train number
        "origin_enc",      # departure station
        "dest_enc",        # next station / destination
        "month",           # raw month (1-12)
        "month_sin",       # cyclical encoding
        "month_cos",
        "day_of_week",     # 0=Mon ... 6=Sun
        "day_of_year",     # 1-365 (captures seasonal drift)
        "hour",            # time of day
        "is_rush_hour",    # peak flag
        "is_weekend",      # weekend flag
    ]

    # Add direction if it was created
    if "direction_enc" in df.columns:
        feature_cols.append("direction_enc")

    X     = df[feature_cols].copy()
    y_reg = df["late"]        # regression target: minutes late
    y_cls = df["is_delayed"]  # classification target: binary

    # 80/20 split — random_state=42 throughout (same as HW2)
    (X_train, X_test,
     y_reg_train, y_reg_test,
     y_cls_train, y_cls_test) = train_test_split(
        X, y_reg, y_cls,
        test_size=0.2,
        random_state=RANDOM_STATE
    )

    print(f"Training set : {len(X_train):,} rows")
    print(f"Test set     : {len(X_test):,} rows")
    print(f"Features     : {feature_cols}")

    # Scale for linear models (RF does not need scaling)
    print("\n=== 5. Z-SCORE STANDARDIZATION ===")
    scaler     = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)   # fit on train only — no leakage
    X_test_sc  = scaler.transform(X_test)
    print(f"Mean after scaling : {X_train_sc.mean():.2f}")
    print(f"Std  after scaling : {X_train_sc.std():.2f}")
    print("Advantage: All features on same scale (mean=0, std=1)")
    print("  Required for Linear/Logistic Regression")
    print("  Random Forest is scale-invariant — unscaled features used")

    return (X_train, X_test, X_train_sc, X_test_sc,
            y_reg_train, y_reg_test,
            y_cls_train, y_cls_test,
            scaler, feature_cols)


# =============================================================================
# 5. MODEL TRAINING
# =============================================================================

def train_all_models(X_train, X_train_sc,
                     y_reg_train, y_cls_train,
                     tune_rf: bool = False):
    """
    Three models in order of complexity:

    1. Linear Regression       — baseline (scaled)
    2. Random Forest Regressor — primary  (unscaled, handles non-linearity)
    3. Logistic Regression     — classification comparison (scaled)
       Same as HW2 diabetes_classifier: max_iter=500, random_state=42
    """
    print("\n=== 6. MODEL TRAINING ===")

    print("  Training Linear Regression (baseline)...")
    lr_base = LinearRegression()
    lr_base.fit(X_train_sc, y_reg_train)

    if tune_rf:
        print("  Tuning Random Forest with GridSearchCV (extra credit)...")
        param_grid = {
            "n_estimators":      [100, 200],
            "max_depth":         [None, 10, 20],
            "min_samples_split": [2, 5],
        }
        gs = GridSearchCV(
            RandomForestRegressor(random_state=RANDOM_STATE),
            param_grid, cv=5, scoring="neg_mean_squared_error",
            n_jobs=-1, verbose=1
        )
        gs.fit(X_train, y_reg_train)
        rf_model = gs.best_estimator_
        print(f"  Best RF params: {gs.best_params_}")
    else:
        print("  Training Random Forest Regressor (primary)...")
        rf_model = RandomForestRegressor(n_estimators=100, random_state=RANDOM_STATE)
        rf_model.fit(X_train, y_reg_train)

    print("  Training Logistic Regression (classification)...")
    log_reg = LogisticRegression(max_iter=500, random_state=RANDOM_STATE)
    log_reg.fit(X_train_sc, y_cls_train)

    print("  All models trained.")
    return lr_base, rf_model, log_reg


# =============================================================================
# 6. EVALUATION
# =============================================================================

def evaluate_regression(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    mae  = mean_absolute_error(y_test, y_pred)
    r2   = r2_score(y_test, y_pred)
    print(f"\n  {model_name}")
    print(f"    RMSE : {rmse:.4f} min")
    print(f"    MAE  : {mae:.4f} min")
    print(f"    R2   : {r2:.4f}")
    return {"model": model_name, "RMSE": rmse, "MAE": mae, "R2": r2}


def evaluate_classification(model_or_preds, X_test, y_test, model_name: str) -> dict:
    # Accepts either a fitted model or a pre-computed predictions array
    if isinstance(model_or_preds, np.ndarray):
        y_pred = model_or_preds
    else:
        y_pred = model_or_preds.predict(X_test)
    acc  = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, zero_division=0)
    rec  = recall_score(y_test, y_pred, zero_division=0)
    f1   = f1_score(y_test, y_pred, zero_division=0)
    cm   = confusion_matrix(y_test, y_pred)
    print(f"\n  {model_name}")
    print(f"    Accuracy  : {acc:.4f}")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")
    print(f"    Confusion Matrix:\n{cm}")
    return {"model": model_name, "Accuracy": acc, "Precision": prec,
            "Recall": rec, "F1": f1}


# =============================================================================
# 7. VISUALIZATIONS
# =============================================================================

def make_visualizations(df: pd.DataFrame, rf_model, feature_cols: list) -> None:
    """Generate and save three plots for the report."""

    # Plot 1 — Feature importance
    importances = pd.Series(rf_model.feature_importances_, index=feature_cols)
    importances = importances.sort_values()
    fig, ax = plt.subplots(figsize=(8, 5))
    importances.plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Importance")
    ax.set_title("Random Forest Feature Importances — SEPTA OTP")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()
    print("Saved: feature_importance.png")

    # Plot 2 — Avg delay by day of week
    days = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    day_avg = df.groupby("day_of_week")["late"].mean()
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar([days[int(d)] for d in day_avg.index], day_avg.values, color="coral")
    ax.set_ylabel("Avg Minutes Late")
    ax.set_title("Average Delay by Day of Week — SEPTA Regional Rail")
    plt.tight_layout()
    plt.savefig("delay_by_day.png", dpi=150)
    plt.close()
    print("Saved: delay_by_day.png")

    # Plot 3 — Top 10 most delayed destinations
    top_lines = df.groupby("dest")["late"].mean().nlargest(10)
    fig, ax = plt.subplots(figsize=(8, 5))
    top_lines.sort_values().plot(kind="barh", ax=ax, color="steelblue")
    ax.set_xlabel("Avg Minutes Late")
    ax.set_title("Top 10 Most Delayed Next Stations — SEPTA Regional Rail")
    plt.tight_layout()
    plt.savefig("delay_by_station.png", dpi=150)
    plt.close()
    print("Saved: delay_by_station.png")


# =============================================================================
# 8. RESEARCH QUESTION ANSWERS
# =============================================================================

def answer_research_questions(df: pd.DataFrame, reg_results: list) -> None:
    print("\n=== 9. RESEARCH QUESTIONS ===")

    rf_r2   = next(r["R2"]   for r in reg_results if "Random Forest" in r["model"])
    base_r2 = next(r["R2"]   for r in reg_results if "Linear"        in r["model"])
    rf_rmse = next(r["RMSE"] for r in reg_results if "Random Forest" in r["model"])

    print(f"\nQ1 — Can we predict minutes late from route and time features?")
    print(f"  Random Forest R2  = {rf_r2:.4f}")
    print(f"  Linear Baseline   = {base_r2:.4f}")
    print(f"  RF RMSE           = {rf_rmse:.2f} min avg prediction error")
    if rf_r2 > base_r2:
        print("  -> RF outperforms baseline. Non-linear patterns exist in delay data.")
    else:
        print("  -> Models perform similarly. Relationship may be largely linear.")

    print(f"\nQ2 — Which routes and times are most delay-prone?")
    worst = df.groupby("dest")["late"].mean().nlargest(5)
    print("  Most delayed next stations (avg min late):")
    for station, avg in worst.items():
        print(f"    {station}: {avg:.1f} min")
    days = ["Mon","Tue","Wed","Thu","Fri","Sat","Sun"]
    worst_day = df.groupby("day_of_week")["late"].mean().idxmax()
    best_day  = df.groupby("day_of_week")["late"].mean().idxmin()
    print(f"  Worst day: {days[int(worst_day)]}  |  Best day: {days[int(best_day)]}")

    print(f"\nQ3 — Does rush hour significantly worsen delays?")
    rush    = df[df["is_rush_hour"] == 1]["late"].mean()
    offpeak = df[df["is_rush_hour"] == 0]["late"].mean()
    print(f"  Rush hour avg  : {rush:.2f} min late")
    print(f"  Off-peak avg   : {offpeak:.2f} min late")
    print(f"  Difference     : {rush - offpeak:+.2f} min")

    if "direction_enc" in df.columns:
        print(f"\nBonus — Direction effect:")
        dir_df = df.dropna(subset=["direction_enc"])
        nb = dir_df[dir_df["direction_enc"] == 1]["late"].mean()
        sb = dir_df[dir_df["direction_enc"] == 0]["late"].mean()
        print(f"  Northbound avg : {nb:.2f} min late")
        print(f"  Southbound avg : {sb:.2f} min late")


# =============================================================================
# 9. SUMMARY TABLE
# =============================================================================

def print_summary_table(reg_results: list, cls_results: list) -> None:
    print("\n=== 10. MODEL COMPARISON SUMMARY ===")
    print(f"\n  {'Model':<42} {'RMSE':>8} {'MAE':>8} {'R2':>8}")
    print("  " + "-" * 68)
    for r in reg_results:
        print(f"  {r['model']:<42} {r['RMSE']:>8.4f} {r['MAE']:>8.4f} {r['R2']:>8.4f}")

    print(f"\n  {'Model':<42} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8}")
    print("  " + "-" * 78)
    for r in cls_results:
        print(f"  {r['model']:<42} {r['Accuracy']:>8.4f} "
              f"{r['Precision']:>8.4f} {r['Recall']:>8.4f} {r['F1']:>8.4f}")


# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    # Set TUNE_RF = True for GridSearchCV hyperparameter tuning (extra credit)
    # Warning: slower on large datasets — keep False for a quick run
    TUNE_RF = False

    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    (X_train, X_test, X_train_sc, X_test_sc,
     y_reg_train, y_reg_test,
     y_cls_train, y_cls_test,
     scaler, feature_cols) = encode_and_split(df)

    lr_base, rf_model, log_reg = train_all_models(
        X_train, X_train_sc, y_reg_train, y_cls_train, tune_rf=TUNE_RF
    )

    print("\n=== 7. EVALUATION RESULTS ===")
    reg_results = []
    reg_results.append(evaluate_regression(
        lr_base,  X_test_sc, y_reg_test, "Linear Regression (Baseline)"))
    reg_results.append(evaluate_regression(
        rf_model, X_test,    y_reg_test, "Random Forest Regressor (Primary)"))

    cls_results = []
    cls_results.append(evaluate_classification(
        log_reg, X_test_sc, y_cls_test, "Logistic Regression"))

    rf_cls_preds = (rf_model.predict(X_test) >= DELAY_THRESHOLD).astype(int)
    cls_results.append(evaluate_classification(
        rf_cls_preds, X_test, y_cls_test,
        f"Random Forest (threshold >= {DELAY_THRESHOLD} min)"))

    print("\n=== 8. VISUALIZATIONS ===")
    make_visualizations(df, rf_model, feature_cols)

    answer_research_questions(df, reg_results)
    print_summary_table(reg_results, cls_results)

    print("\n=== DONE ===")
    print("Metrics above are ready to copy into your report.")
    print("Plots saved: feature_importance.png, delay_by_day.png, delay_by_station.png")


if __name__ == "__main__":
    main()