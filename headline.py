"""Reproduce the project headline.

Runs only the steps the headline depends on, with the same code, settings and seed as
er-triage-acuity.ipynb, then prints and returns the headline paragraph.

Usage (from the repo folder):
    python headline.py
"""
import sys

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
TARGET_FPR = 0.15
DATA_FILE = "./nhamcs_data_2018_22.csv"
DATA_URL = "https://raw.githubusercontent.com/sabrinatouch/ER-triage-acuity-model/main/nhamcs_data_2018_22.csv"
TARGET_COL = "target_triage_acuity"
TEXT = ["chief_complaint_text", "injury_cause_text"]
CATEGORICAL = ["residence", "sex", "insurance", "region", "episode", "is_injury_poison", "race"]


def log(msg):
    """Progress messages go to stderr, so stdout holds only the headline."""
    print(msg, file=sys.stderr, flush=True)


def wilson_ci(successes, n, conf=0.95):
    """Wilson score interval for a proportion successes/n."""
    z = norm.ppf(1 - (1 - conf) / 2)
    p = successes / n
    denom = 1 + z**2 / n
    center = (p + z**2 / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / denom
    return center - half, center + half


def load_data():
    """Local CSV first; if it can't be read, download the same file from the GitHub repo."""
    try:
        data = pd.read_csv(DATA_FILE)
    except (OSError, pd.errors.ParserError, pd.errors.EmptyDataError):
        log(f"Could not read {DATA_FILE} locally; downloading from GitHub...")
        data = pd.read_csv(DATA_URL)
    assert len(data) == 58_124, f"Expected 58,124 rows, got {len(data):,}"
    return data


def clean_and_encode(data):
    """Pre-triage walk-ins only; fixed row-wise cleaning and one-hot encoding (same as the notebook)."""
    triage_measured = ["temp", "heart_rate", "resp_rate", "sys_bp", "dias_bp", "spo2", "pain_score", "vitals_during_visit"]
    post_triage = ["wait_time_minutes", "intervention_iv_fluids"]
    df = data.drop(columns=triage_measured + post_triage)
    df = df[df["ems_arrival"] == "No"].copy().drop(columns=["ems_arrival"])

    placeholders = {"Blank": "Unknown", "Unknown/Blank": "Unknown", "Unknown_Blank": "Unknown", "Blank/Unknown": "Unknown"}
    for col in CATEGORICAL:
        df[col] = df[col].fillna("Unknown").replace(placeholders)
    df["insurance"] = df["insurance"].replace({"Workers_Comp": "Other", "Charity": "Other"})
    df["seen_last_72h"] = df["seen_last_72h"].map({1: 1, 2: 0})
    df["arrival_hour"] = df["arrival_time"] // 100
    df = df.drop(columns=["arrival_time"])
    for col in TEXT:
        df[col] = df[col].fillna("").replace({"Blank/Unknown": "", "Blank": "", "Unknown": ""}).str.lower()

    encoded = pd.get_dummies(df, columns=CATEGORICAL, drop_first=True, dtype=int)
    return df, encoded


def make_split(df, years):
    part = df[df["year"].isin(years)]
    X = part.drop(columns=[TARGET_COL, "year"])
    y_esi = part[TARGET_COL]
    y_binary = y_esi.isin([1, 2]).astype(int)
    return X, y_binary, y_esi


def main():
    log("Loading and cleaning data...")
    data_clean, data_encoded = clean_and_encode(load_data())
    X_train, y_train_binary, y_train = make_split(data_encoded, [2018, 2019, 2020, 2021])
    X_test, y_test_binary, _ = make_split(data_encoded, [2022])

    # Logistic regression with the best settings from the grid search (C = 1, L2)
    numeric = [c for c in X_train.columns if c not in TEXT]
    preprocess = ColumnTransformer([
        ("num", Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                          ("scale", StandardScaler())]), numeric),
        ("cc", TfidfVectorizer(ngram_range=(1, 2), min_df=20, stop_words="english"), "chief_complaint_text"),
        ("inj", TfidfVectorizer(ngram_range=(1, 2), min_df=20, stop_words="english"), "injury_cause_text"),
    ])
    lr_model = Pipeline([
        ("prep", preprocess),
        ("model", LogisticRegression(solver="saga", C=1.0, l1_ratio=0, max_iter=5000, random_state=SEED)),
    ])
    log("Fitting logistic regression on 2018-2021...")
    lr_model.fit(X_train, y_train_binary)

    # Threshold: lowest cutoff with FPR <= 15% on 2018-2021 out-of-fold predictions
    log("Choosing the threshold from out-of-fold predictions...")
    cv_folds = list(StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED).split(X_train, y_train))
    oof_probs = cross_val_predict(lr_model, X_train, y_train_binary, cv=cv_folds,
                                  method="predict_proba", n_jobs=-1)[:, 1]
    fpr, _, thresholds = roc_curve(y_train_binary, oof_probs)
    threshold = thresholds[np.where(fpr <= TARGET_FPR)[0][-1]]

    # 2022 test predictions: the model and the age >= 65 baseline
    lr_pred = (lr_model.predict_proba(X_test)[:, 1] >= threshold).astype(int)
    age_pred = (X_test["age"] >= 65).astype(int).to_numpy()
    y = y_test_binary.to_numpy()
    crit = y == 1
    n_crit, n_non = crit.sum(), (~crit).sum()
    lr_tp, lr_fp = lr_pred[crit].sum(), lr_pred[~crit].sum()
    age_tp, age_fp = age_pred[crit].sum(), age_pred[~crit].sum()

    lr_r_lo, lr_r_hi = wilson_ci(lr_tp, n_crit)
    lr_f_lo, lr_f_hi = wilson_ci(lr_fp, n_non)
    age_r_lo, age_r_hi = wilson_ci(age_tp, n_crit)
    age_f_lo, age_f_hi = wilson_ci(age_fp, n_non)

    # Paired bootstrap of the recall gain (2,000 resamples of 2022 patients)
    rng = np.random.default_rng(SEED)
    n, B = len(y), 2000
    gains = []
    for _ in range(B):
        idx = rng.integers(0, n, n)
        yb = y[idx]
        gains.append((lr_pred[idx] & yb).sum() / yb.sum() - (age_pred[idx] & yb).sum() / yb.sum())
    gain_lo, gain_hi = np.percentile(gains, [2.5, 97.5])

    # Worst slice: lowest recall across age groups and insurance types (>= 30 critical patients)
    slices = pd.DataFrame({
        "Age": pd.cut(X_test["age"], bins=[-1, 17, 64, 200], labels=["0–17", "18–64", "65+"]).astype(str),
        "Insurance": data_clean.loc[X_test.index, "insurance"],
        "critical": y,
        "flagged": lr_pred,
    }, index=X_test.index)
    rows = []
    for var in ["Age", "Insurance"]:
        for value, g in slices.groupby(var):
            crit_g = g[g["critical"] == 1]
            if len(crit_g) >= 30:
                rows.append({"slice": f"{var} {value}", "n visits": len(g), "n critical": len(crit_g),
                             "caught": int(crit_g["flagged"].sum())})
    worst = min(rows, key=lambda r: r["caught"] / r["n critical"])
    assert worst["slice"] == "Age 0–17", f"Worst slice changed to {worst['slice']}; update the headline text"
    w_rec = worst["caught"] / worst["n critical"]
    w_lo, w_hi = wilson_ci(worst["caught"], worst["n critical"])

    return (
        f"Before triage, using only information available at sign-in, logistic regression caught "
        f"{lr_tp / n_crit:.1%} of critical (ESI 1–2) walk-ins [95% CI: {lr_r_lo * 100:.1f}–{lr_r_hi:.1%}] "
        f"(n = {n_crit:,} critical) on the held-out 2022 data, at an FPR of {lr_fp / n_non:.3f} "
        f"[{lr_f_lo:.3f}–{lr_f_hi:.3f}] (n = {n_non:,} non-critical). "
        f"The baseline rule of prioritizing patients 65+ caught {age_tp / n_crit:.1%} "
        f"[{age_r_lo * 100:.1f}–{age_r_hi:.1%}] at an FPR of {age_fp / n_non:.3f} [{age_f_lo:.3f}–{age_f_hi:.3f}]. "
        f"The gain was {(lr_tp - age_tp) / n_crit * 100:+.1f} percentage points "
        f"(paired bootstrap 95% CI: {gain_lo * 100:+.1f} to {gain_hi * 100:+.1f}). "
        f"The model struggled most on children aged 0–17, catching only {w_rec:.1%} "
        f"[{w_lo * 100:.1f}–{w_hi:.1%}] of critical pediatric walk-ins "
        f"(n = {worst['n visits']:,} visits, {worst['n critical']:,} critical). "
        f"If the ED added a quick vital-signs check at registration, the main weakness of this model would be "
        f"solvable with better data, and the recommendation would change to a model retrained with vitals."
    )


if __name__ == "__main__":
    print(main())
