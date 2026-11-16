**2. Quickest check: print the headline.** From the repo folder:

```bash
python headline.py
```

This runs only the steps the headline depends on, with the same code, settings and seed as the notebook, and prints the headline paragraph above. It takes about 1–2 minutes.

# ER triage acuity model

CSCI 5622 project. Using only information available at sign-in, a logistic regression ranks emergency department walk-ins so staff can pull likely-critical patients (ESI 1–2) forward in the triage line. Data: NHAMCS emergency department visits, 2018–2022. The model trains on 2018–2021, and 2022 is the held-out test set.

## Headline result

Before triage, using only information available at sign-in, logistic regression caught 57.8% of critical (ESI 1–2) walk-ins [95% CI: 54.8–60.7%] (n = 1,076 critical) on the held-out 2022 data, at an FPR of 0.164 [0.156–0.173] (n = 7,052 non-critical). The baseline rule of prioritizing patients 65+ caught 26.3% [23.8–29.0%] at an FPR of 0.147 [0.139–0.156]. The gain was +31.5 percentage points (paired bootstrap 95% CI: +27.9 to +35.2). The model struggled most on children aged 0–17, catching only 33.1% [26.5–40.5%] of critical pediatric walk-ins (n = 1,881 visits, 169 critical). If the ED added a quick vital-signs check at registration, the main weakness of this model would be solvable with better data, and the recommendation would change to a model retrained with vitals.

## Contents

| File | What it is |
|---|---|
| `er-triage-acuity.ipynb` | The full analysis: cleaning, baseline, both models, 2022 test, failure analysis, clustering |
| `headline.py` | A standalone script that reproduces and prints the headline paragraph |
| `nhamcs_data_2018_22.csv` | The data: 58,124 NHAMCS ED visits, 2018–2022. Included in the repo. If the notebook can't read the local copy, it downloads the same file from this repo on GitHub |

## Reproduce the headline number

**1. Install the dependencies.** These are the versions the results were checked with (Python 3.13.7):

```bash
pip install numpy==2.5.4 pandas==3.1.1 scikit-learn==1.9.1 scipy matplotlib jupyter
```

**2. Quickest check: print the headline.** From the repo folder:

```bash
python headline.py
```

This runs only the steps the headline depends on, with the same code, settings and seed as the notebook, and prints the headline paragraph above. It takes about 1–2 minutes.

**Or run the full notebook from the repo folder:**

```bash
jupyter nbconvert --to notebook --execute er-triage-acuity.ipynb --output er-triage-acuity-run.ipynb
```

You can also open the notebook in Jupyter or VS Code and use **Restart & Run All**. With the default settings, a full run takes about 3 minutes on a 24-core machine.

**3. Find the numbers.** The last cell, **Headline**, prints the full headline paragraph above. The underlying table is under **"FINAL EVAL ON TEST 2022"**, in the *2022 walk-ins (held-out test set)* table:

| Rule | Recall | Recall 95% CI | FPR |
|---|---|---|---|
| Baseline: age ≥ 65 | 0.263 | 0.238–0.290 | 0.147 |
| Logistic regression | 0.578 | 0.548–0.607 | 0.164 |
| Gradient boosting | 0.564 | 0.534–0.593 | 0.162 |

The seed is fixed (`SEED = 42`), so a rerun gives the same numbers.

## Fast run vs. full tuning: `RUN_GRID_SEARCH`

The settings cell near the top of the notebook has one flag:

```python
RUN_GRID_SEARCH = False
```

- **`False` (default, fast):** skips the hyperparameter grid searches. The notebook fits the best settings those searches found (logistic regression: `C = 1`, L2 penalty; gradient boosting: learning rate 0.05, 31 leaf nodes, 100 samples per leaf). This is the same model the search refits, so every reported number is unchanged. The logistic regression tuning-curve plot is skipped.
- **`True` (slow):** reruns both grid searches (27 logistic regression settings and 8 gradient boosting settings, each over 5 folds) and draws the tuning curve. The searches use all CPU cores (`n_jobs=-1`).

The 2022 test set is never used to tune anything. Hyperparameters come from 5-fold cross-validation on 2018–2021, and each model's decision threshold comes from its out-of-fold predictions on 2018–2021.
