# ER triage acuity model

CSCI 5622 project. Using only information available at sign-in, a logistic regression ranks emergency department walk-ins so staff can pull likely-critical patients (ESI 1–2) forward in the triage line. Data: NHAMCS emergency department visits, 2018–2022. The model trains on 2018–2021, and 2022 is the held-out test set.

## Headline result

On 2022 walk-ins, logistic regression caught **57.8% of critical patients (95% CI 54.8–60.7%)** at a false-positive rate of 0.164. The baseline rule "prioritize patients aged 65+" caught **26.3% (95% CI 23.8–29.0%)** at a false-positive rate of 0.147.

## Contents

| File | What it is |
|---|---|
| `er-triage-acuity.ipynb` | The full analysis: cleaning, baseline, both models, 2022 test, failure analysis, clustering |
| `nhamcs_data_2018_22.csv` | The data: 58,124 NHAMCS ED visits, 2018–2022. Included in the repo. If the notebook can't read the local copy, it downloads the same file from this repo on GitHub |

## Reproduce the headline number

**1. Install the dependencies.** These are the versions the results were checked with (Python 3.13.7):

```bash
pip install numpy==2.5.4 pandas==3.1.1 scikit-learn==1.9.1 scipy matplotlib jinja2 jupyter
```

**2. Run the notebook from the repo folder:**

```bash
jupyter nbconvert --to notebook --execute er-triage-acuity.ipynb --output er-triage-acuity-run.ipynb
```

You can also open the notebook in Jupyter or VS Code and use **Restart & Run All**. With the default settings, a full run takes about 3 minutes on a 24-core machine.

**3. Find the numbers.** The headline is in the table under **"FINAL EVAL ON TEST 2022"**, in the *2022 walk-ins (held-out test set)* table:

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
