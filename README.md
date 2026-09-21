![CI](https://github.com/LuisMckellen/ml--journey/actions/workflows/ci.yml/badge.svg)

ML work as a first-year CSE student. One project taken properly from raw data to a deployed link 

## Water Potability

Predicting whether water is safe to drink from nine chemical measurements.

**Live API:** https://ml-journey-mpqh.onrender.com

The short version of what's below: the headline metric moves by ±0.017 depending on which random split you take, so it is reported as a distribution rather than a single number. Getting to that point required retracting two earlier claims in this README, both of which are kept here with their corrections attached.

## Dataset

- 3,276 samples, Kaggle, 9 chemical measurements: `ph`, `Hardness`, `Solids`, `Chloramines`, `Sulfate`, `Conductivity`, `Organic_carbon`, `Trihalomethanes`, `Turbidity`
- Target: `Potability` (0 = not potable, 1 = potable), roughly 61/39 class split
- Missing values in `Sulfate` (781 rows), `ph` (491), `Trihalomethanes` (162) — median-imputed. The other six columns have none.
- `ph` runs exactly 0.00–14.00; `Solids` reaches ~61,000 while `Turbidity` stays under 7
- Input validation: non-negative, upper bounds at dataset max +20%, `ph` clamped 0–14, non-finite values rejected

## Choosing the metric

Accuracy is wrong here — 61% of samples are "not potable," so a model predicting that label every time scores 61% and has learned nothing. Recall alone is gameable too: a dummy predicting "not potable" always gets recall₀ = 1.0000 while never identifying a single safe sample. Not hypothetical — logistic regression did exactly this in early comparisons:

| Model | macro F1 | recall₀ | recall₁ |
|---|---|---|---|
| LogisticRegression | 0.3788 | 1.0000 | 0.0000 |
| RandomForest | 0.5887 | 0.8925 | 0.3047 |
| XGBoost | 0.6015 | 0.8050 | 0.3984 |

*Single split, notebook environment. Given the instability documented below, the gap between RandomForest and XGBoost here is not meaningful — treat this table as evidence that LogisticRegression collapses, nothing finer.*

Accuracy would have called the LogisticRegression model 61% correct. Macro F1 scores it 0.3788, because it can't score well without doing something useful on both classes — that's why macro F1 is the metric used throughout, not accuracy and not recall₀ alone.

## Results

Production model, `src/train.py`, xgboost 3.4.1, hyperparameters pinned (`n_estimators=100`, `max_depth=6`, `learning_rate=0.3`), split seed 42.

| Metric | Score |
|---|---|
| Macro F1 | 0.5613 |
| Recall (not potable) | 0.8150 |
| Recall (potable) | 0.3203 |
| Accuracy | 0.6220 |

**That number is a pessimistic draw, and it is reported anyway.** Across split seeds 0–9, clean macro F1 averages **0.6068 ± 0.0168**, ranging 0.5826 to 0.6287. The deployed model's 0.5613 falls below all ten — roughly 2.7 standard deviations under the mean. Seed 42 produced an unusually hard test set.

The deployed model was not retrained on a friendlier seed. Picking the seed that flatters the model is selecting on the test set, which is the same error as leakage wearing a different hat. The honest summary is that this model scores **macro F1 ≈ 0.61 ± 0.02** on this data, and the particular artifact being served happens to land at the low end of that range.

The number with real-world cost is recall₀ = 0.8150: the model catches 82% of unsafe samples, and the 18% it misses are unsafe water called safe. Flagging safe water as unsafe only wastes it. That is also exactly why the *metric* isn't recall₀ — optimising for it has a degenerate solution, as the table above shows.

**5-fold stratified CV** (`src/validate.py`, on the 2,620-row train split, medians recomputed inside each fold):

| Metric | Mean ± std |
|---|---|
| Macro F1 | 0.5957 ± 0.0060 |
| Recall (not potable) | 0.7804 ± 0.0291 |
| Recall (potable) | 0.4110 ± 0.0271 |

Per-fold macro F1: 0.6054, 0.5917, 0.5900, 0.5961, 0.5954.

Note that CV's ± 0.0060 badly understates real uncertainty. All five folds draw from the same 2,620-row pool and overlap heavily in training data, so their agreement is partly an artifact of sharing. Resampling the split entirely gives ± 0.0168 — nearly three times wider. **k-fold spread answers "how consistent is this across folds of one sample," not "how would this do on a different sample."** The CV mean of 0.5957 and the seed-sweep mean of 0.6068 agree well; it was seed 42's test split, not the CV, that was the outlier.

## Stability

`src/stability.py`, 30 training runs.

| Source of variation | Spread in macro F1 |
|---|---|
| Train/test split seed | 0.0168 (range 0.0460) |
| Model seed, split fixed | **0.0000** |

XGBoost is fully deterministic here given fixed data and pinned hyperparameters — `random_state` on the model changes nothing. All variance is data-side.

The sharpest result: the leaky-vs-clean *difference*, measured on identical splits, has a spread of 0.0270 — wider than the split-driven spread itself. The only thing separating those two pipelines is ~490 `ph` values shifted by 0.0017 and ~160 `Trihalomethanes` values shifted by 0.057 (see below). A perturbation that small, in two columns, moving the metric that much means the model sits on a knife edge: small input changes flip enough tree split thresholds to cascade through the predictions.

That is consistent with everything else known about this dataset — no feature correlates with the target above 0.05, and a shuffled-label control showed only weak signal.

## The leakage experiment, and a retraction

The first working version computed medians on the full dataframe and then split, leaking the test distribution into training. Found by reading the code, not by a failing test. The fix is an ordering change:

```python
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
medians = X_train.median()      # train only
X_train = X_train.fillna(medians)
X_test = X_test.fillna(medians)
```

At seed 42, clean scored 0.5613 and leaky scored 0.6126 — a gap of +0.0513, which this README previously reported as the cost of the leak.

**That was wrong.** Run across ten split seeds, the leaky-minus-clean difference is:

| | mean | std | min | max |
|---|---|---|---|---|
| leaky − clean | **−0.0092** | 0.0270 | −0.0555 | +0.0250 |

The effect straddles zero and is slightly negative on average. The +0.0513 at seed 42 was split noise, not a leakage advantage.

Diffing the median vectors directly shows why it could never have been large:

| Column | full data | train only | diff | n missing |
|---|---|---|---|---|
| Sulfate | 333.073546 | 333.073546 | **0.000000** | 781 |
| ph | 7.036752 | 7.035037 | 0.001715 | 491 |
| Trihalomethanes | 66.622485 | 66.565709 | 0.056776 | 162 |

`Sulfate`, the column with the most missing values, has an identical median either way. The other two differ by 0.02% and 0.085%. The leaky and clean pipelines train on very nearly the same data, so there was almost no test-set information available to transfer.

**The leak is still worth fixing, but for a different reason than the obvious one.** Leaky runs have a spread of 0.0265 against clean's 0.0168. Leakage here didn't inflate the score — it inflated the variance. A leaky pipeline returns a number that is unreliable in both directions, which is worse than one that is reliably optimistic.

## The version-drift claim, also retracted

An earlier revision attributed a gap of 0.6015 (Kaggle, xgboost 3.2.0) versus 0.5613 (local, xgboost 3.4.1) to the xgboost minor version. Given that split noise alone spans 0.0460, a 0.040 gap between two runs on different machines is not evidence of a version effect. The claim is withdrawn; the underlying comparison was never controlled.

Version pinning stays, on its own merits:

- `xgboost==3.4.1` pinned exactly in `requirements.txt`. Everything else is `>=` — a floor, not a pin. Accepted trade-off, not an oversight.
- Hyperparameters set explicitly rather than inherited, so a future version bump can't silently change the model. Pinning them reproduces 0.5613 exactly, confirming they match 3.4.1's current defaults.
- Model saved with `save_model`, not pickled — a pickled booster is tied to the version that wrote it.
- `/ping` reports the running xgboost version.
- Metrics are reported with the version that produced them.

Both retractions came from the same mistake: a plausible explanation, never tested, that survived several revisions of this README because it sounded right. The controlled runs took under an hour between them.

## Imputed fields

The API accepts partial input — you don't need to send all nine fields. Missing fields are filled with the training-set medians (computed after the split, never before), and the response includes an `imputed_fields` list naming exactly which were filled. Silent imputation hides how much of a prediction is inferred versus measured, which matters for something safety-adjacent.

## Calling `/predict`

```bash
curl -X POST https://ml-journey-mpqh.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{
    "ph": 7.0,
    "Hardness": 200,
    "Solids": 15000,
    "Chloramines": 7,
    "Sulfate": 300,
    "Conductivity": 400,
    "Organic_carbon": 10,
    "Trihalomethanes": 60,
    "Turbidity": 4
  }'
```

Response (all nine fields — verified identical across local, Docker, and Render):

```json
{
  "potability": 0,
  "probability_potable": 0.1700534224510193,
  "imputed_fields": []
}
```

Partial input works too:

```bash
curl -X POST https://ml-journey-mpqh.onrender.com/predict \
  -H "Content-Type: application/json" \
  -d '{"ph": 7.0, "Hardness": 200, "Solids": 15000}'
```

```json
{
  "potability": 0,
  "probability_potable": 0.2627227306365967,
  "imputed_fields": [
    "Chloramines", "Sulfate", "Conductivity",
    "Organic_carbon", "Trihalomethanes", "Turbidity"
  ]
}
```

Six fields omitted, and the probability moves from 0.170 to 0.263 — same class, materially different number. Six of nine inputs are now population medians rather than measurements, and `imputed_fields` is what tells a caller that.

`potability` is the thresholded class (`probability_potable > 0.5`); `probability_potable` is the raw model output, not a calibrated probability — it orders samples correctly but isn't a true likelihood.

A request with every field `None` is rejected with 422. If model artifacts fail to load, predictions return 503 rather than serving a default guess.

`/ping` returns service health plus the running XGBoost version.

## Docker

```bash
docker build -t water-potability-api .
docker run -p 8000:8000 water-potability-api
curl http://localhost:8000/ping
```

Image ~472MB content / 1.4GB on disk. Respects `$PORT` (defaults to 8000). Predictions verified byte-identical across local `uvicorn`, Docker, and Render.

## CI

GitHub Actions on every push to `main` and every PR: checkout → Python 3.14.7 → install pinned dependencies → 14 tests → Docker build. A red run blocks the merge — verified on a real test PR before trusting the badge.

## Running locally

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
pytest
uvicorn app.main:api --reload
```

Model artifacts (`models/model.json`, `features.pkl`, `medians.pkl`) are committed — clone and run, no training step required. Retrain: `python -m src.train`. Reproduce the CV numbers: `python -m src.validate`. Reproduce the stability numbers: `python -m src.stability`.

## Stack

Python 3.14.7, pandas, scikit-learn, XGBoost 3.4.1, FastAPI, Docker, GitHub Actions, deployed on Render.

## Other gotchas

**The dtype bug** — an omitted optional field made pandas infer `object` dtype for a column XGBoost expects as numeric. Fixed with explicit `.astype(float)` after `.fillna(medians)`, applied in training and inference both so the two can't diverge. The value was right; the type wasn't, and the error surfaced three layers down in XGBoost rather than where the mistake was.

**A second leak, inside the cross-validation** — the first CV implementation imputed once on the full train split, then folded, so each fold's validation slice contributed to the medians filling its own training rows. Same bug as the original, one level down. Fixing it barely moved the mean (0.5983 → 0.5957) but halved the spread (± 0.0121 → ± 0.0060).

## Still open

- `scale_pos_weight` / class-weighting — and given a split-driven spread of 0.0168, any tuning result under ~0.03 is not distinguishable from noise on a single split. Evaluate across seeds or not at all.
- SHAP feature importance.
- Rerun the baseline comparison table across seeds, so the model ordering rests on distributions rather than one draw.
- `src/train.py`, `src/validate.py`, and `src/stability.py` duplicate the split logic — shared setup should move to a module all three import.
