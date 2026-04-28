# librec

## Getting Started

This project uses [uv](https://docs.astral.sh/uv/getting-started/installation/) for package management. Install it, then run:

```bash
uv sync   # install dependencies
./run.sh  # run the project
```

## Pipeline

`./run.sh` runs `src/main.py`, which executes these stages in order. Each stage
writes an artifact into `data/` and is skipped on subsequent runs if its
artifact already exists.

| Stage | Module | Artifact |
| --- | --- | --- |
| `download` | `src/download.py` | `data/raw/{lthing,epinions}_data/` |
| `exploration` | `src/exploration.py` | `data/{lthing,epinions}_stats.txt`, `_ratings.png` |
| `split` | `src/split.py` | `data/{lthing,epinions}_{train,val,test}.parquet` |
| `global_mean` | `src/global_mean.py` | `data/{lthing,epinions}_global_mean.txt` |
| `baseline` | `src/baseline.py` | `data/{lthing,epinions}_baseline.txt` |
| `lightgbm` | `src/lightgbm_model.py` | `data/evals/{lthing,epinions}_lightgbm.csv` |

### Split

Temporal 80 / 10 / 10 split on the `time` column. Each parquet row is
`(user, item, stars, time)` — that's the shared schema individual variants
should consume.

### Global-mean baseline

A trivial floor: predict the train-set mean `stars` for every (user, item)
and report RMSE on val and test. Useful as a reference point that any real
model should beat.

### LightGBM baseline

A LightGBM regressor is trained per dataset to predict `stars` from
`user_code` and `item_code` (user/item ids encoded as categorical features
using the train-set vocabulary; unseen ids in val/test get code `-1`).
Training uses early stopping on validation RMSE, and the final RMSE is
reported on both the validation and test splits.

### Engineered LightGBM

The `lightgbm` model keeps the same train/validation/test split but trains one
LightGBM regressor on a richer feature matrix. These features include user and
item rating counts, means, biases, dispersion, rating ranges,
popularity interactions, cold-start flags, and social graph neighbor rating
summaries, including trusted-neighbor ratings for the same item. Aggregates are
fit only on the training split and then joined onto validation/test rows to
avoid label leakage. Training rows use random out-of-fold aggregates, so a
row's own rating does not contribute to its label-derived user/item or
social-neighbor features.

## Rebuilding stages

Each stage short-circuits when its artifact exists. To force recomputation,
pass `--rebuild` (or `-r`) with the stage name:

```bash
./run.sh --rebuild baseline       # retrain just the LightGBM models
./run.sh --rebuild lightgbm       # retrain the engineered LightGBM model
./run.sh --rebuild global_mean    # recompute global-mean baselines
./run.sh --rebuild split          # rebuild splits (remember to also -r baseline)
./run.sh --rebuild exploration    # recompute dataset summary stats
./run.sh --rebuild download       # re-download raw data
./run.sh --rebuild                # rebuild everything (same as --rebuild all)
```

## Logging

The project uses Python's built-in `logging` module. To add logging to a new module:

```python
import logging
logger = logging.getLogger(__name__)
```

Then use `logger.info()`, `logger.warning()`, `logger.error()`, etc. See the [Python logging docs](https://docs.python.org/3/library/logging.html) for more.

## Contributing

**Branches**
To make contributing easy we're going to use development branches then merge
to main. Workflow will look along the lines of:

```bash
git checkout -b <name>/<feature>
# write the code
git add .
git commit -m <description of work>
git push
```

We should be able to handle merges from the gh ui.

**Workflow**
Our goal is a single script that handles everything. Running the project should
be as simple as:

```bash
./run.sh
```

Probably this will just point to a main function. This function will handle 
data download, preprocessing, cross validation, model training, and evalution.
The broad philosophy is that each of these steps should produce an artifact
that serves as the input for the next step. When we run the main script it will
check each step and, if an artifact doesn't already exist, will produce that
artifact. This makes it super easy to work together as state will be entirely
contained in the build script. 
