# Training data, assessment data, and recorded results

Binary artifacts are stored with Git LFS. Download them from the repository with
`git lfs pull --include="" --exclude=""` and check them
using `make verify-full` from the repository root.

## Training and validation

| Files | Purpose |
|---|---|
| `block_training_cache.npz` | F fitting inputs, variable labels, weights, dataset groups, and input normalization |
| `current_training.json` | 5,216 training datasets, seeds, generation settings, and the two-stage fitting recipe |
| `current_prior_training.npz`, `prior_folds.json` | g fitting inputs and grouped folds |
| `current_standard_validation.npz`, `retention_new_validation.npz` | Validation measurements used during model development |
| `retained_mlp_*.json`, `seed_exclusions.json` | Release provenance and seed exclusions |

The fitting cache contains 1,636,538 variable rows. Validation records are not
the current 4,000-dataset assessment. Model-release metadata retains historical
development checks; current reported experiments are identified by the three
assessment protocols below.

## Assessment records

| Prefix | Coverage |
|---|---|
| `paper_evaluation_` | 28 settings × 100 draws = 2,800 datasets |
| `common_benchmark_` | 12 settings × 100 draws = 1,200 datasets |
| `lfdr_comparison_` | Additional methods evaluated on exactly those same 4,000 datasets |

Each `*_protocol.json` records the settings, dimensions, seeds, target grid,
methods, and frozen checksums. SQLite databases retain per-dataset selections and
diagnostics. `*_counts.npz` stores true/false selection counts; `*_results.json`
stores summary curves. `*_audit.json` and `*_verification.json` record checks.
The local-FDR summary is also available as `lfdr_comparison_summary.csv`.
All 40 settings are reported; none are filtered by comparative performance.

## Real predictor pools

The original response labels are not used. Rows and columns of the real predictor
pools are sampled according to the frozen protocols, and new responses are
generated from known planted variables. These are **semisynthetic** assessments.

| Predictor pool | Pool dimensions | Source |
|---|---:|---|
| Ovarian RNA-seq | 569 × 6,426 | [IPSS Bioinformatics paper](https://github.com/omelikechi/ipss_bioinformatics/tree/b9a4d4e8d7e8a738039ab58ce2b5718680629f5a/simulations/data) |
| ISOLET | 1,560 × 617 | [scikit-feature](https://github.com/jundongl/scikit-feature/tree/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/data) |
| COIL20 | 1,440 × 1,024 | [scikit-feature](https://github.com/jundongl/scikit-feature/tree/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/data) |
| Human activity recognition | 10,299 × 561 | [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) |
| MNIST training images | 60,000 × 784 | [Keras MNIST archive](https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz) |
| BASEHOCK | 1,993 × 4,862 | [scikit-feature](https://github.com/jundongl/scikit-feature/tree/48cffad4e88ff4b9d2f1c7baffb314d1b3303792/skfeature/data) |

These are pool sizes; each assessment draw satisfies `50 <= n <= p <= 500`.
Exact source URLs, source checksums, preprocessing, and pool construction are in
[`paper_evaluation_sources.json`](paper_evaluation_sources.json) and
[`common_benchmark_sources.json`](common_benchmark_sources.json).
The files retain their upstream data provenance; this project does not claim
authorship of those datasets. `paper_ipss_source.zip` retains the IPSS simulation
source snapshot used to establish the assessment design.

See the [reproduction guide](../docs/REPRODUCING.md) for execution commands and
the [technical note](../final_model_note.pdf) for each response-generating formula.
