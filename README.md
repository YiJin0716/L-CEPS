# A Learned Variable Selector Built on IPSS

Official implementation and reproducibility materials by **Yi Jin** and
**Omar Melikechi**. This directory contains the released model, fitted weights,
training inputs, assessment records, and the technical note. The paper title is
not yet finalized; the citation currently describes the software.

[Technical note](final_model_note.pdf) · [Reproduction guide](docs/REPRODUCING.md) ·
[Data sources](data/README.md) · [Model weights](models/README.md) · [Citation](CITATION.cff)

## The model

Given `X`, `y`, and target FDR `alpha`, the model:

1. Runs GB-IPSS and Lasso-IPSS and retains the union of their selections at
   `alpha`, using each fixed delta in `{-1, 0, 1, 2, 4}`.
2. Calculates dataset and variable measurements. The fitted estimator `g`
   estimates the relevant-variable fraction from the dataset measurements.
3. Forms a 443-dimensional vector for each candidate and applies the fitted
   MLP `F`, with widths `443 → 128 → 64 → 1`, to estimate its null probability.
4. Sorts candidates by that probability and selects the largest complete-tie
   prefix whose average predicted null probability is at most `alpha`.

The fitted components stay fixed on a new dataset. The implementation and weights
are identified by [`models/active_model.json`](models/active_model.json).
Empirical results do not establish a general FDR guarantee; screening can discard
relevant variables. See the note for the full algorithm and its limits.

## Quick start

Tested with Python 3.10 on Linux. CPU inference is supported; a GPU is not required.
Install [Git LFS](https://git-lfs.com/) before downloading binary artifacts.

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone https://github.com/YiJin0716/final_model.git
cd final_model
git lfs install --local
git lfs pull --include="models/**" --exclude=""
python3.10 -m venv .venv
. .venv/bin/activate
python -m pip install -r requirements.txt
make verify
make demo
```

The initial LFS download contains only fitted weights, about 0.7 MB. Download the
complete training and assessment artifacts, about 750 MiB, to reproduce experiments:

```bash
git lfs pull --include="" --exclude=""
make verify-full
```

A GitHub source ZIP may contain LFS pointers instead of weights or datasets.
`make verify` detects missing weights and unresolved pointers.

### Python API

From this directory:

```python
import numpy as np
import torch
from explore import load_model, select_features

torch.set_num_threads(1)
model = load_model()
with np.load("input.npz", allow_pickle=False) as data:
    result = select_features(data["X"], data["y"], targets=[0.05, 0.10, 0.20],
                             model=model, random_seed=302)
print(result["selected"])
```

`X` must be a finite numeric `n × p` array and `y` a finite, nonconstant length-`n`
vector. Outputs contain one list of **zero-based original column indices** per
target, plus `candidate_counts`, `final_counts`, and prefix scores. The current
assessment covers `50 <= n <= p <= 500`.

### Command line

```bash
python -m explore.current input.npz --targets 0.05 0.10 0.20 --seed 302 --output selections.json
```

[`examples/quickstart.py`](examples/quickstart.py) generates a small synthetic
input and prints the selected columns.

## Repository layout

```text
final_model/
├── explore/                 # Released inference implementation
├── models/                  # F and g weights, input schema, model manifest
├── data/                    # Training inputs, predictor pools, protocols, results
├── simulation/              # Training, assessment, audits, plotting
├── examples/                # Runnable use of the public API
├── scripts/                 # Artifact checks and R dependency installation
├── docs/REPRODUCING.md       # Commands and artifact-to-result mapping
├── Makefile                 # Entry points for common tasks
├── requirements.txt         # Tested Python dependency versions
├── CITATION.cff              # Software citation and authors
├── release_manifest.json    # Checksums of the distributed files
├── final_model_note.pdf     # Algorithm and all assessment figures
├── final_model_note.tex     # Note source
├── final_model_note_figures.pdf
├── lfdr_comparison_figures.pdf
└── lfdr_comparison.png      # Overall local-FDR comparison
```

`explore/` is the retained inference package, despite its historical name.
No earlier model weights are needed. Runtime paths are relative to this directory.

## Reproduce the results

After downloading the complete artifacts, run:

```bash
make help             # List available commands
make audit            # Recount the original IPSS/comparison assessments
make note             # Rebuild the complete note and local-FDR figures
make r-deps           # Install R dependencies for local-FDR comparators
make audit-lfdr       # Check 4,000 paired draws and rerun 40 comparator cases
```

`make note` needs `pdflatex` and `pdftotext`; R is needed only for local-FDR
comparators. Training, regeneration, hardware details, and audit descriptions are
in the [reproduction guide](docs/REPRODUCING.md). `make help` does not start training
or assessments.

The assessment contains **40 settings × 100 draws = 4,000 datasets**, all with
`50 <= n <= p <= 500`. Every setting is reported. The note has 49 paired FDR/TPR
figures. Baselines include GB-IPSS, Lasso-IPSS, LassoCV, Sun–Cai, cumulative
ℓ-value, and Lasso local FDR. The last method is reported with its pointwise rule
and an additional cumulative-average rule. All comparisons use the same planted
variable sets; the distinction from marginal-association null hypotheses is
explained in the note.

## Citation and related work

Use [`CITATION.cff`](CITATION.cff) to cite this software and record the Git commit
used for an experiment. The technical note is the current model description.

This release uses [IPSS](https://github.com/omelikechi/ipss). Its organization
follows the separation of implementation, data, and simulations in the
[IPSS JASA reproduction repository](https://github.com/omelikechi/ipss_jasa),
and the executable reproduction entry points in the
[Lasso FDR paper repository](https://github.com/pbreheny/lassoFDRpaper).
Comparator references and implementation choices are recorded in
[`data/lfdr_comparison_protocol.json`](data/lfdr_comparison_protocol.json).

This standalone repository contains the official artifact for the current model.
