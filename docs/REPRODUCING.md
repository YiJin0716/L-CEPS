# Reproducing the released model and assessments

Run the commands below from the repository root, after following the
[installation instructions](../README.md#quick-start). This repository contains
the complete, self-contained artifact.
Use a separate checkout for new experiments so the distributed records remain
available for comparison.

## 1. Verify the download and run the model

```bash
make verify
make demo
git lfs pull --include="" --exclude=""
make verify-full
```

`verify` checks the released inference sources, fitted weights, and model schema.
`verify-full` checks all distributed files against `release_manifest.json`.
Both use Python's standard library and report missing files or unresolved LFS
pointers. The demo generates an `80 × 100` Gaussian predictor matrix with five
planted variables and invokes the public API; it is a usage example, not an
additional assessment setting.

The two fitted components are described in [models/README.md](../models/README.md).
Inference runs on CPU and does not read training or assessment datasets.

## 2. Rebuild the figures and note from saved results

Install a LaTeX distribution with `pdflatex`, the packages listed in the preamble
of `final_model_note.tex` (including `algorithm` and `algpseudocode`), and Poppler's
`pdftotext` command. Then run:

```bash
make note
```

The renderer checks the frozen assessment inputs and the completed comparator
audit before rebuilding the local-FDR figures and note. It retains the original
IPSS comparison figures. It also checks the compiled PDF for paired axes,
dataset descriptions above each figure, and unresolved compilation warnings.
The distributed note has 37 pages and 49 paired figures. The original comparison
occupies 24 figures; the local-FDR comparison adds 24 dataset-group figures and
one overall figure. All plot text is English, with FDR on the left and TPR on
the right.

| Report content | Data and results | Code |
|---|---|---|
| IPSS Gaussian/RNA-seq and additional assessments: 28 settings, 2,800 draws | `data/paper_evaluation_*` | `simulation/paper_evaluation.py` |
| Sparse linear, Friedman #1, MNIST and BASEHOCK assessments: 12 settings, 1,200 draws | `data/common_benchmark_*` | `simulation/common_benchmarks.py` |
| Local-FDR comparisons on the same 4,000 draws | `data/lfdr_comparison_*` | `simulation/lfdr_evaluation.py`, `simulation/lfdr_methods.R` |
| Original 24 figure groups | `data/paper_evaluation_plot_data.json` | `simulation/render_paper_evaluation.py` |
| Additional 25 figure groups | `data/lfdr_comparison_plot_data.json` | `simulation/render_lfdr_comparison.py` |

Recompiling or rerunning analyses can change output-file checksums (including PDF
metadata). `verify-full` checks the downloaded release, not newly generated files.
Numerical audits are separate from that byte-for-byte integrity check.

## 3. Independently check saved selections and curves

```bash
make audit
make audit-api
```

The first command regenerates inputs from the recorded seeds, checks support
sets and input hashes, and recounts saved selections. The second additionally
runs the public selector on one draw from each of the 28 paper-assessment
settings and compares its selections with the saved records. These are
substantive numerical checks, not only checks that a script can import.

Every assessment has 100 independent draws per setting, with
`50 <= n <= p <= 500`; all 40 settings are retained. The protocols fix the
generating parameters, dimensions, random seeds, methods, and target grid.
See [data/README.md](../data/README.md) and the descriptions before each figure
in the note for the exact data-generating methods.

### Generate the original assessment results again

The `run` commands resume existing SQLite records. To recompute every draw,
work in a separate copy of this directory and move its four saved result
databases out of `data/` first (`paper_evaluation_0.sqlite`,
`paper_evaluation_1.sqlite`, `common_benchmark_0.sqlite`,
`common_benchmark_1.sqlite`). Keep the frozen protocols and predictor pools.
Run both shards of each assessment:

```bash
python simulation/paper_evaluation.py run --workers 4 --shards 2 --shard 0
python simulation/paper_evaluation.py run --workers 4 --shards 2 --shard 1
python simulation/paper_evaluation.py summarize
python simulation/common_benchmarks.py run --workers 4 --shards 2 --shard 0
python simulation/common_benchmarks.py run --workers 4 --shards 2 --shard 1
python simulation/common_benchmarks.py summarize
```

Full regeneration reruns IPSS fits and is much more expensive than checking
saved outputs. Do not use `freeze` to reproduce this assessment: it would define
a new protocol. New experimental outputs should be compared numerically with
the release. Frozen local-FDR comparisons require the original database bytes,
so keep the distributed databases for that comparison and for rebuilding its note.

## 4. Reproduce the local-FDR comparisons

The recorded software is R 4.5.2, `ncvreg` 3.16.0, `ashr` 2.2-63, and
`jsonlite` 2.0.0. Package compilation may require a C/C++/Fortran toolchain.

```bash
make r-deps
make audit-lfdr
```

`r-deps` installs the specified direct dependencies into `.r-library/` using
CRAN and its archives; transitive dependencies are resolved by R. This is not a
complete lockfile for every transitive dependency. The local directory is ignored
by Git. To use an existing installation:

```bash
make audit-lfdr IPSS_LFDR_RLIB=/path/to/R/library
```

The Makefile explicitly supplies the library directory, overriding the historical
cluster default in the frozen evaluator. For direct Python commands, set
`IPSS_LFDR_RLIB` yourself. The audit checks all 4,000 rows and exactly reruns one
comparator evaluation from each of the 40 settings.

`make lfdr WORKERS=4` resumes comparator work, summarizes it, and runs the audit.
To recompute all comparator rows, use a separate artifact copy and move only
`data/lfdr_comparison.sqlite` aside first. Keep the original paper/common result
databases: their saved model selections are reused for the paired comparison.

The recorded Sun–Cai implementation uses marginal association statistics;
cumulative ℓ-values use those same statistics. The Lasso scores come from the
authors' `ncvreg` implementation. Its pointwise and cumulative selection rules
are reported separately. Performance is measured against the columns that
directly generated the response, which differs from the marginal null hypotheses
of these methods. Methods and assumptions are specified in the frozen
[`lfdr_comparison_protocol.json`](../data/lfdr_comparison_protocol.json).

## 5. Refit the learned components

```bash
make train-f DEVICE=cuda OUTPUT_DIR=runs/refit
make train-prior OUTPUT_DIR=runs/refit
```

Use `DEVICE=cpu` if a GPU is unavailable; the full fit can be substantially slower.
`train-f` uses the saved measurements for 5,216 training datasets and 1,636,538
variable rows, follows the recorded initial fit and continuation, and writes
`runs/refit/F.pt`. `train-prior` writes `runs/refit/g.ubj` from the saved prior
inputs and grouped weights. The training scripts refuse to overwrite existing
outputs. Newly fitted components are not automatically installed as the released
model.

Training settings, seeds, optimizer choices, and selected epochs are in
[`current_training.json`](../data/current_training.json). Raw training generators
are retained in `simulation/training_generators.py` and
`simulation/current_data.py`; the distributed measurement cache is the input to
the refitting commands. Validation arrays are kept separately from fitting inputs.

## Numerical environment and provenance

Python package versions are pinned in `requirements.txt`. The original numerical
checks used single-threaded BLAS and `OPENBLAS_CORETYPE=SkylakeX` on an AVX-512
capable CPU. On compatible hardware, use the same environment for exact checks:

```bash
OPENBLAS_CORETYPE=SkylakeX make audit
```

On other CPUs, leave OpenBLAS's automatic kernel selection enabled. Floating-point
differences can affect exact input-hash or score-equality assertions; matching the
recorded numerical environment matters for those checks. GPU fitting may also
differ slightly across hardware and library versions.

Historical paths inside frozen protocols are resolved through `relocation.json`
by `simulation/release_files.py`; they do not require access to the original
filesystem. Model source and weight hashes remain those of the released selector.
The [root README](../README.md) and `make help` are the maintained public entry points.
