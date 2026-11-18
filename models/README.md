# Released fitted components

[`active_model.json`](active_model.json) identifies the released model and checksums
its inference code, weights, and input schema. Loading verifies these checksums.

| File | Component |
|---|---|
| `retained_mlp_hard_rows.pt` | MLP F, widths 443 → 128 → 64 → 1, including input normalization |
| `final/prior.ubj` | XGBoost estimator g of the relevant-variable fraction |
| `expanded_f.json` | Dataset and variable measurement names and ordering |
| `retained_mlp.json`, `retained_mlp_hard_rows.json` | Fitting and release metadata for this model |

The filenames are retained for compatibility with the frozen model and
assessment records. There are two fitted components, F and g. The selector uses
the union of ten IPSS selection sets at the target FDR, followed by F's ordering
and cumulative-average selection. It has no additional fitted scorer or
reference-data calibration step.

Download weights with Git LFS, then run `make verify` in the parent directory.
See the [public API](../README.md#python-api) and
[refitting commands](../docs/REPRODUCING.md#5-refit-the-learned-components).
