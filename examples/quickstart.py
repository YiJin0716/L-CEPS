"""Run the released selector on a small reproducible synthetic dataset."""
import os
for key in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(key, "1")

import json
from pathlib import Path
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from explore import load_model, select_features


def main():
    torch.set_num_threads(1)
    rng = np.random.default_rng(302)
    X = rng.normal(size=(80, 100))
    truth = np.array([0, 9, 20, 37, 53])
    y = X[:, truth] @ np.array([2., -2., 1.5, -1.5, 2.]) + rng.normal(size=80)
    result = select_features(X, y, targets=[0.05, 0.10, 0.20],
                             model=load_model(), random_seed=302)
    print(json.dumps({"n": len(y), "p": X.shape[1], "planted_columns": truth.tolist(),
                      "targets": result["targets"], "selected": result["selected"],
                      "candidate_counts": result["candidate_counts"],
                      "final_counts": result["final_counts"]}, indent=2))


if __name__ == "__main__":
    main()
