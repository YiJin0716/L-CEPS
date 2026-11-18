PYTHON ?= python3
RSCRIPT ?= Rscript
WORKERS ?= 4
DEVICE ?= cpu
OUTPUT_DIR ?= runs/refit
IPSS_LFDR_RLIB ?= $(CURDIR)/.r-library
export IPSS_LFDR_RLIB
export PYTHONDONTWRITEBYTECODE = 1
export OPENBLAS_NUM_THREADS = 1
export OMP_NUM_THREADS = 1
export MKL_NUM_THREADS = 1

.DEFAULT_GOAL := help
.PHONY: help demo verify verify-full audit audit-api audit-lfdr note r-deps lfdr train-f train-prior

help:
	@printf '%s\n' 'demo         Run a small synthetic example' 'verify       Check model code and weights (no scientific dependencies)' 'verify-full  Check every distributed artifact' 'audit        Recount the original 4,000 assessment datasets' 'audit-api    Rerun the public API on 28 representative datasets' 'audit-lfdr   Audit all paired local-FDR results and rerun 40 cases (needs R)' 'note         Rebuild the complete PDF note (needs LaTeX)' 'r-deps       Install pinned R comparator packages in IPSS_LFDR_RLIB' 'lfdr         Resume and summarize comparator evaluations, then audit' 'train-f      Refit F into OUTPUT_DIR, using DEVICE=cpu or cuda' 'train-prior  Refit g into OUTPUT_DIR'

demo:
	"$(PYTHON)" examples/quickstart.py

verify:
	"$(PYTHON)" scripts/verify_release.py

verify-full:
	"$(PYTHON)" scripts/verify_release.py --full

audit:
	"$(PYTHON)" simulation/audit_paper_evaluation.py
	"$(PYTHON)" simulation/audit_common_benchmarks.py

audit-api:
	"$(PYTHON)" simulation/paper_evaluation.py verify

audit-lfdr:
	"$(PYTHON)" simulation/audit_lfdr_evaluation.py

note:
	"$(PYTHON)" simulation/render_current_note.py

r-deps:
	"$(RSCRIPT)" --vanilla scripts/install_r_dependencies.R "$(IPSS_LFDR_RLIB)"

lfdr:
	"$(PYTHON)" simulation/lfdr_evaluation.py run --workers "$(WORKERS)"
	"$(PYTHON)" simulation/lfdr_evaluation.py summarize
	"$(PYTHON)" simulation/audit_lfdr_evaluation.py

train-f:
	"$(PYTHON)" simulation/train_current_model.py --component F --device "$(DEVICE)" --output "$(OUTPUT_DIR)/F.pt"

train-prior:
	"$(PYTHON)" simulation/train_current_model.py --component prior --output "$(OUTPUT_DIR)/g.ubj"
