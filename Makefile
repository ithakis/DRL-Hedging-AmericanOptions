# Reproduce the results of the revised paper (submission 4842).
#
#   make all       every result, table and figure from the stored weights, then the PDF   (~30-60 min on the M1 Air)
#   make paper     only tables + figures + PDF from the stored result files                (~2 min)
#   make retrain   train all 45 agents from scratch, then `make all`                       (several days, local only)
#
# Stored inputs (in git): review/results/retrain/weights/*.npz (agents), review/results/retrain/runs/*
# (validation curves, selection), pricing_tables/*.npz (Heston price tables). Simulated markets are cached
# in markets/ (not in git, rebuilt on first use). Generated paper files go to review/results/paper/.

PY     ?= python
PAPER  ?= Paper__Higher_Order_Hedging_for_Am_Opt_DRL
export PYTHONPATH := $(CURDIR)
A      := $(PY) review/retrain/analysis.py

.PHONY: all paper results baselines evaluate tables figures pdf retrain markets train export help

all: results paper

paper: tables figures pdf

# ---------------------------------------------------------------- results (CSV / JSON / markdown)
results: baselines evaluate
	$(PY) review/retrain/report.py > /dev/null

baselines:                       # tune rules (a)/(b) on the validation paths, all baselines on the test paths
	$(A) rules --block A
	$(A) rules --block B

evaluate:                        # RL agents on the test paths; needs baselines_*.json
	$(A) main --block A
	$(A) main --block B
	$(A) deltacost --block B
	$(A) robustness
	$(A) early
	$(A) rulesel --block A
	$(A) rulesel --block B

# ---------------------------------------------------------------- paper inputs
tables:                          # review/results/paper/tables/*.tex (\input by main.tex)
	$(PY) review/retrain/paper_tables.py

figures:                         # review/results/paper/figures/*.pdf
	$(PY) review/retrain/make_figure1.py
	$(PY) review/retrain/make_figure_seeds.py

pdf:
	@if [ ! -f "$(PAPER)/main.tex" ]; then echo "paper source not found in $(PAPER)/ (set PAPER=...)"; exit 1; fi
	cd "$(PAPER)" && latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex

# ---------------------------------------------------------------- retrain from scratch (local, long)
retrain: markets train export all

markets:                         # all training / validation / test markets (uses all cores)
	$(PY) review/retrain/prebuild_markets.py --blocks A B --robustness

train:                           # finished runs in logs_torch/retrain are skipped; delete that folder for a clean start
	bash review/retrain/run_block.sh A 6
	bash review/retrain/run_block.sh B 6

export:                          # logs_torch/retrain -> review/results/retrain/{weights,runs,mapping.csv}
	$(A) export

help:
	@sed -n '1,9p' Makefile
