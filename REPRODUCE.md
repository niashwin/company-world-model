# Reproduce the paper

This file tells you how to get every table, figure and number of the paper from this repository. There are two
paths. Path 1 rebuilds the paper from the stored results in a few minutes. Path 2 runs every experiment again,
from the simulated company to the PDF.

## What you need

| Item | Path 1 | Path 2 |
| --- | --- | --- |
| Python 3.12 or 3.13 and `pip install -r requirements.txt` | yes | yes |
| A LaTeX installation with `latexmk` and `pdflatex` | yes | yes |
| Docker, to run the seven Era by Eon simulators | no | yes |
| Network access to Docker Hub (Era images) and Hugging Face (all-MiniLM-L6-v2) | no | yes |
| Disk space for the caches in `results/` | no | about 2 GB |
| Time on a CPU with two cores | minutes | several hours |

`requirements-lock.txt` lists the exact versions that made the stored results (Python 3.13, CPU only). A GPU is not
necessary.

## Path 1: rebuild the paper from the stored results

```sh
pip install -r requirements.txt
make all            # figures, then every number of the paper from results/, then the PDF
pytest -q tests     # checks that need no Era data
```

If your system has `python3` but no `python`, run `make all PY=python3`.

`cwm/fill_paper.py` reads `results/` and writes each number and each generated table of `paper/main.in.tex` into
`paper/main.tex`. The build is reproducible: the same sources give the same `paper/main.pdf`. Two figures (the
t-SNE maps and the state persistence plot) need caches from Path 2. Without the caches, `make` keeps the stored
versions of these two figures.

## Path 2: run every experiment again

```sh
bash scripts/reproduce.sh                 # every stage, in order
bash scripts/reproduce.sh v4 paper site   # only the stages that you name
```

| Stage | What it does | Main outputs |
| --- | --- | --- |
| `data` | Starts the seven simulators (seed 42, date 31 October 2026), exports day 1, moves the simulators to day 2, exports day 2, counts the records | `data/era/day1`, `data/era/day2`, `results/corpus.json` |
| `v1` | Main comparison, alignment, new data, extra seeds, masks, wrong links, held-out records, state tracking, access control, memory reader | `prh.json`, `e3b.json`, `extras_a`–`d.json`, `state.json`, `acl.json`, `memory2.json`, `memory2_seeds.json` |
| `v2` | Customers and employees masked, pretrained encoder, months, access control with a backbone, dynamics, primitives, memory stores with two channels | `extras_e.json`, `v2_*.json`, `memory2_minilm4.json`, `memory2_dual.json`, `memory2_route.json` |
| `v3` | Named questions with retrieval inside the quarter, and questions that come with a record of the customer | `memory2_q_*.json`, `memory_anchor_*.json` |
| `v4` | Decisions: actions, dynamics with actions, event windows, decision simulation, a check of months after the export | `actions.json` |
| `paper` | Figures and PDF | `paper/figures/*.pdf`, `paper/main.pdf` |
| `site` | Data and page of the project website. It needs the data, the caches of `v1` to `v4` and `paper/main.aux` | `website/data/site-data.js`, `website/index.html` |

Later stages read caches that earlier stages write to `results/`: encoders (`*.pt`), embeddings (`*.npy`) and
access-control runs (`*.npz`). Git ignores these files. Run the stages in order the first time.

Runs are deterministic for a given seed and thread count, but not across thread counts. The paper used one thread
per process (`THREADS=1 OMP_NUM_THREADS=1`), which is the default of `scripts/reproduce.sh`. If you run a command of
the table below by itself, set these two variables first. After a stage, run `git status results/` to see which
stored results changed and which files are new.

`cwm/era_up.sh` starts the latest images of Era's simulators. If Era changes its generator, the same seed and date
can give a different company. The `data` stage then prints a warning, because `results/corpus.json` changes. Record
the image versions that you use with `docker images --digests "erabyeon/*"`.

## Where each part of the paper comes from

| Part of the paper | Result files | Command |
| --- | --- | --- |
| Figure 1 (the idea) | drawn in TikZ: `paper/figures/idea.tex` | `make paper` |
| Table 1 and the counts in Section 5 | `corpus.json` | `python cwm/corpus_stats.py` |
| Table 2, Figure 2, Table 15 | `prh.json` (E1), `extras_a.json`, `extras_b.json` | `run_prh.py e1`, `run_extras.py extras_a`, `run_extras.py extras_b` |
| Pretrained rows of Tables 2 and 3 | `v2_pretrained.json` | `make_minilm.py`, then `run_v2.py pretrained` |
| Figure 3 (t-SNE maps) | embeddings cached by `run_prh.py e1` | `make figures` |
| Table 3 (held-out records) | `extras_c.json`, `extras_d.json` | `run_extras.py extras_c`, `run_extras.py extras_d` |
| Figure 4 and Section 6.3 (names masked, wrong links) | `extras_b.json`, `extras_e.json` | `run_extras.py extras_b`, `run_extras.py extras_e` |
| Figure 5 and Section 6.5 (alignment) | `prh.json` (E2) | `run_prh.py e2` |
| Figure 6 (state persistence) | `persistence.json`, from `state_vectors.npz` of `run_state.py` | `make figures` |
| Figure 7 and Section 6.6 (new data, retraining) | `prh.json` (E3), `e3b.json` | `run_prh.py e3`, `run_e3b.py` |
| Table 4, Figure 8, Table 14 (access control) | `acl.json` | `run_acl.py train:... eval` |
| Table 5 (access control with a backbone) | `v2_acl.json` | `run_v2.py acl acl_combo acl_pub` |
| Table 6 (what states hold) | `v2_state.json` | `run_v2.py state` |
| Table 7 (named questions) | `memory2_q_*.json`, `memory2_q_retrieval.json` | `KEYS=quarter ... run_memory2.py`, `memory_retrieval.py` |
| Table 8 (anchored questions) | `memory_anchor_{masked,plain}*.json` | `run_memory_anchor.py` |
| Table 9 (quarters and months) | `v2_monthly.json` | `run_v2.py monthly` |
| Table 10 (dynamics) | `v2_time.json` | `run_v2.py time` |
| Table 11 (drift, misalignment, similar situations) | `v2_primitives.json` | `run_v2.py primitives situations` |
| What each reader may read (Section 6.7) | `v2_scope.json` | `run_v2.py scope` |
| Tables 12 and 13 and Appendix B (decisions) | `actions.json` | `run_actions.py table dynamics events simulate checks` |
| Table 16 (state tracking) | `state.json` | `run_state.py` |
| Table 17 (memory reader) | `memory2.json`, `memory2_seeds.json`, `memory2_minilm4.json`, `memory2_dual.json`, `memory2_route.json` | `run_memory2.py` with the settings in `scripts/reproduce.sh` |
| Training times | `train_seconds` and `seconds` in `prh.json`, `extras_*.json`, `v2_pretrained.json`, `v2_monthly.json`, `acl.json` | the commands above |

`results/memory_anchor_masked_a.json`, `..._b.json` and the same `plain` files are parallel parts of one run.
`cwm/fill_paper.py` merges them. A single run of `run_memory_anchor.py` writes `memory_anchor_masked.json` (or
`..._plain.json`), and when that file exists, `cwm/fill_paper.py` uses it in place of the parts. The files
`extras_a.json` to `extras_e.json` are five different experiments, not parts of one run.

## Checks

- `pytest -q tests` runs smoke tests on a small synthetic company, the access policy, non-interference, the
  good-view head and the rotation invariance of placement. It also checks that the numbers typed in the text of
  the paper agree with `results/`.
- `python tools/ste_check.py paper/main.tex` measures how closely the text follows the writing rules of ASD-STE100
  (Simplified Technical English). `make check` runs the tests and requires at least 90%.
- `python tools/tex_invariants.py old.tex new.tex` checks that an edit of the paper keeps every placeholder, label,
  reference and citation.

## The data

Era by Eon generates the company. Era is free, and no real person, account or conversation is in the data. This
repository does not include the generated data. `scripts/reproduce.sh data` makes it again from the fixed seed and
date. Read Era's terms before you share the data.

## The website

The repository includes the generated website data (`website/data/site-data.js`), so the site works from a clone:
run `python -m http.server -d website 8000` and open `http://localhost:8000`. To make the data again, run the `site`
stage after Path 2. It needs the data and the caches, because it computes the t-SNE maps from the stored embeddings.
