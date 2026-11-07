# Toward World Models of Organizations

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

**Every system sees the same company.** Email, chat, the CRM, support tickets, call recordings, calendars and
issue trackers each record part of what is happening with every customer, deal and project, yet none of them
records the situation itself. This repository learns that missing piece: a shared **state space** in which every
record lands according to the situation it describes, whichever system wrote it. It is the first part of a
**world model of an organization**.

![The idea](paper/figures/idea.png)

Every company will come to run on an *operational world model*: a model of how the company actually functions,
which people and AI agents can understand the company through, learn on the way a new employee learns it, and use
to try changes before making them. This repository builds the first layers of such a model: a shared state space
learned from the links a company already keeps, and access control built into it, so that each person's view
depends only on what they may read.

The training signal comes for free. The Platonic Representation Theorem (Balestriero & LeCun, 2026) shows that
JEPAs trained on *good views*, two different observations of the same underlying state, converge to the same
representation up to a rotation. Organizations already record good views: records that different systems link to
the same customer in the same time window. We train small text JEPAs from scratch on those links, on a simulated
company from [Era by Eon](https://www.eon.io/blog/era-build-and-test-ai-agents), and compare them with
corruption-based views and next-token prediction on identical text.

Paper: [`paper/main.pdf`](paper/main.pdf) (source in [`paper/`](paper)).

## Main results

One simulated company: seven systems, 21,972 customer-linked records, 21 active customers, exact ground truth.

| Test | Good views | Bad views | Next-token | TF-IDF |
| --- | --- | --- | --- | --- |
| **Held-out records** (links hidden in training): placed with the right customer | **76%** | 19% | 20% | 53% |
| Training records: 10 nearest neighbours **from other systems** on the same customer (3 seeds) | **95%** | 14% | 13% | 35% |
| ...with customer names, contacts, domains and ids masked (training records) | **48%** | | | 14% |
| ...when 25% / 50% of the training links are wrong | **84% / 68%** | | | |
| Encoders trained **separately** on messages and on systems of record, one rotation from 64 anchors: same-state recall@1 (chance 4%) | **21%** | 4% | 4% | |
| **Retrained** encoder searching the **old** index through one rotation, queries unseen by both models: same-customer precision@10 | **79%** (3% without rotation; old model 72%) | | | |

The links are weak supervision (they define the positive pairs), so numbers on training records are upper bounds;
the held-out test is the fair one. Alignment across separately trained encoders is partial, and only the leading
directions of the space survive retraining.

**Access control, built in.** A learned world model can leak across access boundaries in three ways: search
results, states aggregated over records a reader may not open, and the encoder's own training. `cwm/acl.py`
assigns every record its readers (a role-based policy for Era's company) and the model closes all three paths:
search and states use only readable records, and each reader is served an encoder trained only on records they may
read. What a reader receives then depends only on what they may read (non-interference), which `cwm/run_acl.py`
checks with planted secrets and a byte-for-byte retraining test. In that test, a support engineer who may not read
CRM records searches for a codename planted in them: with an encoder trained on everything, the right customer
comes back 100% of the time even though search is filtered to records the engineer may read; with access control
built in, every result is identical with and without the planted secret (paper, Section 6.7).

What does not work yet, and is reported in the paper: the state vectors do not forecast next quarter's facts
better than the majority class, they are poor at facts about which system wrote a record (such as "was there a
call?"), and a memory-token reader gave inconclusive gains. A useful organizational world model probably needs a
second, detail channel per record and a learned model of dynamics; both are next steps.

## Reproduce

Everything runs on a laptop CPU. A full rerun of the experiments takes a few hours on two cores.

```sh
pip install -r requirements.txt

# 1. Generate the company (needs Docker; Era's simulator images are pulled from Docker Hub, no account needed)
sh cwm/era_up.sh day1
python cwm/era_export.py --out data/era/day1
for p in 8080 8081 8082 8083 8084 8085 8087; do curl -s -XPOST localhost:$p/_admin/state/day2; done
python cwm/era_export.py --out data/era/day2

# 2. Experiments (each writes results/<name>.json)
STEPS=1500 python cwm/run_prh.py e1            # good vs bad views vs next-token (seed 0); caches embeddings
STEPS=1000 python cwm/run_prh.py e2            # separately trained encoders + one rotation
STEPS=1500 python cwm/run_prh.py e3            # a new quarter of data
python cwm/run_e3b.py                          # retraining (needs e1's cached encoder): alignment, old-index search
python cwm/run_extras.py extras_a seeds:good,bad
python cwm/run_extras.py extras_b mask:good noise:0.25,0.5 seeds:ntp
python cwm/run_extras.py extras_c heldout:base,good,bad
python cwm/run_extras.py extras_d heldout:ntp
python cwm/run_state.py                        # state tracking and next-quarter forecasting
python cwm/run_acl.py train:open train:account_executive train:support_engineer train:support_engineer_clean \
                      train:software_engineer train:financial_analyst eval      # access control
python cwm/run_memory2.py                      # memory-token reader (appendix)
SEEDS=1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_seeds python cwm/run_memory2.py
STORE=minilm SUFFIX=_minilm SEEDS=0,1,2 VARIANTS=closed_book,top16_pretrained python cwm/run_memory2.py

# 3. Figures and paper (numbers in the paper are filled from results/ by cwm/fill_paper.py)
make all
```

Quick checks that need no Era data (a tiny synthetic company, the access policy, non-interference):

```sh
pytest -q tests
```

The exact company is fixed by the seed (42) and simulated date (2026-10-31) in `cwm/era_up.sh`. Generated data
is not included in this repository; regenerate it with the commands above.

## Layout

| Path | What it is |
| --- | --- |
| `cwm/era_up.sh`, `cwm/era_export.py` | Start Era's seven simulators and export every record into one table plus ground truth |
| `cwm/data.py` | Records to states (customer x quarter), tokens, system families |
| `cwm/jepa.py` | Bag-of-n-grams encoder, LeJEPA loss with SIGReg, good and bad views, next-token baseline |
| `cwm/metrics.py` | Cross-system neighbour precision, probes, explained fractions, Procrustes, mutual kNN |
| `cwm/run_prh.py`, `cwm/run_e3b.py` | Main comparison, alignment across systems, behaviour over time |
| `cwm/run_extras.py` | Extra seeds, identity masking, wrong-link robustness, held-out records |
| `cwm/run_state.py` | State tracking and forecasting probes |
| `cwm/acl.py`, `cwm/run_acl.py` | Access policy, per-reader views, leak tests, cost and non-interference |
| `tests/` | Smoke tests run by GitHub Actions on every push |
| `cwm/memory.py`, `cwm/run_memory2.py` | Memory-token reader over the store (appendix) |
| `cwm/make_figures.py`, `cwm/fill_paper.py` | Figures, and the paper's numbers from `results/` |
| `results/` | Every result reported in the paper, as JSON |
| `paper/` | LaTeX source (`main.in.tex` with placeholders, generated `main.tex`), figures, bibliography, PDF |

## Moving to real data

Replace `era_export.py` with connectors for your own systems and keep the record schema
(`system, ts, author, thread, text` plus link fields such as `account_id`). Link fields only decide which records
are positives; the encoders never read them. Replace the role-based policy in `cwm/acl.py` with each system's own
permissions (channel membership, email recipients, record sharing); the three controls stay the same.

## Data and licence

Era by Eon is free and generates a synthetic company; no real person, account or conversation is involved. The
generated data is not redistributed here; regenerate it with `cwm/era_up.sh` and `cwm/era_export.py`, and check
Era's terms before sharing it. Code is MIT-licensed (see `LICENSE`).

## Citation

```bibtex
@article{gopinath2026organizations,
  title  = {Toward World Models of Organizations: Every System Sees the Same Company},
  author = {Gopinath, Ashwin},
  year   = {2026},
  note   = {Technical report}
}
```

## Acknowledgments

Era by Eon is free and made this work possible. SIGReg follows LeJEPA (Balestriero & LeCun, 2025); the
implementation here is our own. The memory reader follows Memorilla (Fallahpour et al., 2026).
