# Toward World Models of Organizations

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

**Every system sees the same company.** Email, chat, the CRM, support tickets, call recordings, calendars and
issue trackers each record part of what is happening with every customer, deal and project, yet none of them
records the situation itself. This repository learns that missing piece: a shared **state space** in which every
record lands according to the situation it describes, whichever system wrote it, with **access control built in**.
It is the first layer of a **world model of an organization**.

![The idea](paper/figures/idea.png)

Every company will come to run on an *operational world model*: a model of how the company actually functions,
which people and AI agents can understand the company through, learn on the way a new employee learns it, and use
to try changes before making them. This repository builds the first layers of such a model and tests, honestly,
how far the rest of it gets on the same data.

The training signal comes for free. The Platonic Representation Theorem (Balestriero & LeCun, 2026) shows that
JEPAs trained on *good views*, two different observations of the same underlying state, converge to the same
representation up to a rotation. Organizations already record good views: records that different systems link to
the same customer in the same time window. We train small text JEPAs on those links, on a simulated company from
[Era by Eon](https://www.eon.io/blog/era-build-and-test-ai-agents), compare them with corruption-based views and
next-token prediction on identical text, and repeat the key comparisons with a frozen pretrained encoder.

**Paper:** [`paper/main.pdf`](paper/main.pdf) (version 2; source in [`paper/`](paper)).
**What changed in version 2 and why:** [`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md) and
[`CHANGELOG.md`](CHANGELOG.md).

## Results at a glance

One simulated company: seven systems, 21,972 customer-linked records, 21 active customers, exact ground truth.
Every number below is in the paper and in `results/`; one seed unless noted.

### The state space (what works)

| Test | Good views | Bad views | Next-token | TF-IDF | Pretrained encoder | Pretrained + good-view head |
| --- | --- | --- | --- | --- | --- | --- |
| **Held-out records** (links hidden in training): placed with the right customer | **76%** | 19% | 20% | 53% | 53% | 62% |
| Training records: 10 nearest neighbours **from other systems** on the same customer | **95%** (3 seeds) | 14% | 13% | 35% | 40% | 72% |
| ...with customer names, contacts, domains and ids masked | **48%** | | | 14% | | |
| ...with **employee** names, handles and chat names masked as well | **47%** | | | 14% | | |
| ...when 25% / 50% of the training links are wrong | **84% / 68%** | | | | | |
| Encoders trained **separately** on messages and on systems of record, one rotation from 64 anchors: same-state recall@1 (chance 4%) | **21%** | 4% | 4% | | | |
| **Retrained** encoder searching the **old** index through one rotation: same-customer precision@10 | **79%** (3% without rotation) | | | | | |

- The links are weak supervision, so numbers on training records are upper bounds; the held-out test is the fair one.
- The same good views lift a frozen pretrained sentence encoder (all-MiniLM-L6-v2) from 53% to 62% on held-out
  records, so the recipe also improves an off-the-shelf backbone. On this company it stays below the small encoder
  trained from scratch, which learns the company's own vocabulary.
- Masking employees as well as customers changes almost nothing (48% → 47%): the residual signal is not "which
  employee handles this account".

### Access control, built in

A learned world model can leak across access boundaries in three ways: search results, states aggregated over
records a reader may not open, and the encoder's own training. `cwm/acl.py` gives every record its readers and the
model closes all three paths, so what a reader receives depends only on what they may read (non-interference,
checked byte for byte).

- **Filtering is not enough.** A support engineer who may not read CRM records searches for a codename planted in
  them: with an encoder trained on everything, the right customer comes back **100%** of the time even though search
  is filtered to readable records. With access control built in, every result is identical with and without the
  planted secrets, for the from-scratch encoder and for the pretrained backbone with per-reader heads.
- **The cost, and what can be recovered without breaking the guarantee** (held-out readable records placed with
  the right customer):

| Reader (share readable) | Own encoder | Backbone + own head | Both side by side | Trained on everything (leaks) |
| --- | --- | --- | --- | --- |
| CEO (100%) | **76%** | 62% | | 76% |
| Account executive (81%) | **76%** | 70% | | 78% |
| Support engineer (86%) | **74%** | 61% | | 75% |
| Software engineer (22%) | 28% | 29% | **32%** | 66% |
| Financial analyst (9%) | 23% | **43%** | 30% | 65% |

  A pretrained backbone holds no company data, so every reader may share it: it rescues the financial analyst
  (23% → 43%) but barely helps the software engineer, whose records are 87% support tickets. A base trained on the
  534 linked records everyone may read does not help. **Rotations cannot help by construction**: they make
  different readers' views comparable, but nearest-centroid placement is unchanged by any rotation (unit-tested),
  and a rotation toward a space the reader may not read would itself depend on records outside their rights.
  A size-matched random sample of all systems places 42% and 31% of the two readers' records: most of the gap to
  "trained on everything" is the records a reader may not read at all. Knowledge from outside the company can make up
  part of it, as the backbone does for the analyst; beyond that, only a change in the access policy can.

### The rest of a world model (mostly negative, reported as such)

| Question | Result |
| --- | --- |
| **Two channels** (state + detail) on the 144 memory questions, 4 seeds | Detail only (pretrained) 70.8%, state only (good views) 66.1%, concatenated 69.6%, routed (retrieve by state, read detail) 69.3%. **Neither combination beats the detail channel alone**; these questions are all about one system's records. |
| Two channels for state facts ("was there a sales call this quarter?") | Good views alone 72% (about the majority), with a pretrained detail channel 87%, with bad views 95%: the detail channel restores the facts, but no combination beats its better channel on average. |
| **Monthly windows** (405 states, 398 transitions) | Held-out records placed in the right customer-month 17% vs 13% for "customer known, busiest month": finer windows help a little. |
| Forecasting next month's facts | 46–54% for every representation vs 54% majority class and 72% for copying this month's facts: **no forecast**. |
| Linear dynamics (state → next state) | With its regularization tuned by nested cross-validation, a ridge map beats "no change" in every space (2–31% less error) but never by more than 3.4 points beyond one shrink-toward-the-mean factor: it learns mean reversion, not dynamics. |
| **Drift** (weekly state jump vs escalations and deal-stage changes, 279 event weeks) | AUROC 0.47–0.54, within about two standard errors of chance; record volume alone 0.57. |
| **Cross-team misalignment** (sales vs support state distance vs at-risk months) | AUROC 0.48–0.57, within noise. |
| **Similar situations across customers** | 18% of the 5 nearest other-customer states share the situation vs 8% chance (every representation does about as well); still 17% vs 7% when restricted to other quarters. For customers in trouble, the detail channel finds others in trouble (bad views 35%, pretrained 25% vs 12% base rate); the state channel does not (4%). |

The state channel finds *which situation* a record belongs to; the detail channel keeps *what it says*. A memory
should serve each question from the channel suited to it, and the predictor of how states move, the second half of
a world model, remains the open problem.

## Reproduce

Everything runs on a CPU. A full rerun of all experiments takes several hours on two cores; the figures and the
paper rebuild from `results/` in a minute.

```sh
pip install -r requirements.txt

# 1. Generate the company (needs Docker; Era's simulator images are pulled from Docker Hub, no account needed)
sh cwm/era_up.sh day1
python cwm/era_export.py --out data/era/day1
for p in 8080 8081 8082 8083 8084 8085 8087; do curl -s -XPOST localhost:$p/_admin/state/day2; done
python cwm/era_export.py --out data/era/day2

# 2. Version-1 experiments (each writes results/<name>.json)
STEPS=1500 python cwm/run_prh.py e1            # good vs bad views vs next-token (seed 0); caches embeddings
STEPS=1000 python cwm/run_prh.py e2            # separately trained encoders + one rotation
STEPS=1500 python cwm/run_prh.py e3            # a new quarter of data
python cwm/run_e3b.py                          # retraining (needs e1's cached encoder): alignment, old-index search
python cwm/run_extras.py extras_a seeds:good,bad
python cwm/run_extras.py extras_b mask:good noise:0.25,0.5 seeds:ntp
python cwm/run_extras.py extras_c heldout:base,good,bad
python cwm/run_extras.py extras_d heldout:ntp
python cwm/run_state.py                        # state tracking and next-quarter forecasting
python cwm/run_acl.py train:open train:open_clean train:account_executive train:support_engineer \
                      train:support_engineer_clean train:software_engineer train:financial_analyst eval
python cwm/run_memory2.py                      # memory-token reader, good-view store (seed 0)
SEEDS=1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_seeds python cwm/run_memory2.py

# 3. Version-2 experiments
python cwm/run_extras.py extras_e maskall:good # customers AND employees masked
python cwm/make_minilm.py                      # cache frozen all-MiniLM-L6-v2 features (about 10 minutes)
python cwm/run_v2.py embed2 pretrained monthly acl state time primitives situations scope acl_combo acl_pub
# memory stores, four seeds each: detail only, both channels concatenated, both channels routed
STORE=minilm SEEDS=0,1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_minilm4 python cwm/run_memory2.py
STORE=dual   SEEDS=0,1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_dual    python cwm/run_memory2.py
STORE=route  SEEDS=0,1,2,3 VARIANTS=top16_pretrained            SUFFIX=_route   python cwm/run_memory2.py

# 4. Figures and paper (every number in the paper is filled from results/ by cwm/fill_paper.py)
make all
```

Quick checks that need no Era data (a tiny synthetic company, the access policy, non-interference, the
good-view head, and rotation invariance of placement):

```sh
pytest -q tests
```

The exact company is fixed by the seed (42) and simulated date (2026-10-31) in `cwm/era_up.sh`. Generated data
is not included in this repository; regenerate it with the commands above. Runs are deterministic for a given seed
and thread count, not across thread counts; the version-2 runs used `THREADS=1 OMP_NUM_THREADS=1` on two cores.

## Layout

| Path | What it is |
| --- | --- |
| `cwm/era_up.sh`, `cwm/era_export.py` | Start Era's seven simulators and export every record into one table plus ground truth |
| `cwm/data.py` | Records to states (customer × quarter, month or ISO week), tokens, system families |
| `cwm/jepa.py` | Bag-of-n-grams encoder, LeJEPA loss with SIGReg, good and bad views, next-token baseline |
| `cwm/pretrained.py` | Frozen all-MiniLM-L6-v2 backbone and the good-view JEPA head trained on its features |
| `cwm/metrics.py` | Cross-system neighbour precision, probes, explained fractions, Procrustes, mutual kNN |
| `cwm/run_prh.py`, `cwm/run_e3b.py` | Main comparison, alignment across systems, behaviour over time |
| `cwm/run_extras.py` | Extra seeds, identity masking (customers; customers and employees), wrong links, held-out records |
| `cwm/run_state.py` | State tracking and forecasting probes |
| `cwm/acl.py`, `cwm/run_acl.py` | Access policy, per-reader encoders, leak tests, cost and non-interference |
| `cwm/memory.py`, `cwm/run_memory2.py` | Memory-token reader over the store: state-only, detail-only, concatenated and routed stores |
| `cwm/make_minilm.py` | Caches the frozen backbone's features for every linked record |
| `cwm/run_v2.py` | Version-2 experiments: pretrained baselines, monthly windows, access control with a shared backbone, two channels, dynamics, drift, misalignment, similar situations, scope diagnostic |
| `cwm/make_figures.py`, `cwm/fill_paper.py` | Figures, and every number and generated table in the paper from `results/` |
| `tests/` | Smoke tests run by GitHub Actions on every push |
| `results/` | Every result reported in the paper, as JSON (`v2_*.json`, `extras_e.json` and `memory2_{minilm4,dual,route}.json` are new in version 2) |
| `paper/` | LaTeX source (`main.in.tex` with placeholders, generated `main.tex`), figures, bibliography, PDF |
| `docs/REVIEW_RESPONSE.md` | Point-by-point answer to the review of version 1 |

## Moving to real data

Replace `era_export.py` with connectors for your own systems and keep the record schema
(`system, ts, author, thread, text` plus link fields such as `account_id`). Link fields only decide which records
are positives; the encoders never read them. Replace the role-based policy in `cwm/acl.py` with each system's own
permissions (channel membership, email recipients, record sharing); the three controls stay the same. A frozen
pretrained backbone may be shared across all access scopes because it holds no company data; heads, states and
indexes are per scope.

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
  note   = {Technical report, version 2}
}
```

## Acknowledgments

Era by Eon is free and made this work possible. SIGReg follows LeJEPA (Balestriero & LeCun, 2025); the
implementation here is our own. The memory reader follows Memorilla (Fallahpour et al., 2026). The pretrained
backbone is all-MiniLM-L6-v2 from Sentence-Transformers.
