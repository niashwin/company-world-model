# Toward World Models of Organizations

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

**Every system sees the same company.** Email, chat, the CRM, support tickets, call recordings, calendars and
issue trackers each record part of what is happening with every customer, deal and project, yet none of them
records the situation itself. This repository learns that missing piece: a shared **state space** in which every
record lands according to the situation it describes, whichever system wrote it, with **access control built in**.
It is the first layer of a **world model of an organization**.

![The idea](paper/figures/idea.png)

We expect every company to come to run on an *operational world model*: a model of how the company actually functions,
which people and AI agents can understand the company through, learn on the way a new employee learns it, and use
to try changes before making them. This repository builds the first layers of such a model and tests, honestly,
how far the rest of it gets on the same data.

The training signal comes for free. The Platonic Representation Theorem (Balestriero & LeCun, 2026) shows that
JEPAs trained on *good views*, two different observations of the same underlying state, converge to the same
representation up to a rotation. Organizations already record good views: records that different systems link to
the same customer in the same time window. We train small text JEPAs on those links, on a simulated company from
[Era by Eon](https://www.eon.io/blog/era-build-and-test-ai-agents), compare them with corruption-based views and
next-token prediction on identical text, and repeat the key comparisons with a frozen pretrained encoder.

**Paper:** [`paper/main.pdf`](paper/main.pdf) (version 4; source in [`paper/`](paper)).
**What changed and why:** [`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md) (every review, point by point) and
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

### The rest of a world model: a division of labor

| Question | Result |
| --- | --- |
| **Two channels** (state + detail) on the 144 memory questions, which name their customer, 4 seeds | Retrieval by content within the question's quarter. Detail only (pretrained) 71.2%, state only (good views) 65.8%, concatenated 72.2%, routed (retrieve by state, read detail) 70.1%. **No combination clearly beats the detail channel** (concatenated minus detail: +1.0 points, 95% interval −2.4 to +4.7). Both channels find the named customer's records equally often (65%), so the detail store's lead over the state store comes from reading, not finding. |
| **Questions that come with one record of the customer instead of its name** (127 questions, 4 seeds) | Names kept: **placing the record with the state channel** and reading its customer's records **63.8%**, placing it with the pretrained encoder 54.9% (**paired +8.9 points, 95% interval +3.9 to +13.8**), nearest neighbours of the record 58–60%, no store 52.2%, the record's own link (oracle) 68.7%. Placement with both channels concatenated 64.2%. Names masked: only the oracle helps clearly (67.1%); the state channel's neighbours help a little (57.7% vs 52.0%). Given the right records, reading only their system and time codes answers 67.5–68.1%, about as well as reading their content. |
| Two channels for state facts ("was there a sales call this quarter?") | Good views alone 72% (about the majority), with a pretrained detail channel 87%, with bad views 95%: the detail channel restores the facts, but no combination beats its better channel on average. |
| **Monthly windows** (405 states on day 1; 398 transitions through day 2) | Held-out records placed in the right customer-month 17% vs 13% for "customer known, busiest month": finer windows help a little. |
| Forecasting next month's facts | 46–54% for every representation vs 54% majority class and 72% for copying this month's facts: **no forecast**. |
| Linear dynamics (state → next state) | With its regularization tuned by nested cross-validation, a ridge map beats "no change" in every space (2–31% less error) but never by more than 3.4 points beyond one shrink-toward-the-mean factor: it learns mean reversion, not dynamics. |
| **Drift** (weekly state jump vs escalations and deal-stage changes, 279 event weeks) | AUROC 0.47–0.54, within about two standard errors of chance; record volume alone 0.57. |
| **Cross-team misalignment** (sales vs support state distance vs at-risk months) | AUROC 0.48–0.57, within noise. |
| **Decisions: action-conditioned dynamics** (six company actions per customer-month, read from the company's logs and Era's ground truth: sales calls, leaders' messages, CRM tasks, engineering escalations, support replies, outbound email) | This month's actions add nothing to predicting next month's state (error change −0.1% to +0.6% of the state-only error; positive is worse). In the link-trained space the state already encodes most of them (each action is predicted from the state with AUROC 0.63–0.88). Next month's actions lower the error by 9–19%, mostly as a description of next month's records: with next month's mix of systems in the model they add 3–5%. Next-month outcomes: actions add +0.02 to +0.26 AUROC, and adding support replies alone does as well as all six; replies mark months with tickets, and tickets persist. |
| **Decisions: around single actions** (14-day windows, records of the action's kind left out) | No action moves the state detectably more than matched times once each window's record count is accounted for: engineering escalations +0.009 (95% interval −0.003 to +0.017; +0.011 before the adjustment), sales calls 0.000, leaders' messages −0.024. Knowing the action does not help predict where the state goes. |
| **Decisions: simulating intervene vs not for customers in trouble** (29 customer-months) | The machinery works (similar states of other customers, split by what the company did), but the logs barely contain the alternative: the company intervened in 25 of 29. Among similar states, those where it intervened were in trouble 19–24% of the time vs 0.2–4% where it did not, in all four state spaces, so intervening appears to be followed by more urgent tickets (+29 points), as one would see if calls and escalations are made when problems are severe; estimates move with the state space and the outcome model attributes little to the decision. **Learning from decisions needs actions whose assignment is known.** |
| **Similar situations across customers** | 18% of the 5 nearest other-customer states share the situation vs 8% chance (every representation does about as well); still 17% vs 7% when restricted to other quarters. For customers in trouble, the detail channel finds others in trouble (bad views 35%, pretrained 25% vs 12% base rate); the state channel does not (4%). |

The experiments point to a division of labor. The **state space** can be learned from links alone, places records
without links, can be shared across encoders through rotations and can carry access control. A **memory** should find a
question's records through the state space (or by name, when the question has one) and read them from the store;
once the right records are found, their provenance carries most of the answer. A **predictor** of
how states change did not emerge from a space trained for static similarity, and conditioning it on the company's
logged actions does not change that. The company intervened in situations unlike those where it held back, so its logs
show the situations in which it acted rather than the effect of acting. A **decision layer** needs actions whose
assignment is known (logged with reasons, deliberately varied, or replayed in a simulator).

Every encoder and head we timed trains in **2–7 minutes on a CPU**, without a GPU, so at this scale a company could
retrain its state space and each access scope's model daily, or whenever an access rule changes. We have not tested
how training time grows with millions of records.

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

# 4. Version-3 experiments
# the named-question stores again, retrieving by content within the question's quarter (Table 7)
for s in jepa_good minilm dual route; do
  KEYS=quarter STORE=$s SEEDS=0,1,2,3 VARIANTS=closed_book,top16_pretrained SUFFIX=_q_${s#jepa_} python cwm/run_memory2.py
done
python cwm/memory_retrieval.py                 # which records those stores retrieve
# questions that come with a record of the customer instead of its name (Table 8)
CORPUS=masked PREP=1 python cwm/run_memory_anchor.py   # state encoder (held-out threads hidden) and features
CORPUS=plain  PREP=1 python cwm/run_memory_anchor.py
CORPUS=masked python cwm/run_memory_anchor.py          # all eleven stores, four seeds
CORPUS=plain  python cwm/run_memory_anchor.py

# 5. Version-4 experiments: decisions (actions, action-conditioned dynamics, event windows, decision simulation)
python cwm/run_actions.py table dynamics events simulate

# 6. Figures and paper (every number in the paper is filled from results/ by cwm/fill_paper.py)
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
| `cwm/run_memory_anchor.py` | Questions that come with a record of the customer instead of its name: nearest-neighbour, placement and link-oracle stores, names masked or kept |
| `cwm/memory_retrieval.py` | Which records the named-question stores retrieve |
| `cwm/run_actions.py` | Decisions: the company's actions per customer and month, action-conditioned dynamics, event windows, decision simulation |
| `cwm/make_minilm.py` | Caches the frozen backbone's features for every linked record |
| `cwm/run_v2.py` | Version-2 experiments: pretrained baselines, monthly windows, access control with a shared backbone, two channels, dynamics, drift, misalignment, similar situations, scope diagnostic |
| `cwm/make_figures.py`, `cwm/fill_paper.py` | Figures, and every number and generated table in the paper from `results/` |
| `tests/` | Smoke tests run by GitHub Actions on every push |
| `results/` | Every result reported in the paper, as JSON (`v2_*.json`, `extras_e.json` and `memory2_{minilm4,dual,route}.json` are new in version 2; `memory_anchor_*.json`, `memory2_q_*.json` in version 3; `actions.json` in version 4; files ending `_a`, `_b`, ... are parallel runs merged by `fill_paper.py`) |
| `paper/` | LaTeX source (`main.in.tex` with placeholders, generated `main.tex`), figures, bibliography, PDF |
| `docs/REVIEW_RESPONSE.md` | Point-by-point answers to the reviews of versions 1 and 2 |
| `docs/ARXIV_ABSTRACT.txt` | A shorter abstract that fits arXiv's 1,920-character limit |

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
  note   = {Technical report, version 4}
}
```

## Acknowledgments

Era by Eon is free and made this work possible. SIGReg follows LeJEPA (Balestriero & LeCun, 2025); the
implementation here is our own. The memory reader follows Memorilla (Fallahpour et al., 2026). The pretrained
backbone is all-MiniLM-L6-v2 from Sentence-Transformers.
