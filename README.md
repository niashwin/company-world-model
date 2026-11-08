# Toward World Models of Organizations

[![tests](../../actions/workflows/tests.yml/badge.svg)](../../actions/workflows/tests.yml)

**Every system sees the same company.** Email, chat, the CRM, support tickets, recorded calls, calendars and issue
trackers each record part of what happens with every customer, deal and project. But no system records the
situation itself. People rebuild that situation in their heads, and AI assistants rebuild it from documents for
each question. This repository learns it: one **state space** for the whole company, trained on the links that the
systems already keep, with **access control built in**. It is the first learned layer of a **world model of an
organization**.

![The idea](paper/figures/idea.png)

## Summary

A recent theorem predicts that encoders trained to predict one observation of a state from another converge to one
representation (Balestriero & LeCun, 2026). Records that different systems link to the same customer in the same
period are such observations. So the links that a company already keeps are a free training signal. We test this on
a simulated company from [Era by Eon](https://www.eon.io/blog/era-build-and-test-ai-agents), with exact ground truth.

- Small encoders trained on these links place **76%** of records with withheld links with the right customer. Text
  similarity places 53%, and a guess of the largest customer 12%.
- With every name and identifier masked, the encoders still place 33% of these records. A pretrained encoder places
  20%.
- The same links raise a frozen pretrained encoder from 53% to 62%.
- Permissions are part of the model. When we plant secrets in records that a reader may not read, no result that
  the reader gets changes.
- The state space indexes a memory. Some questions come with one record of the customer instead of its name. With
  the names kept in the records, they get the right answer 63.8% of the time. With a pretrained encoder, they get
  54.9%, and without a store 52.2%.
- On the same space, we build a first decision layer. We also show what data it needs to learn the effects of
  decisions.
- Each encoder and head that we timed trains in 2–7 minutes on a CPU.

**Paper:** [`paper/main.pdf`](paper/main.pdf), version 6, in the format of a Nature article. The LaTeX source is in
[`paper/`](paper).
**What changed and why:** [`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md) (every review, point by point),
[`docs/CHECKS.md`](docs/CHECKS.md) (how we checked version 6) and [`CHANGELOG.md`](CHANGELOG.md).

## What this means for corporate AI

Corporate AI is AI that understands how a company works, acts within its permissions and helps it decide. It needs
a model of the company itself, which stays current as the company works. We see four layers. Each layer is useful on
its own, and each one makes the next one possible.

| Layer | What it gives | Status |
| --- | --- | --- |
| Memory that accumulates | What happened, who decided it and why, and where each fact came from | Organizational memory products, such as Sentra's, build this today |
| **Representations that learn** | **One state space for the company, which places each record with its customer and finds similar situations** | **This work, shown in simulation** |
| Models that predict | How situations change, and what decisions do | The next layer. This work builds a first version and shows what data it needs |
| Organizations that improve | People and agents test decisions against the model, and the model learns from what happens | Future |

The learned state space does not replace the memory. The memory keeps the facts, sources and permissions. The state
space adds two things: a place for records without a link, and a measure of which situations are alike.

## How it compares

Enterprise AI gives models knowledge of a company in several ways. They are layers of one architecture more than
rivals. Table 1 of the paper gives this comparison with references.

| Approach | What it keeps | Where it is strong | What it leaves open |
| --- | --- | --- | --- |
| Retrieval (RAG) | Documents and a search index | Current and simple, with permission filters on results | Builds each situation again for each question |
| Explicit memory: graphs, ledgers and memory stores (GraphRAG, Zep, MemGPT, Mem0) | Entities, facts, decisions, dates and sources | Facts with sources, easy to audit and update | Facts, not a learned state of each situation |
| Compressed memory ([Memorilla](https://arxiv.org/abs/2610.21375)) | Embeddings of documents, read as 16 memory tokens for each question by a frozen model | A large collection in a few tokens (84.33% on FactKG, against 80.43% with the full context) | Organizes documents, not situations |
| Trained context and skills (Cartridges, [Engram](https://engram.com/blog/learning-in-a-world-of-notes)) | A trained cache of the documents, or notes and a model trained to use them | Better and cheaper answers (Engram reports 7 to 20 points from training to use notes) | Organizes text, not situations |
| Process mining | Event logs that a case number links | Shows how work flows | Needs structured events, not text |
| World models of physical scenes (V-JEPA 2, H-JEPA) | Latent states of a scene | Prediction and planning | Built for sensors, not for records and permissions |
| **This work** | **A learned state of each situation, from the links between systems** | **Places records whose links it never saw; permissions built in** | **Prediction of change, and the effects of decisions; tested on one simulated company** |

Our memory reader follows Memorilla in a small form, so a reader of this kind could read situations instead of
separate documents. Agents trained to use notes could act on a shared state that respects permissions. We tested
only the first of these combinations.

## Website

The interactive project website is in [`website/`](website). GitHub Pages publishes it with
[`.github/workflows/pages.yml`](.github/workflows/pages.yml). Its address will be
[https://niashwin.github.io/company-world-model/](https://niashwin.github.io/company-world-model/).
Turn on GitHub Pages one time: in Settings > Pages, set Source to GitHub Actions. A GitHub Pages site is public on
the internet, also when the repository is private. For a private repository, Pages needs a paid GitHub plan.

To see the website on your computer, run this command. Then open [http://localhost:8000](http://localhost:8000).

```sh
python -m http.server -d website 8000
```

The repository includes the data of the website, so the site works from a clone. To make the data again, run
`python cwm/export_site_data.py` after Path 2 below. It reads `results/`, the filled paper (`paper/main.tex` and
`paper/main.aux`) and the stored embeddings, from which it computes the maps.

## Results at a glance

We use one simulated company. It has seven systems, 21,972 customer-linked records, 21 active customers and exact
ground truth. Every number below is in the paper and in `results/`. Each result uses one seed, unless we give a
different number of seeds.

### The state space (what works)

| Test | Good views | Bad views | Next-token | TF-IDF | Pretrained encoder | Pretrained + good-view head |
| --- | --- | --- | --- | --- | --- | --- |
| **Held-out records** (links hidden in training): placed with the right customer | **76%** | 19% | 20% | 53% | 53% | 62% |
| Training records: 10 nearest neighbours **from other systems** on the same customer | **95%** (3 seeds) | 14% | 13% | 35% | 40% | 72% |
| Held-out records with **every name and identifier masked**: placed with the right customer | **33%** | | | | 20% | |
| ...with customer names, contacts, domains and ids masked (training records) | **48%** | | | 14% | | |
| ...with **employee** names, handles and chat names masked as well | **47%** | | | 14% | | |
| ...when 25% / 50% of the training links are wrong | **84% / 68%** | | | | | |
| Encoders trained **separately** on messages and on systems of record, one rotation from 64 anchors: same-state recall@1 (chance 4%) | **21%** | 4% | 4% | | | |
| **Retrained** encoder that searches the **old** index through one rotation: same-customer precision@10 | **79%** (3% without rotation) | | | | | |

- The held-out test is the main one: the encoder never saw the links of these records. A guess of the largest
  customer places 12% of them.
- The same good views lift a frozen pretrained sentence encoder (all-MiniLM-L6-v2) from 53% to 62% on held-out
  records. So the recipe also improves a ready-made backbone. On this company it stays below the small encoder
  trained from scratch, which learns the company's own vocabulary.
- Masking employees as well as customers changes almost nothing (48% → 47%). So the residual signal is not "which
  employee handles this customer".

### Access control, built in

A learned world model can leak across access boundaries through three paths. They are search results, states
aggregated over records that a reader may not open, and the encoder's own training. `cwm/acl.py` gives every record
its readers. The model closes all three paths. So what a reader receives depends only on what they may read
(non-interference, checked byte for byte).

- **A search filter is not enough.** A support engineer who may not read CRM records searches for a codename that we
  planted in them. With an encoder trained on everything, the search returns the right customer **100%** of the
  time. This happens even though we filter the search to readable records. With access control built in, every
  result is identical with and without the planted secrets. This is true for the from-scratch encoder and for the
  pretrained backbone with per-reader heads.
- **The cost, and what we can recover while the guarantee holds** (held-out readable records placed with the right
  customer):

| Reader (share readable) | Own encoder | Backbone + own head | Both side by side | Trained on everything (leaks) |
| --- | --- | --- | --- | --- |
| CEO (100%) | **76%** | 62% | | 76% |
| Account executive (81%) | **76%** | 70% | | 78% |
| Support engineer (86%) | **74%** | 61% | | 75% |
| Software engineer (22%) | 28% | 29% | **32%** | 66% |
| Financial analyst (9%) | 23% | **43%** | 30% | 65% |

  A pretrained backbone holds no company data, so every reader may share it. It helps the financial analyst
  substantially (23% → 43%, against 65% for the encoder that leaks). But it barely helps the software engineer, whose records are 87% support tickets. A base trained on
  the 534 linked records that everyone may read does not help.

  **Rotations cannot help by construction.** They make the views of different readers comparable. But no rotation
  changes nearest-centroid placement (unit-tested). Also, a rotation toward a space that the reader may not read
  would itself depend on records outside their rights.

  A size-matched random sample of all systems places 42% and 31% of the two readers' records. So most of the gap to
  "trained on everything" comes from the records that a reader may not read at all. Knowledge from outside the
  company can close part of the gap, as the backbone does for the analyst. Only a change in the access policy can
  close the rest.

### Toward prediction and decisions

| Question | Result |
| --- | --- |
| **Two channels** (state + detail) on the 144 memory questions, which name their customer, 4 seeds | The stores retrieve by content within the question's quarter. Detail only (pretrained) 71.2%, state only (good views) 65.8%, concatenated 72.2%, routed (retrieve by state, read detail) 70.1%. **No combination clearly beats the detail channel** (concatenated minus detail: +1.0 points, 95% interval −2.4 to +4.7). Both channels find the named customer's records equally often (65%). So the lead of the detail store comes from how the reader reads its records, or from how we embed the question. The detail store does not find more of the right records. |
| **Questions that come with one record of the customer instead of its name** (127 questions; 4 seeds of the reader and anchors; 1 state encoder) | Names kept: **the state channel places the record**, then the reader reads its customer's records and answers **63.8%**. Placement with the pretrained encoder: 54.9% (**paired +8.9 points, 95% interval +3.9 to +13.8**). Nearest neighbours of the record 58–60%, no store 52.2%, the record's own link (oracle) 68.7%, placement with both channels concatenated 64.2%. Names masked: the state channel's neighbours, read by detail, answer 57.7% against 52.0% without a store (paired +5.7 points, 95% interval +0.4 to +10.8). The oracle answers 67.1%. Given the right records, the reader answers 67.5–68.1% from only their system and time codes, about as well as from their content. |
| Two channels for state facts ("was there a sales call this quarter?") | Good views alone: 72% (about the majority). With a pretrained detail channel: 87%. With bad views: 95%. The detail channel restores the facts. But no combination beats its better channel on average. |
| **Monthly windows** (405 states on day 1, 398 transitions through day 2) | Held-out records placed in the right customer-month: 17%, vs 13% for "customer known, busiest month". Finer windows help a little. |
| Forecasting next month's facts | 44–54% for every representation, vs 54% for the majority class and 72% if we copy this month's facts. A forecast needs training for change. |
| Linear dynamics (state → next state) | We tune the regularization of a ridge map by nested cross-validation. Then the map beats "no change" in every space (2–31% less error). But it never beats one shrink-toward-the-mean factor by more than 3.4 points. Most of what it learns is a return toward the mean. |
| Drift and cross-team misalignment (weekly state jumps; distance between sales and support states) | As signals of escalations, deal-stage changes or at-risk months, AUROC 0.47–0.57. Record volume alone gives 0.57. |
| **Decisions: action-conditioned dynamics.** Six company actions per customer-month, which we read from the company's logs and Era's ground truth. They are sales calls, leaders' messages, CRM tasks, engineering escalations, support replies and outbound email. | This month's actions add nothing to predictions of next month's state (error change −0.1% to +0.6% of the state-only error, where positive is worse). In the good-view space, the state already encodes most of them: we predict each action from the state with AUROC 0.63–0.88. Next month's actions lower the error by 9–19%, mostly as a description of next month's records. With next month's mix of systems in the model, they add 3–5%. Next-month outcomes: actions add +0.02 to +0.26 AUROC, and support replies alone do as well as all six. Replies mark months with tickets, and tickets persist. |
| **Decisions: around single actions** (14-day windows, records of the action's kind left out) | After an adjustment for the record count of each window, no action moves the state detectably more than matched times. Engineering escalations +0.009 (95% interval −0.003 to +0.017, +0.011 before the adjustment), sales calls 0.000, leaders' messages −0.024. Knowledge of the action does not help to predict where the state goes. |
| **Decisions: a simulation of intervention vs no intervention for customers in trouble** (29 customer-months) | The machinery works: it finds similar states of other customers and splits them by what the company did. But the logs barely contain the alternative: the company intervened in 25 of 29. In all four state spaces, similar states where it intervened were in trouble 19–24% of the time, vs 0.2–4% where it did not. So more urgent tickets appear to follow interventions, as we would see if calls and escalations come when problems are severe. The estimates move with the state space: +12 to +15 points for deals and +8 to +29 for urgent tickets. Six of the eight intervals include zero. The outcome model attributes little to the decision. **Learning from decisions needs actions with a known assignment.** |
| **Similar situations across customers** | Of the 5 nearest states of other customers, 18% share the situation, vs 8% by chance. Every representation does about as well. When we restrict the neighbours to other quarters, it is still 17% vs 7%. Good views make the state sensitive to whose situation it is. So for "customers in trouble", representations that keep detail are the right tool. Bad views find 35%, pretrained 25% and good views 4%, against a 12% base rate. The two channels concatenated find 5%, so each channel should answer its own kind of query. |

The experiments point to a division of labor. We can learn the **state space** from links alone. It places records
whose links it never saw. Through rotations, different encoders can share it. It can also carry access control.

A **memory** should find a question's records through the state space, or by name when the question has one. It
should then read them from the store. For questions about how many tickets there were or whether a call took place,
the provenance of the right records carries most of the answer.

A **predictor** of how states change needs training for change. A space trained for similarity gives little of it on
its own, also with the company's logged actions. The company intervened in situations unlike those where it did not
intervene. So its logs show the situations in which it acted, rather than the effect of its actions. A
**decision layer** needs actions with a known assignment (logged with reasons, deliberately varied, or replayed in a
simulator).

Every encoder and head that we timed trains in **2–7 minutes on a CPU**, without a GPU. So at this scale, a company
could retrain its state space and each access scope's model daily, or whenever an access rule changes. We did not
test how training time grows with millions of records.

## Reproduce

Everything runs on a CPU. There are two paths. [`REPRODUCE.md`](REPRODUCE.md) gives the details. It also has a
table that maps each part of the paper to its command and result file.

**Path 1** rebuilds the figures and the paper from the stored results in `results/`, in a minute:

```sh
pip install -r requirements.txt
make all            # figures, then the results from results/ into the paper, then the PDF
pytest -q tests     # checks that need no Era data
```

If your system has `python3` but no `python`, run `make all PY=python3`.

**Path 2** runs every experiment again, from the simulated company to the PDF and the website. It takes several
hours on two cores. It needs Docker. You need no Docker Hub account to get Era's simulator images.

```sh
bash scripts/reproduce.sh
```

`cwm/fill_paper.py` writes the results into the paper from `results/`, and `tests/test_paper_numbers.py` checks the
numbers that the text states directly. `pytest -q tests` needs no Era data. It checks a tiny synthetic company, the
access policy, non-interference, the good-view head and the rotation invariance of placement.

The seed (42) and the simulated date (2026-10-31) in `cwm/era_up.sh` fix the exact company. Runs are deterministic
for a given seed and thread count, but not across thread counts. The version-2 runs used
`THREADS=1 OMP_NUM_THREADS=1` on two cores. `scripts/reproduce.sh` uses these settings by default.

## Writing standard

Since version 5, the paper, this README and the website follow the rules of ASD-STE100 (Simplified Technical English).
Our target is that at least 90% of the sentences comply. `tools/ste_check.py` measures this. It checks each sentence
for the main rules: length, active voice, simple tenses and no -ing forms. The score is the share of sentences that
comply. It is 98% for the paper and more than 99% for this README and the website.

The tool cannot check the approved dictionary of STE, so we use the technical terms of the field where we must. To
measure the paper, run:

```sh
python tools/ste_check.py paper/main.tex
```

`make check` runs the tests and fails if the share is less than 90%. `tools/tex_invariants.py` checks that an
edit of the paper keeps every placeholder, label, reference and citation.

## Layout

| Path | What it is |
| --- | --- |
| `cwm/era_up.sh`, `cwm/era_export.py` | Start Era's seven simulators and export every record into one table plus ground truth |
| `cwm/corpus_stats.py` | Counts the records: Extended Data Table 1 and the counts in Methods (`results/corpus.json`) |
| `cwm/data.py` | Records to states (customer × quarter, month or ISO week), tokens, system families |
| `cwm/jepa.py` | Bag-of-n-grams encoder, LeJEPA loss with SIGReg, good and bad views, next-token baseline |
| `cwm/pretrained.py` | Frozen all-MiniLM-L6-v2 backbone and the good-view JEPA head trained on its features |
| `cwm/metrics.py` | Cross-system neighbour precision, probes, explained fractions, Procrustes, mutual kNN |
| `cwm/run_prh.py`, `cwm/run_e3b.py` | Main comparison, alignment across systems, behaviour over time |
| `cwm/run_extras.py` | Extra seeds, identity masking (customers, or customers and employees), wrong links, held-out records |
| `cwm/run_state.py` | State tracking and forecasting probes |
| `cwm/acl.py`, `cwm/run_acl.py` | Access policy, per-reader encoders, leak tests, cost and non-interference |
| `cwm/memory.py`, `cwm/run_memory2.py` | Memory-token reader over the store: state-only, detail-only, concatenated and routed stores |
| `cwm/run_memory_anchor.py` | Questions that come with a record of the customer instead of its name: nearest-neighbour, placement and link-oracle stores, names masked or kept |
| `cwm/memory_retrieval.py` | Which records the named-question stores retrieve |
| `cwm/run_actions.py` | Decisions: the company's actions per customer and month, action-conditioned dynamics, event windows, decision simulation |
| `cwm/make_minilm.py` | Caches the frozen backbone's features for every linked record |
| `cwm/run_v2.py` | Version-2 experiments: pretrained baselines, monthly windows, access control with a shared backbone, two channels, dynamics, drift, misalignment, similar situations, scope diagnostic |
| `cwm/make_figures.py`, `cwm/fill_paper.py` | `make_figures.py` draws the figures. `cwm/fill_paper.py` writes the results into the paper from `results/`, and `tests/test_paper_numbers.py` checks the numbers that the text states directly. |
| `cwm/export_site_data.py` | Rebuilds the data of the website (`website/data/site-data.js`) from `results/` and the filled paper, and writes `website/index.html` from `website/template.html` |
| `scripts/reproduce.sh` | Runs every experiment again, from the simulated company to the PDF and the website, in stages (`data`, `v1` to `v4`, `paper`, `site`) |
| [`REPRODUCE.md`](REPRODUCE.md) | How to reproduce the paper, by two paths. A table maps each part of the paper to its command and result file. |
| `tests/` | Smoke tests that GitHub Actions runs on every push |
| `tools/ste_check.py`, `tools/tex_invariants.py` | `ste_check.py` measures how closely a text follows ASD-STE100. `tex_invariants.py` checks that an edit of the paper keeps every placeholder, label, reference and citation. |
| `tools/make_og_image.py` | Renders `website/assets/og.png`, the image that link previews of the website show |
| `results/` | Every result that the paper reports, as JSON. New in version 2: `v2_*.json`, `extras_e.json` and `memory2_{minilm4,dual,route}.json`. New in version 3: `memory_anchor_*.json` and `memory2_q_*.json`. New in version 4: `actions.json`. Files that end in `_a`, `_b`, ... are parallel runs, and `fill_paper.py` merges them. |
| `paper/` | LaTeX source (`main.in.tex` with placeholders, generated `main.tex`), figures, bibliography, PDF |
| `website/` | The project website. `template.html` is the source of the page, and `cwm/export_site_data.py` generates `index.html` from it. |
| `.github/workflows/pages.yml` | Publishes `website/` and the PDF of the paper with GitHub Pages |
| `docs/REVIEW_RESPONSE.md` | Point-by-point answers to every review, from version 1 to version 6 |
| `docs/CHECKS.md` | The checks of version 6 and their results: tests of the numbers, a clean rebuild, the STE measure and the independent reviews |
| `docs/ARXIV_ABSTRACT.txt` | A shorter abstract that fits arXiv's 1,920-character limit |

## Moving to real data

Replace `era_export.py` with connectors for your own systems. Keep the record schema:
`system, ts, author, thread, text`, plus link fields such as `account_id`. Link fields only decide which records
are positives. The encoders never read them.

Replace the role-based policy in `cwm/acl.py` with each system's own permissions (channel membership, email
recipients, per-record access). The three controls stay the same. All access scopes may share a frozen pretrained
backbone, because it holds no company data. Heads, states and indexes are per scope.

## Data and licence

Era by Eon is free and generates a synthetic company. No real person, customer or conversation is part of it. We do
not redistribute the generated data here. Regenerate it with `cwm/era_up.sh` and `cwm/era_export.py`, or with
`bash scripts/reproduce.sh data`. Check Era's terms before you share it. The MIT licence applies to the code (see
`LICENSE`).

## Citation

```bibtex
@techreport{gopinath2026organizations,
  title       = {Toward World Models of Organizations: Every System Sees the Same Company},
  author      = {Gopinath, Ashwin},
  institution = {Sentra},
  address     = {San Francisco},
  year        = {2026},
  month       = nov,
  note        = {Version 6},
  url         = {https://github.com/niashwin/company-world-model}
}
```

## Acknowledgments

Era by Eon is free and made this work possible. SIGReg follows LeJEPA (Balestriero & LeCun, 2025). The
implementation here is our own. The memory reader follows Memorilla (Fallahpour et al., 2026). The pretrained
backbone is all-MiniLM-L6-v2 from Sentence-Transformers.
