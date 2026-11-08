# Changelog

## v5 (November 2026)

- **Simplified Technical English.** The paper, the README and the website follow the writing rules of ASD-STE100:
  short sentences, active voice, simple tenses and one term for each concept. The target was at least 90% of the
  sentences. `tools/ste_check.py` measures 98% for the paper. `tools/tex_invariants.py` checks that an edit keeps
  every placeholder, label, reference and citation. The results did not change. Independent reviews of the text
  against `results/` led to corrections of some wording and some claims; `docs/CHECKS.md` lists them. One term for
  each concept: "good views" replaces "link-trained views" of versions 2 to 4.
- **Checks** (`docs/CHECKS.md`). A record of each check of version 5 and its result: the tests of the numbers, the
  clean rebuild, the measure of the writing, and the independent reviews with what they found.
- **Project website** (`website/`). An interactive site with the paper's own numbers: the records of one customer
  in seven systems, the state-space map in six spaces, the leak test, the two memory channels, dynamics, and the
  decision simulation. `cwm/export_site_data.py` writes its data from `results/` and the filled paper.
  `.github/workflows/pages.yml` publishes it with GitHub Pages. The charts draw at the width of the screen, so they
  stay legible on phones, and the site has light and dark themes.
- **Reproduction from a fresh clone.** `REPRODUCE.md` maps each table and figure to its command and result file.
  `scripts/reproduce.sh` runs every stage, from the simulated company to the PDF. `cwm/corpus_stats.py` writes
  Table 1 and the counts of Section 5 to `results/corpus.json`, so `cwm/fill_paper.py` generates Table 1.
  `tests/test_paper_numbers.py` checks each number that the text of the paper states directly against `results/`.
- **A check of months after the export** (`run_actions.py checks`, Appendix B). One of the 396 transitions leads
  into February 2027, after the export ends. Appendix B gives the observed rates of Table 13 without it.
- **Reproducible builds.** The figures and the PDF have no timestamps, so the same sources give the same files.
  A rebuild of every figure from the caches gives images identical to the stored ones.
- The range of training times (2 to 7 minutes) now counts one encoder or head per timing. `cwm/fill_paper.py` leaves
  out the timing of the alignment experiment, which trains two encoders together.
- Removed `results/memory2_minilm.json`, which no part of the paper used.

## v4 (November 2026)

Answers the "corporate AI" review ([`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md)).

- **Decisions** (`cwm/run_actions.py`, §6.11, Tables 12 and 13). Six company actions per customer and month, read
  from the records and Era's ground truth (sales calls, leaders' messages, CRM tasks, engineering escalations, support
  replies, outbound email), and outcomes (deal progress, urgent tickets). The dynamics, outcome and simulation
  analyses run in four representations (link-trained views, bad views, the pretrained encoder, both channels); the
  event windows use link-trained views.
  - Action-conditioned dynamics: this month's actions add nothing to next month's state; in the link-trained space
    the state already encodes most of them (AUROC of each action from the state). Next month's actions lower the
    error by 9–19%, mostly as a description of next month's records: with next month's mix of systems in the model
    they add 3–5%. Shuffled-action and system-mix controls are included.
  - Next-month outcomes: adding support replies alone to the state does as well as adding all six actions; replies
    mark months with tickets, and tickets persist.
  - Event windows of 14 days around single actions, against matched times, with an adjustment for each window's
    record count: no action moves the state detectably more than matched times once counts are accounted for.
  - Decision simulation for 29 customer-months in trouble: case-based in four state spaces, unmatched, and
    model-based, with intervals that resample customers, repeating the neighbour search and refitting the outcome
    model; balance of the matched groups in all four spaces, propensity of intervening from the state, and the timing
    of interventions relative to the first urgent ticket. The company intervened in 25 of 29, in situations unlike
    those where it held back, so the estimates reflect the situations in which it acted, not what the action did.
- **Abstract, introduction, §7 and conclusion reframed** around corporate AI, with training costs as measured (every
  encoder and head we timed trains in 2–7 minutes on a CPU; `run_acl.py eval` now records each reader encoder's
  training time) and the decision layer's requirement: actions whose assignment is known.
- Typographic minus signs for every generated number.

## v3 (November 2026)

Answers the review of v2 ([`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md)).

- **Questions that come with a record of the customer instead of its name** (`cwm/run_memory_anchor.py`, Table 8,
  §6.8). Each question is anchored by one record from a held-out thread and another system; names masked or kept.
  Records are found by nearest neighbours, by placement at the nearest customer, or by the anchor's link (oracle).
  With names kept, placing the record with the state channel answers 63.8% against 54.9% with the pretrained encoder
  (paired +8.9 points, 95% interval +3.9 to +13.8), 52.2% with no store and 68.7% for the oracle. With names masked,
  only the oracle helps clearly. Reading only the found records' system and time codes does about as well as reading
  their content.
- **Memory retrieval fixed.** Store keys used to be whole rows, in which a time code outweighed the content; retrieval
  now ranks records of the question's quarter by content alone (`KEYS=quarter` in `run_memory2.py`). Table 7 rerun:
  detail only 71.2%, state only 65.8%, concatenated 72.2%, routed 70.1%; conclusions unchanged.
- **Abstract, introduction, §7 and conclusion reframed** as a division of labor: a state space, a memory that finds
  through the state space and reads from the store, and dynamics that did not emerge from static similarity.
- `memory.py`: retrieval keys separate from content (`split`), per-question correctness for paired comparisons, and
  pre-training queries that see the sampled record (restricted to training links in the anchored test);
  `fill_paper.py` reports paired bootstrap intervals and merges parallel result files; a test for content/key
  splitting; a short abstract for arXiv's form.

## v2 (November 2026)

Answers a review of v1; [`docs/REVIEW_RESPONSE.md`](docs/REVIEW_RESPONSE.md) goes through each point and its result.

- **Pretrained baseline.** A frozen all-MiniLM-L6-v2 out of the box and with a small JEPA head trained on good
  views (`cwm/pretrained.py`, `cwm/make_minilm.py`, `run_v2.py pretrained`), added to Tables 2 and 3. Good views
  raise it from 40% to 72% (training records) and from 53% to 62% (held-out records).
- **Employee masking.** Masking repeated with the company's own employees masked too (`run_extras.py extras_e
  maskall:good`): 47% against 48% with customers only.
- **Access control for readers who may read little.** Per-reader heads on a shared pretrained backbone, a base
  trained on records everyone may read, both channels per reader, and a scope diagnostic (`run_v2.py acl acl_combo acl_pub
  scope`). The financial analyst rises from 23% to 43% and the software engineer from 28% to 32%, with zero leakage
  and exact non-interference. Rotations are shown, and unit-tested, to leave placement accuracy unchanged.
- **Two channels, state and detail.** Concatenated and routed stores for the memory reader (`run_memory2.py` with
  `STORE=dual` and `STORE=route`, four seeds each, plus a four-seed detail-only store), and two-channel state probes
  (`run_v2.py state`). Neither combination beats the detail channel alone on the 144 questions.
- **Finer time.** Customer-month states (held-out placement, next-month forecasting, linear dynamics with a
  shrink-to-mean control) and customer-week states for drift (`run_v2.py monthly time`). Mostly negative.
- **Operational primitives.** Drift against escalations and deal-stage changes, cross-team misalignment, and similar
  situations across customers with time and "trouble" controls (`run_v2.py primitives situations`). Only similar
  situations works, at about twice chance.
- Paper: new method paragraphs, five new tables, updated Figure 8, rewritten abstract, contributions, discussion,
  limitations and conclusion; `\hyphenation{rotation rotations}`; two new references.
- Repository: consolidated README, review response, two new tests (good-view head; rotation invariance of
  placement), `fill_paper.py` exits with an error if any number is left unfilled.

## v1 (November 2026)

First version: good views from cross-system links, held-out records, identity masking, wrong links, alignment by
one rotation, retraining without re-embedding, state persistence, state probes, memory reader, and access control
built in (planted secrets, cost per reader, non-interference).
