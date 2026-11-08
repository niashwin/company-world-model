# Changelog

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
