# Changelog

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
