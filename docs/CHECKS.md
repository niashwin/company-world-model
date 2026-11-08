# Checks of version 5

This file records the checks that we did before the release of version 5, and what they found. Each check has a
command, so that you can do it again. [`REPRODUCE.md`](../REPRODUCE.md) tells how to make every result again.

## 1. The numbers in the paper

- `cwm/fill_paper.py` writes every result into the paper from `results/`. It stops with an error if a placeholder
  has no value. The last run of version 5 left no placeholder without a value.
- `tests/test_paper_numbers.py` checks the numbers that the text of the paper states directly, against `results/`.
  Examples are the TF-IDF and random rows of Table 2, the baselines of Section 6.1 and the identity ceiling of
  16.3%. Other tests cover the explained fractions, the alignment, the new quarter and the retrained
  encoder. The last tests cover persistence, the majority classes, the state facts, the transitions and the new
  records of day 2.
- `pytest -q tests` runs these 10 tests and 7 smoke tests. All 17 pass.

## 2. A clean rebuild

We copied the files that git tracks to an empty folder, without data and without caches. Then we ran
`make all PY=python3`, `make clean` and `make all PY=python3` again.

- Both builds gave a PDF that is identical, byte for byte, to the PDF in the repository. The PDF has 42 pages,
  no overfull boxes and no undefined references.
- The figures that need no cache (Figures 1, 2, 4, 5, 7 and 8 of the paper) came out identical to the stored
  files. The two figures that need the caches (the maps and the persistence plot) use the stored files.
- The paper and its figures have no dates or random identifiers, so the same sources give the same files.

## 3. The writing

`python tools/ste_check.py FILE` measures how many sentences follow the main rules of ASD-STE100. A sentence must
have at most 25 words, active voice, a simple tense and no -ing forms. A paragraph must have at most six sentences.
The target was 90%.

| Text | Sentences | Share that complies |
| --- | ---: | ---: |
| `paper/main.tex` | 1,450 | 98.2% |
| `README.md` | 246 | 100% |
| `website/template.html` | 267 | 99.3% |
| `REPRODUCE.md` | 90 | 96.7% |
| `docs/CHECKS.md` | 60 | 100% |

The tool cannot check the approved dictionary of STE. We kept the technical terms of the field where no simple
word has the same sense. `tools/tex_invariants.py` compared the paper before and after the rewrite: every placeholder,
label, reference and citation stayed.

## 4. Independent reviews

Separate agents reviewed the text. They did not write it. Each one had the results, the code and the paper, and
reported each problem with a quote. We fixed every problem that they found.

| Review | What it found | Examples of fixes |
| --- | --- | --- |
| Paper against the results (first) | Sentences that were stale after earlier versions, and claims that were not exact | A stale count in the caption of Table 7; the CEO's placement with the backbone; the description of the abstract's test with hidden links |
| Paper against the results (second) | Claims stronger than the results, and conditions that the text did not state | The tuned linear predictor beats shrinkage by at most 3.4 points; a condition that names stay in the records; the decision simulation is observational |
| A fresh user who follows `REPRODUCE.md` | Three defects and errors in the documents | `.gitignore` excluded the website's data; stored parts of a run overwrote a new single run; a figure had a date in it |
| The website against the paper | Claims stronger than the paper, and text that was too small on phones | The backbone helps one of two readers; the oracle line is not an upper limit; charts now draw at their real width |
| The website and the README against the paper (last) | Conditions that the summaries dropped, and two claims that the paper does not make | The 63.8% names its condition (names kept); the detail store does as well as any combination; next-month facts are 44–54%, not 46–54% |

The last review also measured the map of Figure 2 of the website. It took the ten nearest records from other
systems on the map, and counted how many belong to the same customer. The share is 90% for good views and 59% for
the pretrained encoder with a good-view head. It is at most 17% for the four spaces that do not learn from the links.

The fresh user ran Path 1 and the `data` and `v4` stages of Path 2. The corpus counts and the decision results were
identical to the stored files.

## 5. A check of the data

One of the 396 transitions of the decision analysis leads into February 2027, after the export ends. Appendix B of
the paper gives the observed rates of Table 13 without it (`cwm/run_actions.py checks`).

## 6. What these checks do not cover

- All results come from one simulated company. The checks show that the code makes the numbers of the paper. They
  do not show that the results hold for a real company.
- Path 2 needs the Docker images of Era. If Era changes its generator, the same seed can give a different company.
  The `data` stage then prints a warning.
- Runs are deterministic for a given seed and thread count, but not across thread counts.
