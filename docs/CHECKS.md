# Checks of version 6

This file records the checks that we did before the release of version 6, and what they found. Each check has a
command, so that you can do it again. [`REPRODUCE.md`](../REPRODUCE.md) tells how to make every result again.
Version 6 has no new experiments. It gives the paper the format of a Nature article, and it answers an expert
assessment ([`REVIEW_RESPONSE.md`](REVIEW_RESPONSE.md)).

## 1. The numbers in the paper

- `cwm/fill_paper.py` writes every result into the paper from `results/`. It stops with an error if a placeholder
  has no value. The last run of version 6 left no placeholder without a value.
- `tests/test_paper_numbers.py` checks the numbers that the text of the paper states directly, against `results/`.
  Examples are the TF-IDF and random rows of Extended Data Table 2, the baselines in Supplementary Note 1 and the
  identity ceiling of 16.3%. Other tests cover the explained fractions, the alignment, the new quarter and the
  retrained encoder. The last tests cover persistence, the majority classes, the state facts, the transitions and
  the new records of day 2.
- `pytest -q tests` runs these 10 tests and 7 smoke tests. All 17 pass.
- New numbers in the main text come from `results/` through new keys of `cwm/fill_paper.py`. They are the guess
  of the largest customer (12%), the alignment result and its chance level (20.7% and 3.8%), and the persistence of
  states (0.88, 0.38 and 0.05).

## 2. A clean rebuild

We copied the files of the repository to an empty folder, without data and without caches. Then we ran
`make all PY=python3`, `make clean` and `make all PY=python3` again.

- The clean build gave a PDF that is identical, byte for byte, to the PDF in the repository. We repeated this check
  after the last change. The PDF has 44 pages, no overfull boxes and no undefined references.
- Every figure that needs no cache came out identical to the stored file. The two figures that need the caches (the
  maps and the persistence plot) use the stored files.
- The first comparison found two stored figures that were older than their code. We drew them again, and the
  stored files now match the clean build.

## 3. The writing

`python tools/ste_check.py FILE` measures how many sentences follow the main rules of ASD-STE100. A sentence must
have at most 25 words, active voice, a simple tense and no -ing forms. A paragraph must have at most six sentences.
The target was 90%. In version 6, the tool reads the whole paper, with Extended Data and Supplementary Information.

| Text | Sentences | Share that complies |
| --- | ---: | ---: |
| `paper/main.tex` | 1,457 | 98.4% |
| `README.md` | 298 | 100% |
| `website/template.html` | 397 | 99.2% |
| `REPRODUCE.md` | 94 | 96.8% |
| `docs/ARXIV_ABSTRACT.txt` | 17 | 100% |
| The new response in `docs/REVIEW_RESPONSE.md` | 110 | 100% |

The tool cannot check the approved dictionary of STE. We kept the technical terms of the field where no simple
word has the same sense. The summary paragraph and the figure legends of a Nature article can have more than six
sentences.

## 4. Independent reviews

Two agents reviewed the work. They did not write it. One had the paper, `results/` and the expert assessment. The
other had the website, the README and the paper, and it read the site as an executive would. Each reported its
findings with quotes, and we fixed every one that held.

| Review | What it found | Examples of fixes |
| --- | --- | --- |
| Paper against the results and the assessment | It checked 124 numbers and found no wrong one. It found claims stronger than the evidence, and descriptions of other work that were not fair to it. | "Establishes the first level" became evidence for the first level, more for the customer than for the period. Records whose links it never saw replace "records without links". Engram and Cartridges train skill with notes or a cache, not knowledge in the weights. Memorilla learns nothing from the company. Logs can identify effects only under two conditions, which do not hold here. |
| Paper: conditions and references | Results without their comparison, and references to the wrong note or table | The masked result now gives 75% with the names kept. The alignment states that both encoders saw the test states in training. The cost for the software engineer and the analyst is in the text. Five cross-references are now correct. |
| Website and README for an executive | Lines on what a result means that went beyond the paper, terms before their explanation, and a navigation row too small on phones | "Reaches the right customer's history" became "more often reaches the right customer's records". The cost of access control is in its result. The seeds line in the scope list is now exact. Phones show the section links in their own row. |
| Website: other work | Claims about other work that their sources do not support | The Engram gain is "reported" by Engram, and the gain held as the notes changed. The knowledge-graph row no longer says that graphs cannot find similar situations. |

The first review also counted the access scopes of the policy with `cwm/acl.py` on the records of day 1. The 12
employees have 11 different sets of readable records, because the rules follow each sender and each direct message.
We checked this count, and the Discussion now says that such rules make almost every person a separate scope.

## 5. The website

We took screenshots of every new section at 1,280 pixels in light and dark themes, and at 390 pixels on a phone.
No page is wider than the screen, and the browser reports no errors. On a phone, the comparison table shows one
card for each approach.

## 6. What these checks do not cover

- All results come from one simulated company. The checks show that the code makes the numbers of the paper. They
  do not show that the results hold for a real company.
- Path 2 needs the Docker images of Era. If Era changes its generator, the same seed can give a different company.
  The `data` stage then prints a warning.
- Runs are deterministic for a given seed and thread count, but not across thread counts.
