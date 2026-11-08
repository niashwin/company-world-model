# Checks of version 7

This file records the checks that we did before the release of version 7, and what they found. Each check has a
command, so that you can do it again. [`REPRODUCE.md`](../REPRODUCE.md) tells how to make every result again.
Version 7 has no new experiments. It makes the website and the README clear to a general reader, and it checks and
links every reference. The checks of version 6 are in the history of this file.

## 1. The references

Three agents checked the 56 entries of `paper/refs.bib`. They did not write them. Each agent compared its entries
with the page of the publisher, Crossref, DBLP or arXiv, and reported the source of each fact.

- 17 entries changed. Thirteen now cite the published version of a preprint. Five now have the authors of the
  version that they cite. The venue of one entry was wrong: the dataset of Mollaev and others appeared at CIKM
  2025, not at KDD 2025. The Platonic Representation Hypothesis now has the title of its ICML version, and
  OrgForge the title of its arXiv listing (see below).
- The title of OrgForge differs between sources. The PDF of its second version has "Organizational", but the
  arXiv listing and the author's own citation have "Corporate". The link of the entry opens the arXiv listing, so
  the entry now uses "Corporate".
- Every entry now has a link: a DOI, an arXiv page or the page of the proceedings. The PDF links each entry. The
  website and the README list all 56, each with its link. `cwm/export_site_data.py` and `tools/readme_refs.py` stop
  with an error if an entry has no link.
- A reader of version 6 asked us to check four reference numbers. They are Engram in Table 1, the year of reference
  6, and two citations in Supplementary Note 10. We read each one in the text of the PDF and in an image of the
  page. All four were already correct. Table 1 cites Engram as reference 13, and reference 6 ends with
  "arXiv:2609.09853, 2026". The two citations point to references 33 and 29. Version 7 keeps these numbers.

## 2. The numbers in the paper

- `cwm/fill_paper.py` writes every result into the paper from `results/`. It stops with an error if a placeholder
  has no value. The last run of version 7 left no placeholder without a value.
- `tests/test_paper_numbers.py` checks the numbers that the text of the paper states directly, against `results/`.
- `pytest -q tests` runs these 10 tests and 7 smoke tests. All 17 pass.
- The website and the README take their numbers from `results/` through `cwm/export_site_data.py`, or state the
  numbers of the paper. A review checked each number on the website and in the README against the paper (section 5).

## 3. A clean rebuild

We copied the files of the repository to an empty folder, without data and without caches. Then we ran
`make all PY=python3`, `make clean` and `make all PY=python3` again.

- The clean build gave a PDF that is identical, byte for byte, to the PDF in the repository. The PDF has 46 pages,
  no overfull boxes and no undefined references.
- Every figure came out identical to the stored file. The two figures that need the caches (the maps and the
  persistence plot) use the stored files.

## 4. The writing

`python tools/ste_check.py FILE` measures how many sentences follow the main rules of ASD-STE100. A sentence must
have at most 25 words, active voice, a simple tense and no -ing forms. A paragraph must have at most six sentences.
The target was 90%.

| Text | Sentences | Share that complies |
| --- | ---: | ---: |
| `paper/main.tex` | 1,455 | 98.3% |
| `README.md` | 346 | 100% |
| `website/template.html` | 385 | 100% |
| `REPRODUCE.md` | 94 | 96.8% |
| `docs/ARXIV_ABSTRACT.txt` | 17 | 100% |

The tool cannot check the approved dictionary of STE. We kept the technical terms of the field where no simple
word has the same sense. The summary paragraph and the figure legends of a Nature article can have more than six
sentences. The tool does not check the lists of references in the README, because the titles of other works are
not our text.

## 5. Independent reviews

Three agents that did not write the work reviewed it. Each reported its findings with quotes, and we fixed every
one that held.

### The paper

One agent looked at all 46 pages as images and read the text. It checked about 45 numbers against `results/`, and
all of them agree. It found these problems, and we fixed them.

- The Methods described the data export, the encoder, the pretrained backbone and the metrics twice. Each pair is
  now one description.
- One note cited itself, and two Supplementary Tables had no citation. Three tables had columns without an
  explanation.
- Figure 1 numbered its panels 1 to 3, its smallest text was about 4 points, and one note claimed more than the
  results show. Its panels are now a to c, its smallest text is above 6 points, and the note is gone.
- Legends sat on data in Fig. 2a and Extended Data Fig. 1b. Labels touched in Fig. 5a, and panel letters sat on
  axis labels in Extended Data Figs. 1, 2 and 4. The key of Fig. 4 described the wrong colors. We drew these
  figures again from `cwm/make_figures.py`.
- A heading stood alone at the bottom of a page, URLs broke after "https:", and the footnote mark looked like a
  citation. We fixed all three.
- The references followed two rules for authors. `paper/unsrtnat-nature.bst` now applies the rule of Nature: up to
  five authors, all names; more than five, the first author and "et al.". The NeurIPS and ICML entries now give
  their volumes in one form.

### The website and the README

An agent that did not write the website or the README read both as an executive would. It also checked each
number against the paper and opened each link. It reported its findings with quotes, and we fixed every one that
held.

| What it found | What we changed |
| --- | --- |
| The first screen had no result and no source for the method. | The first screen now names JEPA and the proof of Balestriero and LeCun, with links. It also gives the main result: 76% against 53% for a keyword search. |
| Two results claimed more than the test showed. "One email is enough" overstated a test in which the right answer came 63.8% of the time. "The AI models that companies already use" was one small text model. | "One email can stand in for the customer's name", with the ceiling of 68.7% for a system that knows the true link. The other result now names MiniLM and says that we tested one small model. |
| The result on leaks named one employee and "one model for everyone". The test used two employees and one model for each group with the same access. | The result now names two employees. It says that a filter on search results did not stop the leak. Each group's model learns only from what the group may read. |
| The result on similar situations said "months". The test uses quarters and the five most similar states. | The result now says quarters and the five most similar, and that the share depends on the representation. |
| The caption of Figure 5 described the wrong measure. | The caption now gives the share of the top results that belong to the customer with the codename, as in Extended Data Table 4. |
| Two low numbers (33% and 63.8%) had no yardstick, and the results read as facts about any company. | Each result now has its baselines. The results say that they come from the simulated company, and the result on cost gives the size of the company. |
| Sentra appeared among the published systems of the comparison, which Table 1 of the paper does not list. | On the website, Sentra now appears as an example of the memory layer, in the byline and in the brand. |
| On phones, the row of section links could not scroll to its first links, and the page hid the Paper button. | The row now starts at its first link, and phones keep the Paper button. |

The agent also reported that Figure 4 showed 41% for the frozen pretrained encoder, where the paper gives 40%.
The figure rounded twice: it rounded the table value 0.405, which is 0.4048 rounded. The left panel of Figure 4
now shows one decimal (40.5%), so it rounds once. The value of the paper, 40%, is correct. The agent found no
other wrong number on the website or in the README. It found no information from Sentra that is not public.

A third agent read the website and the README again after the changes that present the world model as a
foundation. It found two claims that went beyond the paper. The world model files records under their customer,
but not reliably under their quarter. Similar cases are a first version, because every representation does about
as well. Both now say so. It also found that the image for link previews did not name MIT, and that the footer
logo needed space between its parts. We fixed both.

## 6. The website

- The website uses the logo, the colors and the type of Sentra. The logo files in `website/assets/brand/` are copies
  of the files on sentra.app, and their SHA-256 hashes match. The type is Geist and Geist Mono, under the SIL Open
  Font License, which is in `website/assets/fonts/`.
- The four colors of the charts pass the checks of the palette validator in the light and the dark theme. Text
  colors have a contrast of at least 4.5 to 1 with their background.
- The website has 114 links, and every link inside the page reaches its target. The website and the README
  link to 64 different outside pages. The agent of the review opened each one or confirmed it through another
  page. The address of the website works only after GitHub Pages is on for the repository.
- We took screenshots of every section at 1,280 pixels in the light and the dark theme, and at 390 pixels on a
  phone. The header also has screenshots at 820 and 1,100 pixels. No page is wider than the screen, and the
  browser reports no errors.

## 7. What these checks do not cover

- All results come from one simulated company. The checks show that the code makes the numbers of the paper. They
  do not show that the results hold for a real company.
- An agent can open a link and still miss a wrong fact on the page. We checked the facts of each reference against
  two sources where we could.
- Path 2 needs the Docker images of Era. If Era changes its generator, the same seed can give a different company.
  The `data` stage then prints a warning.
- Runs are deterministic for a given seed and thread count, but not across thread counts.
