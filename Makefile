# Reproduce the paper's numbers, figures and PDF from results/ (see README for generating data and results).
PY ?= python

.PHONY: figures paper all clean

figures:
	$(PY) cwm/make_figures.py

paper/figures/idea.pdf: paper/figures/idea.tex
	cd paper/figures && pdflatex -interaction=nonstopmode idea.tex >/dev/null && rm -f idea.aux idea.log

paper: paper/figures/idea.pdf
	$(PY) cwm/fill_paper.py paper/main.in.tex paper/main.tex
	cd paper && latexmk -pdf -interaction=nonstopmode main.tex

all: figures paper

clean:
	cd paper && latexmk -C main.tex
