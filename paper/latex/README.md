# CBSLAO — LaTeX source

This folder contains the LaTeX source for the paper
*Cost- and SLA-Bounded Orchestration for LLM-Agent Tool/Service Composition*
(ICSOC 2026 research track, Springer LNCS).

## File layout

```
paper/latex/
├── main.tex               # top-level document (LNCS class, preamble, \input sections)
├── references.bib         # bibliography, splncs04 style; verify citations before submission
├── sections/
│   ├── introduction.tex
│   ├── related_work.tex
│   ├── formulation.tex    # §3 problem statement
│   ├── hardness.tex       # §4 NP-hardness theorem + proof
│   ├── algorithm.tex      # §5 CBUC + regret theorem + algorithm pseudocode
│   ├── evaluation.tex     # §6 empirical study
│   ├── threats.tex        # §7 threats to validity
│   ├── limitations.tex    # §8 limitations & future work
│   ├── repro.tex          # reproducibility statement
│   ├── conclusion.tex     # §9 conclusion
│   └── appendix.tex       # full proof + additional experiments
└── figures/
    ├── budget_overrun_vs_rho.png
    ├── utility_vs_rho.png
    └── scaling_K.png
```

## Prerequisites

The paper uses the Springer **LNCS** class (`llncs.cls`) and the
associated BibTeX style (`splncs04.bst`). Download them from Springer
(https://www.springer.com/gp/computer-science/lncs/conference-proceedings-guidelines)
and place them alongside `main.tex`, or ensure they are on your
`TEXINPUTS` / `BSTINPUTS` path.

Required LaTeX packages (all standard in TeX Live ≥ 2021):
`amsmath amssymb amsthm mathtools graphicx booktabs array caption
algorithm algpseudocode hyperref xcolor microtype lmodern`.

## Build

```bash
cd paper/latex
pdflatex main
bibtex main
pdflatex main
pdflatex main
```

Or with `latexmk`:

```bash
latexmk -pdf main.tex
```

Output: `main.pdf`.

## Before submission — checklist

- [ ] Replace `Anonymous` author block with real authors/affiliations.
- [ ] Cross-check every reference marked `%% VERIFY` in `references.bib`
      against the original source; remove placeholder entries that
      cannot be confirmed.
- [ ] Fill in the full step-by-step proof of Theorem~\ref{thm:regret}
      in `sections/appendix.tex`.
- [ ] Add paired-seed Wilcoxon signed-rank test on aggregated cells.
- [ ] Insert anonymous repository URL in `sections/repro.tex`.
- [ ] Confirm figure placements after final body-text edits.
- [ ] Proofread for remaining `(verify)` / `(conjecture)` tags.

## Dependency on upstream artefacts

Figures are sourced from `../../plots/` of the outer repo (simulator
output). To regenerate them:

```bash
python3 ../../code/cbslao_sim.py ../../out/ 30
python3 ../../code/analyze.py ../../out/results.csv ../../plots/
cp ../../plots/*.png figures/
```
