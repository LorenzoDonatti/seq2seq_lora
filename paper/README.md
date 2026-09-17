# Manuscript draft — Internet of Things

This directory contains an `elsarticle` manuscript skeleton aligned with the scope of
Elsevier's *Internet of Things*. It is deliberately a protocol-first draft: bracketed
items and `TODO` markers must be replaced only after the confirmatory multi-gateway,
multi-seed runs are frozen.

Compile with:

```bash
latexmk -pdf main.tex
```

The distribution must include the Elsevier class and bibliography style. On
Debian/Ubuntu they are provided by `texlive-publishers`:

```bash
sudo apt install latexmk texlive-publishers
```

The scientific results remain authoritative in `benchmark_results/`; tables and
figures should be generated from those artifacts rather than copied by hand. Before
submission, check the current journal Guide for Authors, add the graphical abstract,
CRediT statement, data/code availability statement and declarations requested by the
submission system.

The generated `main.pdf` and auxiliary files are ignored by Git.

