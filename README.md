# European and American Option Pricing

Binomial trees, Brownian motion, Monte Carlo and the variational PDE, applied to a one-year European call on NVIDIA (NVDA) and to an American put.

The repository contains one executable notebook, one LaTeX report and the code that links them. Every figure, table and number in the report is produced by the notebook.

## What is inside

| path | content |
|---|---|
| `notebooks/option_pricing.ipynb` | the whole analysis: Part I (random walk and Brownian motion), Part II (European call: CRR tree, GBM Monte Carlo, Black-Scholes, Greeks, implied volatility, hedging), Part III (American put: variational PDE, perpetual put, CRR tree, finite-difference, Longstaff-Schwartz, Barone-Adesi-Whaley, dividends, profit and loss), Part IV (method comparison) |
| `report/main.tex`, `report/references.bib` | the report; build with `pdflatex main && bibtex main && pdflatex main && pdflatex main` inside `report/` |
| `report/paras/` | one `.tex` file per prose paragraph, pulled into `main.tex` with `\input` |
| `report/macros.tex` | one `\newcommand` per number quoted in the text, **generated** by the notebook |
| `report/figures/`, `report/tables/` | figures and tables written by the notebook |
| `report/code/` | the code listing of the appendix, cut out of the notebook (`handwritten.py`, `extensions.py`) |
| `todo.txt` | open decisions and clean-up steps |
| `dev_notes/` | working notes and per-paragraph writing guides (development branch only) |

## Design

The classes `BrownianMotion`, `symmetric_random_walk`, `CRRBinomialModel`, `CRRTree`, `gbm_euler_maruyama` and `gbm_exact` are hand-written and are used unchanged. Every extension is a new function or a subclass placed after them and tagged `[EXT]` in the notebook (for example `CRRTreeGen` adds dividends and calls on top of `CRRTree`).

## Reproducing the results

```bash
pip install -r requirements.txt
cd notebooks
jupyter nbconvert --to notebook --execute --inplace option_pricing.ipynb     # about 4-10 minutes, one core
cd ../report && pdflatex main && bibtex main && pdflatex main && pdflatex main
```

All random numbers are seeded. The market inputs (NVDA close of 2026-10-05, volatility from 250 daily log returns, one-year Canadian T-bill rate) are frozen in the notebook; set `RUN_LIVE_DATA = True` in the data cell to refresh them from `yfinance`.

## Scope and limitations

Constant volatility and interest rate, no transaction costs, no jumps, no dividends except in the dedicated dividend section. The report lists the consequences in its conclusions.

## AI assistance

Parts of the code, the derivation write-ups and the first drafts of the prose were produced with the help of an AI assistant (Anthropic's Claude) and reviewed by the author, who is responsible for the content.

## Licence

MIT, see `LICENSE`.
