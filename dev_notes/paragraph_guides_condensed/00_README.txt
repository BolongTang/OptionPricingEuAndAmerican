CONDENSED WRITING PLAN: 15 PARAGRAPHS INSTEAD OF 67
============================================================

Why: the detailed guides in dev_notes/paragraph_guides/ (67 paragraphs) are the full reference; this folder distils them into 15 paragraphs that are
enough for a normal-length paper. The detailed guides stay the source for formulas, wording ideas and caveats; each file here says which ones it replaces.

STRUCTURE (paragraphs in document order, sentences per paragraph in brackets)
  C01 [8] Abstract                 Abstract   <- P00
  C02 [8] 1 Introduction           Introduction: problem, contracts, roadmap, how to reproduce   <- P01, P02, P03
  C03 [8] 2 Setting                Setting, data and parameters   <- P04, P05
  C04 [8] 3 Part A                 Part A: random walk and Brownian motion   <- P06 to P11
  C05 [8] 4.1-4.3                  European model: CRR tree, GBM and Black-Scholes   <- P12, P13, P14, P15
  C06 [7] 4.4-4.5                  European results: tree accuracy and faster trees   <- P16, P17, P18
  C07 [7] 4.6                      European results: Monte Carlo and variance reduction   <- P19, P20, P21
  C08 [8] 4.7-4.8                  European Greeks and implied volatility   <- P22, P23
  C09 [8] 4.9 and Part B summary   Hedging in discrete time and with the wrong volatility   <- P24, P25, P26
  C10 [8] 5.1-5.2                  American put: optimal stopping, variational PDE, perpetual put   <- P27, P28, P29, P30
  C11 [8] 5.3-5.4                  American tree and baseline results   <- P31, P32, P33, P34, P35, P36, P37
  C12 [8] 5.4-5.6                  Long maturities and independent validators   <- P38, P39, P40, P41, P42
  C13 [7] 5.7-5.9                  Early-exercise premium, dividends and the NVIDIA put   <- P43, P44, P45
  C14 [8] 5.8-5.9                  Profit and loss, tail risk and mis-specified volatility   <- P46, P47, P48, P49
  C15 [9] 6-7                      Method comparison, conclusions and limits   <- P50, P51, P52, P53, P54, P55, P56, P57

Total: 15 paragraphs, 118 sentences.

HOW TO USE WITH THE EXISTING FILES
- report/paras/Pxx.tex still holds the 67 detailed drafts. To follow the short plan, write the 15 paragraphs directly in report/main.tex
  (or new files report/paras_short/Cxx.tex) and remove the \input lines of the old paragraphs you no longer need.
- The nine derivation paragraphs P58-P66 stay as the appendix: display the algebra, keep the prose to a sentence or two per step.
- Figures and tables: every guide lists which to keep in the main text; the others go to a supplementary section or are dropped. The notebook still
  produces all of them, so nothing is lost by moving them.
- Every number is a LaTeX macro (\Name); keep the macro when you rewrite so the text matches the notebook after a re-run.
