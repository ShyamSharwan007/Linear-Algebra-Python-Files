# Evaluating Supply Chain Flexibility of 3PL Providers using Linear Algebra

A small end-to-end pipeline that scores and ranks Third-Party Logistics (3PL) providers under different operating conditions — **Normal**, **Peak**, and **Disruption** — using nothing but core linear algebra: normalization, weighted scoring, covariance/correlation, PCA, and SVD.

Built as a course project to answer a practical question: *when things get stressful (peak demand, supply disruption), which logistics provider actually holds up, and which one just looked good on paper?*

## What it does

1. **Generates a synthetic dataset** — 5 providers (A–E) × 6 criteria × 3 scenarios, stored as a `5×6×3` tensor.
2. **Normalizes** each scenario matrix two ways:
   - Min–Max scaling to `[0, 1]` (for weighted scoring)
   - Z-score standardization (for PCA/SVD)
3. **Computes a weighted score** per provider: `S = X_norm · w`, where `w` is a criteria-importance vector (with an optional AHP/eigenvector method to derive `w` from pairwise comparisons instead of guessing it).
4. **Runs PCA** on the covariance matrix to find the directions that explain the most variance in provider performance, and plots providers on PC1 vs PC2 to visualize clustering.
5. **Runs SVD** (`X_c = U Σ Vᵀ`) on the centered data to cross-check PCA and inspect how many latent factors actually matter.
6. **Compares rankings across scenarios** to measure *flexibility* — a provider that stays near the top in Normal, Peak, *and* Disruption is more valuable than one that only wins when conditions are ideal.

## Criteria & scenarios

| Criteria | Scenarios |
|---|---|
| Cost, Time, Flexibility, Reliability, Capacity, Coverage | Normal, Peak, Disruption |

Default criteria weights: `[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]` (Cost and Time weighted highest — adjust in `scorer.py` for your own priorities).

## Project structure

```
.
├── data_generator.py   # builds the 5x6x3 dataset, writes 3PL_dataset.csv
├── normalizer.py        # min-max and z-score normalization
├── scorer.py             # weighted scoring (+ optional AHP weighting)
├── analyzer.py           # covariance, PCA, SVD, and plotting
├── main.py                # orchestrates the full pipeline across all scenarios
├── requirements.txt
└── report/                 # full write-up and slides (math derivations, worked example)
```

## Getting started

```bash
git clone https://github.com/ShyamSharwan007/Linear-Algebra-Python-Files.git
cd Linear-Algebra-Python-Files
pip install -r requirements.txt

python data_generator.py
python main.py
```

This produces, per scenario:
- `scores_<scenario>.csv` — ranked provider scores
- `<scenario>_pca.png` — PC1 vs PC2 clustering plot
- `<scenario>_singular_values.csv` — SVD singular values

## Example: PCA clustering (Normal scenario)

Providers separate cleanly along PC1 (~41% of variance) and PC2 (~34% of variance), with Provider D standing out as a high performer and B/C clustering together as mid-tier.

*(see `report/Normal_pca.png` for the plot)*

## The math, briefly

- **Min-Max normalization:** `x' = (x - min) / (max - min)`, scales each criterion to `[0,1]` so no single criterion dominates due to units/scale.
- **Weighted score:** `S = X_norm · w` — a straightforward matrix-vector multiply that turns 6 criteria into 1 ranking number per provider.
- **PCA:** eigendecomposition of the covariance matrix `C = (1/(n-1)) Xᶜᵀ Xᶜ = V Λ Vᵀ` to find the latent factors driving provider differences.
- **SVD:** `Xᶜ = U Σ Vᵀ` — a numerically stable way to get the same principal directions as PCA (`σᵢ² / (n-1) = λᵢ`), plus a natural tool for low-rank denoising and robust ranking.

Full derivations, a worked numerical example, and discussion are in [`report/LA_Final_Report.pdf`](report/LA_Final_Report.pdf) and the slide deck [`report/LA_Presentation.pdf`](report/LA_Presentation.pdf).

## Key finding

Rankings are **not stable across scenarios** — the provider that leads under Normal conditions is not necessarily the one that leads under Disruption. This is the core argument for why "flexibility" (consistency across scenarios) deserves to be measured explicitly, rather than assuming a single-scenario benchmark tells the whole story.

## References

- Choy, K.L. et al., *"Leveraging the supply chain flexibility of third party logistics – Hybrid knowledge-based system approach"*, Expert Systems with Applications, 2008.
- Jolliffe, I.T., *Principal Component Analysis*, Springer, 2002.
- Golub, G.H. & Van Loan, C.F., *Matrix Computations*, Johns Hopkins University Press.
- Saaty, T.L., *The Analytic Hierarchy Process*, McGraw-Hill, 1980.

## Authors

- Sharwan (CS24B1059)
- Jyothir (CS24B1058)
