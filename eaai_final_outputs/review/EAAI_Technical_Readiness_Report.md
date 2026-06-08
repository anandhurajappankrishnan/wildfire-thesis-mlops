# EAAI Technical Readiness Report

**Author:** Anandhu Rajappan Krishnan  
**Generated:** 2026-06-08

## Executive verdict

**Ready for full manuscript writing: YES, with caveats.**  
**Ready for direct EAAI submission without manuscript: NO.**

## Key numbers (spatial holdout)

| Metric | Value |
|--------|-------|
| Dummy PR-AUC | 0.0270 |
| RF PR-AUC | 0.0910 (3.4x baseline) |
| XGB Combined ablation PR-AUC | 0.0740 |
| Base rate | 3.67% |
| Test positives | 97 / 4142 |

## Safe claims

- Reproducible end-to-end wildfire **risk ranking** pipeline for 0.1 deg areal cells
- RF PR-AUC meaningfully exceeds dummy baseline on unseen cell locations
- Engineering/MLOps contribution (medallion, orchestration, logging)
- Regional performance varies — must discuss honestly

## Unsafe claims (do not use)

- High-accuracy wildfire prediction
- Pixel-level 7-day prediction
- Causal ignition modelling
- Strong generalisation in all regions (SE Australia weak)

## Required manuscript framing

State explicitly: **areal 14-day wildfire risk ranking** using environmental susceptibility features.
Position as decision-support / prioritisation, not operational alarm system.
