# Causal-derivative LOSO summary

- Provenance audit: **PASS**, 30/30 trajectories, 10 physical sequences, three seeds.
- Feature rule: backward differences; first derivative sample is zero; numerical future-sample perturbation preflight passed.
- Sequence-macro ATE: `2.398 m` centered versus `3.882 m` causal (`+61.9%`).
- Sequence-macro heading MAE: `2.569 deg` centered versus `3.845 deg` causal (`+49.7%`).
- Causal ledger: `28` delivery-remediable, `8` persistent, and `4` already qualified.
- The practical 5-Hz/50-ms intervention restores `26/28` remediable cases.

The study removes future-sample dependence from derivative-bearing inputs, so it closes the causal-feature provenance gap. It does not improve trajectory accuracy: retraining with backward derivatives degrades aggregate fidelity, especially on building01/building02 and parking02. The service-level structure nevertheless remains similar to the centered-gradient result (historically 30/6/4 with 27/30 practical recoveries). This is therefore a robustness and negative-result finding, not evidence that causal derivatives improve the model.
