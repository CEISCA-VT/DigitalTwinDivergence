# E1 + E2 publication hardening

Extract this archive at the repository root and run `run_e1_e2_publication.cmd`.

The runner:
1. verifies the E1 frozen service-relative outputs exist;
2. if TerraSentia full-study outputs are absent, runs the repository's existing frozen `aifarms_terrasentia_full_study.py`;
3. runs E1/E2 publication analysis;
4. writes `results/e1_e2_service_contract_publication/`.

Upload that result directory after completion.
