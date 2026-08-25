# E3: MAGNET + FreeTwinEV + TU Wien SNG cross-domain contract audit

This package is designed to extract at the **DigitalTwinDivergence repository root**.

It evaluates one frozen, unit-free service-contract structure across three non-robot digital-twin domains:

- INL MAGNET heat-pipe digital twin;
- FreeTwinEV P45B 1S4P battery electro-thermal experiment + validation simulation;
- TU Wien biomass-to-SNG pilot-plant digital twin campaign.

The runner automatically downloads public data that are not already present. FreeTwinEV is the largest download (~156 MB).

## Run

From the repository root:

```powershell
.\run_cross_domain_contract.cmd
```

If you prefer PowerShell directly:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_cross_domain_contract.ps1
```

## Output

`results\cross_domain_contract_generalization\`

Inspect these first:

1. `CROSS_DOMAIN_GENERALIZATION_REPORT.md`
2. `cross_domain_transfer_gates.csv`
3. `cross_domain_horizon_grid_average.csv`
4. `cross_domain_contract_macro.csv`
5. `freetwinev_pairing_audit.csv`
6. `sng_pairing_audit.csv`
7. `input_schema_audit.csv`
8. `analysis_manifest.json`
9. `figures\cross_domain_horizon_profile.png`

## Important claim boundary

This experiment tests **cross-domain portability of the service-contract representation**. It does not claim universal digital-twin fidelity, shared physical-unit safety limits, or model superiority in the target domains.

If the FreeTwinEV package changes schema, the script deliberately fails and writes/retains the schema evidence rather than guessing a physical/simulation pairing.
