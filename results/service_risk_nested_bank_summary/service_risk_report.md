# Nested-Bank Service-Risk Comparator Summary

This run detected a corrected doubly nested response-surface table rather than a per-window service-risk table.
Therefore it summarizes the matched-acceptance comparator outputs produced by `service_evidence_from_bank`.
No timestamp-level context model is refit in this compatibility path.

## Primary Matched-Acceptance Rows

 scope service                method  target_acceptance  accepted_count  condition_count  false_qualified_count  achieved_acceptance  selective_risk  selective_risk_ci_low  selective_risk_ci_high  n_physical_sequences
pooled     all              aoi_only               0.25             160              800                     28              0.20000        0.175000                0.05625                0.312500                    10
pooled     all                common               0.25             200              800                     30              0.25000        0.150000                0.07500                0.225000                    10
pooled     all       motion_logistic               0.25             199              800                     55              0.24875        0.276382                0.00000                0.489097                    10
pooled     all     service_empirical               0.25             200              800                      4              0.25000        0.020000                0.00000                0.048544                    10
pooled     all service_identity_only               0.25             200              800                     75              0.25000        0.375000                0.26000                0.495000                    10

## Remediability Ledger

            classification  cases
       delivery_remediable     30
    persistent_under_ideal      6
already_qualified_degraded      4

## Files

- `risk_coverage_summary.csv`
- `matched_acceptance_per_sequence.csv`
- `matched_acceptance_pairwise_bootstrap.csv`
- `pooled_method_comparison.csv`
- `primary_operating_point_comparison.csv`
- `sequence_service_case_ledger.csv`
