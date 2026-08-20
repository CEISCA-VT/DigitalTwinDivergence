# Official vs Internal Fidelity Comparison

Official i2Nav metrics and internal DT-fidelity metrics answer different scientific questions. Official APE/RPE measure trajectory error under the standardized benchmark alignment; internal fidelity measures physical-virtual synchronization in the operational/reference frame.

## Hard Sequences

- parking02: internal ATE 11.350 m -> official aligned APE 5.747 m; alignment-effect indicator 0.494; Dp p95 22.345 m; RPE10 0.097 m.
- parking01: internal ATE 4.071 m -> official aligned APE 1.920 m; alignment-effect indicator 0.528; Dp p95 7.763 m; RPE10 0.154 m.

This supports the local-vs-global distinction: short-horizon relative motion can look strong while long-horizon synchronization degrades.
