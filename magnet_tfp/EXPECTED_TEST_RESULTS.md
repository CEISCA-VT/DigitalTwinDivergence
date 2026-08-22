# Expected validation result on the archived official MAGNET files

The package was run end-to-end against the two official INL single-file CSVs before packaging. The archived dataset produced:

- 111 reconstructed 600-s forecast windows;
- 97 strict-eligible windows with all 10 thermowell forecasts present;
- 98 partial-eligible windows with at least 8 thermowells;
- 23 greedily selected non-overlapping strict windows;
- median 0–60 s window RMSE of about 0.085 °C;
- median 301–599 s window RMSE of about 6.67 °C;
- median long/short RMSE ratio of about 78.5×;
- 47 pairs of windows with aggregate RMSE within 10% but materially different tail, persistence, or long-horizon behavior;
- clear component localization, especially increasing long-horizon error in hotter thermowells;
- condition-stratified median RMSE differences across heating, steady, and cooling regimes.

The automated exploratory decision aid returned **5/5 (STRONG)** on this archived snapshot. That score is intentionally not a scientific metric and should never be reported in the manuscript; it only summarizes whether the generated evidence is worth inspecting for a cross-domain section.

One released forecast episode also contains extreme long-horizon numerical divergence in one thermowell (the public forecast itself reaches large negative values). The pipeline retains this rather than clipping it, because tail failure is part of the physical–virtual fidelity question. Always inspect the raw forecast and the generated tail tables before deciding how to discuss these episodes.
