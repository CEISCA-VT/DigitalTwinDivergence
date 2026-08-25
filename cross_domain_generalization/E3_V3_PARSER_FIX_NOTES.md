# E3-v3 FreeTwinEV parser hotfix

This patch changes **no scientific contract settings** from E3-v2. Horizons, tolerance grid, normalization, within-window statistic, transfer gates, MAGNET handling, and TU Wien SNG unit harmonization remain frozen.

## Why this patch exists

The real FreeTwinEV CSV export begins with an Excel-style delimiter directive such as `sep=;`. The E3-v2 flexible reader treated that directive as a header and returned the bogus columns `sep=` / `Unnamed: 1`, so the time column could not be identified.

## Fix

- Detect a leading `sep=<delimiter>` directive.
- Skip the directive row before parsing the actual header.
- Try both decimal-dot and decimal-comma interpretations.
- Rank candidate parses by monotonic time signal and numeric richness.
- Reject the exact bogus `sep=` header signature instead of passing it downstream.
- Add regression tests for both `sep=;` and `sep=;` + decimal comma.

This is a parser-only correction and must not be described as threshold/model tuning.
