# Delivery-grid robustness

The 25-, 50-, and 100-ms conditions have identical saved measurements in both banks.
A 10-Hz integer-clock replay with inclusive arrival selection also gives identical packet-index histories.
The full grid has 800 held-out cells. Keeping one of these delays gives 480 cells (10 recordings x 4 services x 4 rates x 3 delays).

All seven qualification rules are refitted, and every development-only cutoff is reselected.
The existing full-grid scores and all 25% operating-point counts were reproduced before comparison.
The main reduced grid retains 50 ms because it is the configured practical delay. Retaining 25 or 100 ms is also reported.
No twin model or trajectory was regenerated. All intervals use 5,000 physical-recording bootstrap resamples.

## Centered bank

| Method | Full grid: false / accepted (risk) | Unique grid, retain 50 ms: false / accepted (risk) | Reduced acceptance |
| --- | --- | --- | --- |
| Historical surface | 4/200 (2.00%) | 2/118 (1.69%) | 24.58% |
| Service-specific logistic | 3/203 (1.48%) | 1/122 (0.82%) | 25.42% |
| Service-specific tree | 3/200 (1.50%) | 3/118 (2.54%) | 24.58% |
| AoI only | 28/160 (17.50%) | 18/100 (18.00%) | 20.83% |
| Common delivery | 30/200 (15.00%) | 18/120 (15.00%) | 25.00% |
| Service identity | 75/200 (37.50%) | 47/120 (39.17%) | 25.00% |
| Motion logistic | 55/199 (27.64%) | 32/120 (26.67%) | 25.00% |

At 25% requested acceptance, the reduced-grid surface has risk 1.69% (95% interval 0.00--4.13%). Reduced minus full risk is -0.31 percentage points (paired 95% interval -1.87--1.03).
At 50% requested acceptance, the reduced-grid surface has risk 12.92% (95% interval 3.33--23.33%). Reduced minus full risk is 1.17 percentage points (paired 95% interval 0.38--2.13).

Representative-delay sensitivity at 25% request:

- Historical surface: risk 1.69--1.69%; accepted 118--118 of 480.
- Service-specific logistic: risk 0.82--1.63%; accepted 121--123 of 480.
- Service-specific tree: risk 2.54--2.54%; accepted 118--118 of 480.
- AoI only: risk 18.00--18.00%; accepted 100--100 of 480.
- Common delivery: risk 15.00--15.00%; accepted 120--120 of 480.
- Service identity: risk 39.17--39.17%; accepted 120--120 of 480.
- Motion logistic: risk 26.67--28.95%; accepted 114--122 of 480.

## Causal bank

| Method | Full grid: false / accepted (risk) | Unique grid, retain 50 ms: false / accepted (risk) | Reduced acceptance |
| --- | --- | --- | --- |
| Historical surface | 19/168 (11.31%) | 11/98 (11.22%) | 20.42% |
| Service-specific logistic | 18/204 (8.82%) | 13/125 (10.40%) | 26.04% |
| Service-specific tree | 19/190 (10.00%) | 9/110 (8.18%) | 22.92% |
| AoI only | 37/160 (23.12%) | 23/100 (23.00%) | 20.83% |
| Common delivery | 27/144 (18.75%) | 24/112 (21.43%) | 23.33% |
| Service identity | 92/200 (46.00%) | 56/120 (46.67%) | 25.00% |
| Motion logistic | 49/202 (24.26%) | 30/117 (25.64%) | 24.38% |

At 25% requested acceptance, the reduced-grid surface has risk 11.22% (95% interval 0.00--25.45%). Reduced minus full risk is -0.09 percentage points (paired 95% interval -2.25--2.76).
At 50% requested acceptance, the reduced-grid surface has risk 24.31% (95% interval 7.94--43.48%). Reduced minus full risk is 3.36 percentage points (paired 95% interval 0.82--6.47).

Representative-delay sensitivity at 25% request:

- Historical surface: risk 11.22--11.22%; accepted 98--98 of 480.
- Service-specific logistic: risk 9.17--10.40%; accepted 120--129 of 480.
- Service-specific tree: risk 8.18--8.18%; accepted 110--110 of 480.
- AoI only: risk 23.00--23.00%; accepted 100--100 of 480.
- Common delivery: risk 21.43--21.43%; accepted 112--112 of 480.
- Service identity: risk 46.67--46.67%; accepted 120--120 of 480.
- Motion logistic: risk 25.64--27.35%; accepted 117--119 of 480.

The reduced grid changes the weighting of delivery conditions and can change fitted scores, cutoffs, acceptance, and risk.
It is a sensitivity analysis of the qualification comparison. It does not change the three-point delivery ledger.
