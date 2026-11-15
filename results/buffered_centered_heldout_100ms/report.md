# 100-ms buffered-centered held-out experiment

Only availability changes: arrival = original source time + 100 ms + network delay.
Source states, reference, network-cell labels, services, and sensor prefix features remain fixed.
All 300 trajectory hashes verified; unbuffered measurements and seven-method results reproduced.
All development and outer-test labels are replayed. Screening rules and cutoffs are refitted.
Intervals use 5,000 whole-physical-sequence bootstrap samples.

## unbuffered_control

Target | Method | Accepted | False | Risk (%) | 95% interval (%)
--- | --- | --- | --- | --- | ---
25% | aoi_only | 160 | 28 | 17.50 | 5.62--30.63
50% | aoi_only | 390 | 124 | 31.79 | 16.15--47.95
25% | common | 200 | 30 | 15.00 | 7.50--22.50
50% | common | 400 | 81 | 20.25 | 9.25--31.50
25% | motion_logistic | 199 | 55 | 27.64 | 0.00--48.21
50% | motion_logistic | 396 | 108 | 27.27 | 11.89--41.98
25% | service_empirical | 200 | 4 | 2.00 | 0.00--4.88
50% | service_empirical | 349 | 41 | 11.75 | 2.57--22.00
25% | service_identity_only | 200 | 75 | 37.50 | 26.00--50.00
50% | service_identity_only | 400 | 161 | 40.25 | 32.00--48.50
25% | service_specific_logistic | 203 | 3 | 1.48 | 0.00--3.43
50% | service_specific_logistic | 402 | 85 | 21.14 | 9.61--32.51
25% | service_specific_tree | 200 | 3 | 1.50 | 0.00--4.50
50% | service_specific_tree | 346 | 48 | 13.87 | 5.85--23.12

## buffered_100ms

Target | Method | Accepted | False | Risk (%) | 95% interval (%)
--- | --- | --- | --- | --- | ---
25% | aoi_only | 160 | 61 | 38.12 | 21.25--55.62
50% | aoi_only | 390 | 198 | 50.77 | 31.28--70.26
25% | common | 200 | 48 | 24.00 | 11.00--37.51
50% | common | 400 | 161 | 40.25 | 22.75--57.25
25% | motion_logistic | 189 | 82 | 43.39 | 5.66--67.01
50% | motion_logistic | 396 | 194 | 48.99 | 27.56--64.57
25% | service_empirical | 156 | 33 | 21.15 | 5.97--36.09
50% | service_empirical | 311 | 109 | 35.05 | 18.06--52.20
25% | service_identity_only | 200 | 113 | 56.50 | 43.00--70.50
50% | service_identity_only | 400 | 239 | 59.75 | 50.50--69.25
25% | service_specific_logistic | 190 | 37 | 19.47 | 6.45--33.67
50% | service_specific_logistic | 400 | 163 | 40.75 | 23.25--57.71
25% | service_specific_tree | 180 | 31 | 17.22 | 6.04--30.98
50% | service_specific_tree | 386 | 151 | 39.12 | 22.31--56.48
