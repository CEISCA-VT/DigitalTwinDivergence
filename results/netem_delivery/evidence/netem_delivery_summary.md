# Linux netem delivery replay

Frozen states were delivered using measured UDP delay/loss traces captured over a Linux veth link shaped with `tc netem`.
This is OS-level delivery emulation over precomputed states, not live twin inference.

## Case ledger

- Delivery-remediable: **32/40**
- Persistent under ideal delivery: **6/40**
- Already qualified under degraded delivery: **2/40**
- Restored by practical delivery: **27**

Transport repetitions quantify delivery variability; they are not additional physical-sequence replicates.
