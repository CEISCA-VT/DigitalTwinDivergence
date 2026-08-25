# Manuscript rewrite figure collector

Extract this package at the repository root, then run:

```powershell
.\collect_manuscript_rewrite_figures.cmd
```

The collector does **not** blindly copy every historical plot. It curates the figures most useful for the current IoTJ rewrite:

- E1 full-grid service-relative inversion and metric/service rank alignment;
- E2 i2Nav/TerraSentia contract portability;
- E3 MAGNET/FreeTwinEV/TU Wien SNG cross-domain transfer;
- current mechanism/condition/MAGNET/UGV01 figures that remain useful;
- supporting diagnostics in `figures\supplement`.

It writes:

- `figures\FIGURE_REWRITE_MANIFEST.csv`
- `figures\FIGURE_REWRITE_MANIFEST.md`

Use `-Strict` to return a nonzero exit code if any of the five new frozen E1/E2/E3 core figures are missing:

```powershell
.\collect_manuscript_rewrite_figures.cmd -Strict
```

Use `-RefreshExisting` to overwrite already-collected files from the preferred source result trees.
