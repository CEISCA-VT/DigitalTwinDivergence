from __future__ import annotations

import tempfile
from pathlib import Path
import sys

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

from cross_domain_contract_generalization import (  # noqa: E402
    ProtocolConfig,
    build_contract_grid,
    continuous_pairs_to_contract_units,
    transfer_gates,
    _physical_freetwinev_aggregates,
    _simulation_freetwinev_aggregates,
    _align_sim_to_physical_segment,
    _freetwinev_pair_candidates,
    _sng_dfb_pairs,
    _align_sng_softsensor,
)


def main() -> None:
    cfg = ProtocolConfig()
    # Shared contract evaluator: three horizons, non-degenerate threshold surface.
    t = np.arange(0, 3600, 10, dtype=float)
    rows = []
    for comp, phase in [("a", 0.0), ("b", 0.5)]:
        p = 10 + np.sin(t / 100 + phase)
        twin = p + 0.05 * np.sin(t / 35 + phase)
        for ti, pi, vi in zip(t, p, twin):
            rows.append({
                "dataset": "synthetic", "domain": "test", "subsystem": "test", "segment": "seg",
                "component": comp, "unit": "u", "time_s": ti, "physical": pi, "twin": vi,
            })
    paired = pd.DataFrame(rows)
    units = continuous_pairs_to_contract_units(paired, cfg)
    assert set(units["horizon_s"].unique()) == set(cfg.horizons_s)
    grid = build_contract_grid(units, cfg)
    gates = transfer_gates(units, grid, cfg)
    assert len(gates) == 1

    # FreeTwinEV schema heuristics.
    fexp = pd.DataFrame({
        "timestamp [s]": [1305, 1306, 1307],
        "temp_cell-1_TH20": [300.0, 301.0, 302.0],
        "temp_cell-2_TH17": [300.5, 301.5, 302.5],
        "temp_TH1": [298.0, 298.2, 298.4],
    })
    p = _physical_freetwinev_aggregates(fexp, "timestamp [s]")
    assert "cell_mean_K" in p.columns
    fsim = pd.DataFrame({
        "Time [s]": [0, 1, 2],
        "cell_temperature_1": [300.1, 301.1, 302.1],
        "cell_temperature_2": [300.4, 301.4, 302.4],
        "plate_temperature": [298.1, 298.3, 298.5],
    })
    s = _simulation_freetwinev_aggregates(fsim, "Time [s]")
    assert "cell_mean_K" in s.columns

    # FreeTwinEV hardened alignment: null keys, duplicate solver times and a
    # relative simulation clock must not reach pandas merge_asof or crash.
    tphys = np.arange(1305.0, 1366.0, 1.0)
    phys = pd.DataFrame({
        "time_s": np.r_[tphys[:15], np.nan, tphys[15:]],
        "cell_mean_K": np.r_[300.0 + 0.01 * np.arange(15), np.nan, 300.0 + 0.01 * np.arange(15, len(tphys))],
        "cell_max_K": np.r_[301.0 + 0.01 * np.arange(15), np.nan, 301.0 + 0.01 * np.arange(15, len(tphys))],
    })
    tsim = np.arange(0.0, 61.0, 1.0)
    sim = pd.DataFrame({
        "sim_time_s": np.r_[tsim[:20], np.nan, tsim[20:30], 29.0, tsim[30:]],
        "cell_mean_K": np.r_[300.1 + 0.01 * np.arange(20), np.nan, 300.1 + 0.01 * np.arange(20, 30), 300.39, 300.1 + 0.01 * np.arange(30, len(tsim))],
        "cell_max_K": np.r_[301.1 + 0.01 * np.arange(20), np.nan, 301.1 + 0.01 * np.arange(20, 30), 301.39, 301.1 + 0.01 * np.arange(30, len(tsim))],
    })
    aligned, ameta = _align_sim_to_physical_segment(phys, sim, 1305.0, 1365.0, "selftest")
    assert len(aligned) >= 55
    assert np.isfinite(aligned["time_s"]).all()
    assert ameta["relative_clock_detected"] is True
    pcands = _freetwinev_pair_candidates(aligned)
    assert {x[0] for x in pcands} >= {"temperature_mean", "temperature_max"}

    # Pairing must not depend on pandas suffix behavior: generic simulation
    # thermal aggregates can be paired to physical cell aggregates by statistic.
    fallback_aligned = pd.DataFrame({
        "time_s": [1.0, 2.0],
        "phys__cell_mean_K": [300.0, 301.0],
        "phys__cell_max_K": [302.0, 303.0],
        "sim__thermal_mean_K": [300.2, 301.2],
        "sim__thermal_max_K": [302.2, 303.2],
    })
    fcands = _freetwinev_pair_candidates(fallback_aligned)
    assert {x[0] for x in fcands} == {"temperature_mean", "temperature_max"}

    # Minute-based relative simulation clock should be recognized as a unit
    # conversion, not treated as a 36-second experiment.
    phys2_t = np.arange(1305.0, 3501.0, 5.0)
    phys2 = pd.DataFrame({
        "time_s": phys2_t,
        "cell_mean_K": 300 + 0.001 * (phys2_t - 1305),
        "cell_max_K": 301 + 0.001 * (phys2_t - 1305),
    })
    sim2_min = np.arange(0.0, (3500.0 - 1305.0) / 60.0 + 1e-9, 5.0 / 60.0)
    sim2 = pd.DataFrame({
        "sim_time_s": sim2_min,
        "cell_mean_K": 300.05 + 0.001 * sim2_min * 60.0,
        "cell_max_K": 301.05 + 0.001 * sim2_min * 60.0,
    })
    aligned2, ameta2 = _align_sim_to_physical_segment(phys2, sim2, 1305.0, 3500.0, "minute_clock")
    assert abs(ameta2["simulation_time_scale"] - 60.0) < 1e-12
    assert ameta2["alignment_coverage"] > 0.95

    # SNG exact documented schema.
    dfb = pd.DataFrame({
        "Time": [0, 1/3600, 2/3600],
        "Plt1_PGVolFlow_Measurement": [1.0, 1.1, 1.2],
        "Plt1_PGVolFlow_Estimate": [1.0, 1.08, 1.18],
        "Plt3_GasTemp_Measurement": [500, 501, 502],
        "Plt3_GasTemp_Estimate": [499, 501, 503],
    })
    d, audit = _sng_dfb_pairs(dfb)
    assert d["component"].nunique() == 2
    syng = pd.DataFrame({"Time": [0, 1/60, 2/60], **{f"Plt5_PG_{g}": [10, 11, 12] for g in ["H2","CO","CO2","CH4","C2H4"]}})
    soft = pd.DataFrame({"Time": [0, 1/60, 2/60], **{f"y_{g}_pg_wf_filter": [0.101, 0.111, 0.121] for g in ["H2","CO","CO2","CH4","C2H4"]}})
    q, audit2 = _align_sng_softsensor(syng, soft)
    assert q["component"].nunique() == 5
    assert abs(float(q.iloc[0]["twin"]) - 10.1) < 1e-9
    assert set(pd.DataFrame(audit2)["twin_scale_factor"].dropna().unique()) == {100.0}

    # Semicolon-delimited CSV must not be silently accepted as one column.
    from cross_domain_contract_generalization import _read_csv_flexible
    with tempfile.TemporaryDirectory() as td:
        cp = Path(td) / "semi.csv"
        cp.write_text("Time [s];cell_temperature_1;plate_temperature\n0;300.0;298.0\n1;301.0;298.2\n", encoding="utf-8")
        parsed = _read_csv_flexible(cp)
        assert len(parsed.columns) == 3

        # Excel-style separator directive observed in the real FreeTwinEV run.
        sp = Path(td) / "sep_directive.csv"
        sp.write_text("sep=;\nTime [s];cell_temperature_1;plate_temperature\n0;300.0;298.0\n1;301.0;298.2\n", encoding="utf-8")
        parsed2 = _read_csv_flexible(sp)
        assert list(parsed2.columns) == ["Time [s]", "cell_temperature_1", "plate_temperature"]
        assert parsed2["Time [s]"].tolist() == [0, 1]

        # European decimal comma plus sep=; must also become numeric.
        dp = Path(td) / "sep_decimal_comma.csv"
        dp.write_text("sep=;\nTime [s];cell_temperature_1;plate_temperature\n0;300,0;298,0\n1;301,0;298,2\n", encoding="utf-8")
        parsed3 = _read_csv_flexible(dp)
        assert pd.to_numeric(parsed3["cell_temperature_1"], errors="coerce").notna().all()

        # Preamble + spaced separator directive should also parse.
        pp = Path(td) / "preamble_sep.csv"
        pp.write_text("\nsep = ;\nTime [s];cell_temperature_1\n0;300,0\n1;301,0\n", encoding="utf-8")
        parsed4 = _read_csv_flexible(pp)
        assert list(parsed4.columns) == ["Time [s]", "cell_temperature_1"]
        assert np.isfinite(pd.to_numeric(parsed4["Time [s]"], errors="coerce")).all()
    print("cross_domain_contract_selftest: PASS")


if __name__ == "__main__":
    main()
