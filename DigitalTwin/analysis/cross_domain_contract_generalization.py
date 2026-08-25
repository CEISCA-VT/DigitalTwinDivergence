from __future__ import annotations

import argparse
import csv
import hashlib
import io
import json
import math
import re
import shutil
import sys
import traceback
import zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None


# -----------------------------------------------------------------------------
# Frozen cross-domain contract protocol
# -----------------------------------------------------------------------------

MAGNET_SENSORS = [f"Heat Pipe TC-{i:02d}" for i in range(1, 11)]
MAGNET_TIME = "Time (s)"

HORIZONS_S = (60, 300, 600)
NORM_TOLERANCES = (0.01, 0.02, 0.05, 0.10, 0.20, 0.50)
CONTRACT_QUANTILE = 0.95

FREETWINEV_ZENODO_RECORD = "19935693"
FREETWINEV_ZIP = "FTEV_1S4P_IdentValid_SimPackage_V1.zip"
FREETWINEV_MD5 = "a37cf1619d14a28e4e4397b55f0b58ac"
FREETWINEV_URLS = (
    f"https://zenodo.org/records/{FREETWINEV_ZENODO_RECORD}/files/{FREETWINEV_ZIP}?download=1",
    f"https://zenodo.org/api/records/{FREETWINEV_ZENODO_RECORD}/files/{FREETWINEV_ZIP}/content",
)

SNG_RECORD = "6mmjq-1tj37"
SNG_BASE_URLS = (
    f"https://researchdata.tuwien.at/api/records/{SNG_RECORD}/files/{{name}}/content",
    f"https://researchdata.tuwien.ac.at/records/{SNG_RECORD}/files/{{name}}?download=1",
    f"https://researchdata.tuwien.ac.at/api/records/{SNG_RECORD}/files/{{name}}/content",
    f"https://researchdata.tuwien.ac.at/records/{SNG_RECORD}/files/{{name}}?download=1",
)
SNG_FILES = {
    "Data_MPC_DFB.csv": "ac8362f321887c55ae80d05429f2e001",
    "Data_MPC_Syngas.csv": "46b0b80fc160fa073627b0bd7127b191",
    "Data_SoftSensor.csv": "c905a047747d121bb622391e30bf15bb",
    "README.txt": "2194099097a489152d47b21755ab02fa",
}

MAGNET_URLS = {
    "MAGNET_Heat_Pipe_2022-03-30.csv": "https://raw.githubusercontent.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data/main/Experiment/Single_File/MAGNET_Heat_Pipe_2022-03-30.csv",
    "ML_MAGNET_2022-03-30.csv": "https://raw.githubusercontent.com/IdahoLabResearch/MAGNET-Heat-Pipe-Data/main/Machine_Learning/Single_File/ML_MAGNET_2022-03-30.csv",
}


@dataclass
class ProtocolConfig:
    horizons_s: Tuple[int, ...] = HORIZONS_S
    normalized_tolerances: Tuple[float, ...] = NORM_TOLERANCES
    contract_quantile: float = CONTRACT_QUANTILE
    robust_scale_q_low: float = 0.05
    robust_scale_q_high: float = 0.95
    robust_scale_relative_floor: float = 0.01
    min_components_for_transfer: int = 2
    min_horizons_for_transfer: int = 3
    min_units_per_horizon: int = 5
    freetwinev_cd_physical_start_s: float = 1305.0
    freetwinev_cd_physical_end_s: float = 3500.0
    freetwinev_dis_physical_start_s: float = 4110.0
    freetwinev_dis_physical_end_s: float = 8105.0
    random_seed: int = 260825


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _md5(path: Path) -> str:
    h = hashlib.md5()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(urls: Sequence[str], destination: Path, label: str) -> None:
    if requests is None:
        raise RuntimeError("requests is required for automatic data download")
    destination.parent.mkdir(parents=True, exist_ok=True)
    errors = []
    for url in urls:
        try:
            print(f"[data] Downloading {label}: {url}")
            with requests.get(url, stream=True, timeout=120) as r:
                r.raise_for_status()
                tmp = destination.with_suffix(destination.suffix + ".part")
                with tmp.open("wb") as f:
                    for chunk in r.iter_content(chunk_size=1024 * 1024):
                        if chunk:
                            f.write(chunk)
                tmp.replace(destination)
            return
        except Exception as e:  # noqa: BLE001
            errors.append(f"{url}: {e}")
            destination.with_suffix(destination.suffix + ".part").unlink(missing_ok=True)
    raise RuntimeError(f"Could not download {label}. Tried:\n" + "\n".join(errors))


def _ensure_download(path: Path, urls: Sequence[str], label: str, md5: Optional[str], allow_download: bool) -> Path:
    if path.exists():
        if md5 and _md5(path).lower() != md5.lower():
            raise RuntimeError(f"{label} exists but MD5 does not match official record: {path}")
        return path
    if not allow_download:
        raise FileNotFoundError(f"Missing {label}: {path}")
    _download(urls, path, label)
    if md5 and _md5(path).lower() != md5.lower():
        raise RuntimeError(f"Downloaded {label} but MD5 does not match official record")
    return path


def _extract_zip_once(zip_path: Path, out_dir: Path) -> Path:
    marker = out_dir / ".extract_complete"
    if marker.exists():
        return out_dir
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(out_dir)
    marker.write_text("ok\n", encoding="utf-8")
    return out_dir


def _read_csv_flexible(path: Path) -> pd.DataFrame:
    """Read heterogeneous CSVs, including Excel-style ``sep=`` directives.

    FreeTwinEV exports some CSVs with a first line such as ``sep=;``.  Pandas
    interprets that directive as the header unless it is skipped, producing a
    bogus two-column table (for example ``['sep=', 'Unnamed: 1']``).  This
    reader detects that directive explicitly, skips it, and then ranks plausible
    delimiter/decimal interpretations.  The ranking rewards a monotonic time-like
    column and numeric content, and heavily penalizes the known false-header
    signature so a malformed parse cannot silently win.
    """
    errors = []
    candidates = []

    raw = path.read_bytes()[:16384]
    raw_lines = raw.splitlines()

    directive_sep = None
    directive_skiprows = 0
    # Excel and laboratory exports sometimes prepend blank/comment lines and then
    # an Excel-style separator directive such as ``sep=;``.  Search the first
    # few physical lines instead of assuming the directive is line 1.
    for enc0 in ("utf-8-sig", "utf-8", "latin1"):
        try:
            decoded = [ln.decode(enc0, errors="strict").strip() for ln in raw_lines[:12]]
        except Exception:
            continue
        for i, txt in enumerate(decoded):
            if not txt:
                continue
            m = re.match(r"^\s*sep\s*=\s*[\"']?(.)[\"']?\s*$", txt, flags=re.IGNORECASE)
            if m:
                directive_sep = m.group(1)
                directive_skiprows = i + 1
                break
            # Stop once a plausible real tabular header is encountered.
            if any(d in txt for d in (",", ";", "\t")) and "sep" not in txt.lower():
                break
        if directive_sep is not None:
            break

    for enc in ("utf-8-sig", "utf-8", "latin1"):
        attempts = []
        if directive_sep is not None:
            # Respect the file's explicit delimiter first.  Try both decimal
            # conventions because semicolon-delimited European exports often
            # use decimal commas.
            attempts.extend([
                {"sep": directive_sep, "skiprows": directive_skiprows, "decimal": "."},
                {"sep": directive_sep, "skiprows": directive_skiprows, "decimal": ","},
            ])
        attempts.extend([
            {"sep": None, "engine": "python", "skiprows": directive_skiprows},
            {"sep": ",", "skiprows": directive_skiprows},
            {"sep": ";", "skiprows": directive_skiprows, "decimal": "."},
            {"sep": ";", "skiprows": directive_skiprows, "decimal": ","},
            {"sep": "\t", "skiprows": directive_skiprows},
        ])

        seen = set()
        for kwargs in attempts:
            key = tuple(sorted(kwargs.items()))
            if key in seen:
                continue
            seen.add(key)
            try:
                df = pd.read_csv(path, encoding=enc, **kwargs)
                if len(df.columns) == 0 or len(df) == 0:
                    continue

                sample = df.head(min(len(df), 300))
                numeric_cells = 0
                for c in sample.columns:
                    numeric_cells += int(pd.to_numeric(sample[c], errors="coerce").notna().sum())

                cleaned = [_clean_name(c) for c in df.columns]
                bad_sep_header = any(c in {"sep", "sep_"} or str(c).strip().lower().startswith("sep=") for c in df.columns)
                unnamed_fraction = sum(str(c).lower().startswith("unnamed") for c in df.columns) / max(len(df.columns), 1)

                has_monotonic_time = 0
                for c in df.columns:
                    if "time" not in _clean_name(c):
                        continue
                    x = pd.to_numeric(df[c], errors="coerce").dropna()
                    if len(x) >= 3 and x.is_monotonic_increasing and float(x.max() - x.min()) > 0:
                        has_monotonic_time = 1
                        break

                # Lexicographic score: valid time signal first, then avoid known
                # false headers, then prefer richer/numerically usable tables.
                score = (
                    has_monotonic_time,
                    0 if bad_sep_header else 1,
                    1 if unnamed_fraction < 0.5 else 0,
                    1 if len(df.columns) > 2 else 0,
                    len(df.columns),
                    numeric_cells,
                )
                candidates.append((score, df, enc, kwargs))
            except Exception as e:  # noqa: BLE001
                errors.append(f"{enc} {kwargs}: {e}")

    if candidates:
        candidates.sort(key=lambda x: x[0], reverse=True)
        best_score, best_df, best_enc, best_kwargs = candidates[0]
        # Refuse the exact failure signature observed in the E3-v2 audit rather
        # than propagating it to _time_column().
        if any(str(c).strip().lower().startswith("sep=") for c in best_df.columns):
            raise RuntimeError(
                f"CSV parser selected an invalid sep= directive header for {path}; "
                f"best parse encoding={best_enc}, kwargs={best_kwargs}, columns={list(best_df.columns)}"
            )
        return best_df

    raise RuntimeError(f"Could not parse CSV {path}: {' | '.join(errors[-8:])}")


def _numeric(s: pd.Series) -> pd.Series:
    """Robust numeric coercion for heterogeneous public CSV exports.

    Normal pandas coercion is attempted first.  If most values remain non-numeric
    and the source is string-like, retry common European decimal-comma and
    thousands-separator forms.  This is schema harmonization only; no physical
    values are rescaled here.
    """
    out = pd.to_numeric(s, errors="coerce")
    try:
        if len(s) == 0 or float(out.notna().mean()) >= 0.8:
            return out
    except Exception:
        return out
    if not (pd.api.types.is_object_dtype(s) or pd.api.types.is_string_dtype(s)):
        return out
    txt = s.astype("string").str.strip()
    # Decimal comma without decimal point: 12,34 -> 12.34.
    comma_decimal = txt.str.match(r"^[+-]?\d+(?:,\d+)?(?:[eE][+-]?\d+)?$", na=False)
    if comma_decimal.mean() > 0.5:
        alt = pd.to_numeric(txt.str.replace(",", ".", regex=False), errors="coerce")
        if alt.notna().sum() > out.notna().sum():
            out = alt
    # Common grouped form: 1 234,56 / 1,234.56.
    if float(out.notna().mean()) < 0.8:
        cleaned = txt.str.replace(" ", "", regex=False)
        # If both comma and dot exist, whichever appears last is usually decimal.
        def normalize_token(v):
            if v is pd.NA or v is None:
                return v
            v = str(v)
            if "," in v and "." in v:
                if v.rfind(",") > v.rfind("."):
                    v = v.replace(".", "").replace(",", ".")
                else:
                    v = v.replace(",", "")
            return v
        alt2 = pd.to_numeric(cleaned.map(normalize_token), errors="coerce")
        if alt2.notna().sum() > out.notna().sum():
            out = alt2
    return out


def _clean_name(x: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(x).strip().lower()).strip("_")


def _time_column(df: pd.DataFrame) -> str:
    named = [c for c in df.columns if "time" in _clean_name(c)]
    for c in named:
        x = _numeric(df[c]).dropna()
        if len(x) >= 2 and x.is_monotonic_increasing:
            return c
    # Fallback: monotonic numeric column with largest dynamic range.
    candidates = []
    for c in df.columns:
        x = _numeric(df[c]).dropna()
        if len(x) >= max(3, int(0.8 * len(df))) and x.is_monotonic_increasing:
            candidates.append((float(x.max() - x.min()), c))
    if candidates:
        return max(candidates)[1]
    raise ValueError(f"Could not identify a monotonic time column. Columns: {list(df.columns)}")


def _time_to_seconds(s: pd.Series, name: str, expected_duration_s: Optional[float] = None) -> np.ndarray:
    x = _numeric(s).to_numpy(float)
    n = _clean_name(name)
    if "hour" in n or n in {"time_h", "time_hr", "time_hours"}:
        return x * 3600.0
    if "min" in n and "time" in n:
        return x * 60.0
    # SNG official files use Time in hours but the column is simply "Time".
    finite = x[np.isfinite(x)]
    if finite.size and finite.max() <= 24 and finite.max() - finite.min() >= 1:
        if expected_duration_s and expected_duration_s > 3600:
            return x * 3600.0
    return x


def _robust_scale(values: np.ndarray, cfg: ProtocolConfig) -> float:
    v = np.asarray(values, float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return float("nan")
    qlo, qhi = np.quantile(v, [cfg.robust_scale_q_low, cfg.robust_scale_q_high])
    span = float(qhi - qlo)
    med = float(np.median(np.abs(v)))
    floor = max(1e-9, cfg.robust_scale_relative_floor * max(med, 1e-6))
    return max(span, floor)


def _to_kelvin(values: pd.Series) -> pd.Series:
    x = _numeric(values).astype(float)
    med = float(np.nanmedian(x.to_numpy())) if x.notna().any() else float("nan")
    if np.isfinite(med) and -80.0 < med < 180.0:
        return x + 273.15
    return x


def _schema_rows(dataset: str, path: Path, df: pd.DataFrame) -> List[dict]:
    rows = []
    for c in df.columns:
        x = _numeric(df[c])
        rows.append(
            {
                "dataset": dataset,
                "file": str(path),
                "column": c,
                "clean_column": _clean_name(c),
                "numeric_fraction": float(x.notna().mean()),
                "median_numeric": float(x.median()) if x.notna().any() else np.nan,
                "min_numeric": float(x.min()) if x.notna().any() else np.nan,
                "max_numeric": float(x.max()) if x.notna().any() else np.nan,
            }
        )
    return rows


def _finite_time_frame(df: pd.DataFrame, time_col: str, output_col: str = "time_s") -> pd.DataFrame:
    """Return a sorted, finite, duplicate-collapsed time series.

    This helper deliberately sanitizes merge/alignment keys before any nearest
    matching.  Public laboratory exports often include blank footer/header rows,
    NaN timestamps, repeated solver output times, or infinity sentinels.  Pandas
    ``merge_asof`` rejects NaNs in its key; downstream interpolation can also be
    unstable with duplicates.
    """
    out = df.copy()
    out[output_col] = _numeric(out[time_col]).astype(float)
    out = out.replace([np.inf, -np.inf], np.nan)
    out = out[out[output_col].notna()].copy()
    if out.empty:
        return out
    out = out.sort_values(output_col, kind="mergesort")
    # Keep the first complete row for duplicate timestamps at raw-schema level;
    # aggregate frames are collapsed separately by numeric mean below.
    out = out.drop_duplicates(subset=[output_col], keep="first").reset_index(drop=True)
    return out


def _collapse_numeric_time_duplicates(df: pd.DataFrame, time_col: str = "time_s") -> pd.DataFrame:
    """Sanitize a prepared numeric time frame and average duplicate samples."""
    if time_col not in df.columns:
        raise KeyError(f"Missing time column {time_col}")
    d = df.copy().replace([np.inf, -np.inf], np.nan)
    d[time_col] = _numeric(d[time_col]).astype(float)
    d = d[d[time_col].notna()].copy()
    if d.empty:
        return d
    d = d.sort_values(time_col, kind="mergesort")
    value_cols = [c for c in d.columns if c != time_col]
    for c in value_cols:
        d[c] = _numeric(d[c])
    d = d.groupby(time_col, as_index=False).mean(numeric_only=True)
    return d.sort_values(time_col, kind="mergesort").reset_index(drop=True)


def _median_positive_dt(values: pd.Series | np.ndarray) -> float:
    a = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    a = np.unique(a[np.isfinite(a)])
    if a.size < 2:
        return float("nan")
    dif = np.diff(np.sort(a))
    dif = dif[np.isfinite(dif) & (dif > 0)]
    return float(np.median(dif)) if dif.size else float("nan")


def _choose_time_scale(values: pd.Series | np.ndarray, expected_span_s: float, source: str) -> tuple[np.ndarray, float]:
    """Choose a conventional time-unit multiplier using only declared duration.

    FreeTwinEV filenames declare physical segment durations in seconds.  Some
    exports label time simply as ``Time`` while others include [s].  The scale
    candidates are unit conversions only (s, ms, min, h); the best candidate is
    the one whose observed span is closest to the declared segment duration.
    """
    raw = np.asarray(pd.to_numeric(pd.Series(values), errors="coerce"), dtype=float)
    finite = raw[np.isfinite(raw)]
    if finite.size < 2:
        raise ValueError(f"{source}: fewer than two finite time values")
    raw_span = float(np.nanmax(finite) - np.nanmin(finite))
    if raw_span <= 0:
        raise ValueError(f"{source}: non-positive time span")
    candidates = (1.0, 0.001, 60.0, 3600.0)
    def score(scale: float) -> float:
        span = raw_span * scale
        if span <= 0 or expected_span_s <= 0:
            return float("inf")
        return abs(math.log(max(span, 1e-12) / expected_span_s))
    scale = min(candidates, key=score)
    # Do not silently accept a wildly incompatible clock even after unit scaling.
    scaled_span = raw_span * scale
    ratio = max(scaled_span / expected_span_s, expected_span_s / scaled_span)
    if ratio > 20:
        raise ValueError(
            f"{source}: time span {raw_span:g} (best scale x{scale:g} -> {scaled_span:g}s) "
            f"is incompatible with expected segment span {expected_span_s:g}s"
        )
    return raw * scale, scale


def _nearest_time_alignment(
    simulation: pd.DataFrame,
    physical: pd.DataFrame,
    tolerance_s: float,
) -> pd.DataFrame:
    """Nearest-neighbor alignment without pandas merge-key failure modes.

    Simulation times anchor the aligned table.  Both time vectors must already
    be finite, sorted and unique.  Output columns are explicitly prefixed so
    downstream semantic pairing never depends on pandas suffix behavior.
    """
    s = _collapse_numeric_time_duplicates(simulation, "time_s")
    p = _collapse_numeric_time_duplicates(physical, "time_s")
    if s.empty or p.empty:
        raise ValueError("Cannot align empty FreeTwinEV time series")
    st = s["time_s"].to_numpy(float)
    pt = p["time_s"].to_numpy(float)
    idx = np.searchsorted(pt, st, side="left")
    right = np.clip(idx, 0, len(pt) - 1)
    left = np.clip(idx - 1, 0, len(pt) - 1)
    choose_right = np.abs(pt[right] - st) < np.abs(pt[left] - st)
    pi = np.where(choose_right, right, left)
    delta = np.abs(pt[pi] - st)
    keep = np.isfinite(delta) & (delta <= float(tolerance_s))
    if not keep.any():
        raise ValueError(
            f"FreeTwinEV nearest alignment found no matches within {tolerance_s:.3f}s; "
            f"simulation range={st.min():.3f}-{st.max():.3f}, physical range={pt.min():.3f}-{pt.max():.3f}"
        )
    sk = s.iloc[np.flatnonzero(keep)].reset_index(drop=True)
    pk = p.iloc[pi[keep]].reset_index(drop=True)
    out = pd.DataFrame({
        "time_s": sk["time_s"].to_numpy(float),
        "physical_time_s": pk["time_s"].to_numpy(float),
        "alignment_delta_s": delta[keep],
    })
    for c in sk.columns:
        if c != "time_s":
            out[f"sim__{c}"] = sk[c].to_numpy()
    for c in pk.columns:
        if c != "time_s":
            out[f"phys__{c}"] = pk[c].to_numpy()
    return out


def _freetwinev_pair_candidates(aligned: pd.DataFrame) -> List[tuple[str, str, str, str]]:
    """Return robust physical/simulation thermal aggregate pairings.

    Exact semantic pairs are preferred.  If one side exposes only generic
    thermal aggregates, pair by statistic (mean/max/spread) with a documented
    fallback.  This avoids relying on merge suffixes and keeps comparisons
    temperature-to-temperature only.
    """
    sim_bases = {c[len("sim__"):]: c for c in aligned.columns if c.startswith("sim__")}
    phys_bases = {c[len("phys__"):]: c for c in aligned.columns if c.startswith("phys__")}
    pairs: List[tuple[str, str, str, str]] = []
    used_stats = set()

    preferred = (
        "cell_mean_K", "cell_max_K", "cell_spread_K",
        "thermal_mean_K", "thermal_max_K", "thermal_spread_K",
        "plate_mean_K", "plate_max_K", "plate_spread_K",
    )
    for base in preferred:
        if base in sim_bases and base in phys_bases:
            stat = "spread" if "spread" in base else ("max" if "max" in base else "mean")
            if stat in used_stats:
                continue
            pairs.append((f"temperature_{stat}", phys_bases[base], sim_bases[base], f"exact:{base}"))
            used_stats.add(stat)

    def bases_for(stat: str, side: dict[str, str]) -> List[str]:
        return [b for b in preferred if stat in b and b in side]

    # Cross-label fallback by statistic only, preferring cell -> thermal -> plate.
    rank = {"cell": 0, "thermal": 1, "plate": 2}
    for stat in ("mean", "max", "spread"):
        if stat in used_stats:
            continue
        sb = bases_for(stat, sim_bases)
        pb = bases_for(stat, phys_bases)
        if not sb or not pb:
            continue
        sb.sort(key=lambda b: min((rank[k] for k in rank if b.startswith(k)), default=9))
        pb.sort(key=lambda b: min((rank[k] for k in rank if b.startswith(k)), default=9))
        sbase, pbase = sb[0], pb[0]
        pairs.append((f"temperature_{stat}", phys_bases[pbase], sim_bases[sbase], f"stat_fallback:{pbase}<->{sbase}"))
        used_stats.add(stat)
    return pairs


# -----------------------------------------------------------------------------
# MAGNET loader: forecast horizon contract on dependence-reduced windows
# -----------------------------------------------------------------------------

def load_magnet_contract_units(repo: Path, cfg: ProtocolConfig, allow_download: bool) -> Tuple[pd.DataFrame, List[dict], dict]:
    data_dir = repo / "magnet_tfp" / "data"
    physical_path = data_dir / "MAGNET_Heat_Pipe_2022-03-30.csv"
    forecast_path = data_dir / "ML_MAGNET_2022-03-30.csv"
    if not physical_path.exists() or not forecast_path.exists():
        for name, url in MAGNET_URLS.items():
            _ensure_download(data_dir / name, [url], f"MAGNET {name}", None, allow_download)

    physical = _read_csv_flexible(physical_path)
    forecast = _read_csv_flexible(forecast_path)
    missing_p = [c for c in [MAGNET_TIME] + MAGNET_SENSORS if c not in physical.columns]
    missing_f = [c for c in [MAGNET_TIME] + MAGNET_SENSORS if c not in forecast.columns]
    if missing_p or missing_f:
        raise ValueError(f"MAGNET schema mismatch. physical missing={missing_p}; forecast missing={missing_f}")

    window_rows = 600
    if len(forecast) % window_rows != 0:
        raise ValueError(f"MAGNET forecast row count {len(forecast)} is not divisible by 600")

    raw_windows = []
    strict_meta = []
    phys_times = _numeric(physical[MAGNET_TIME]).to_numpy(float)
    for wid in range(len(forecast) // window_rows):
        w = forecast.iloc[wid * window_rows:(wid + 1) * window_rows].copy().reset_index(drop=True)
        pred_present = w[MAGNET_SENSORS].notna().mean()
        strict_pred = bool((pred_present >= 0.95).all())
        pred_time = _numeric(w[MAGNET_TIME]).to_numpy(float)
        # Nearest physical row; released streams are essentially 1 Hz.
        idx = np.searchsorted(phys_times, pred_time)
        idx = np.clip(idx, 1, len(phys_times) - 1)
        left = idx - 1
        choose_right = np.abs(phys_times[idx] - pred_time) < np.abs(phys_times[left] - pred_time)
        pidx = np.where(choose_right, idx, left)
        dt = np.abs(phys_times[pidx] - pred_time)
        match_fraction = float(np.mean(dt <= 0.25))
        strict = strict_pred and match_fraction >= 0.99
        strict_meta.append((wid, float(pred_time[0]), float(pred_time[-1]), strict))
        if not strict:
            continue
        a = pd.DataFrame({"time_s": pred_time, "horizon_s": pred_time - pred_time[0]})
        for s in MAGNET_SENSORS:
            a[f"pred::{s}"] = _numeric(w[s]).to_numpy(float)
            a[f"phys::{s}"] = _numeric(physical.iloc[pidx][s]).to_numpy(float)
        a["window_id"] = wid
        raw_windows.append(a)

    # Greedy non-overlapping strict windows, matching the established MAGNET audit.
    selected = []
    last_end = -np.inf
    for wid, start, end, strict in sorted(strict_meta, key=lambda x: x[1]):
        if strict and start > last_end:
            selected.append(wid)
            last_end = end
    selected_set = set(selected)
    raw_windows = [a for a in raw_windows if int(a["window_id"].iloc[0]) in selected_set]
    if not raw_windows:
        raise RuntimeError("MAGNET yielded no strict non-overlapping windows")

    # Scales use only physical values in selected windows.
    scales = {}
    for s in MAGNET_SENSORS:
        vals = np.concatenate([a[f"phys::{s}"].to_numpy(float) for a in raw_windows])
        scales[s] = _robust_scale(vals, cfg)

    rows = []
    for a in raw_windows:
        wid = int(a["window_id"].iloc[0])
        for s in MAGNET_SENSORS:
            scale = scales[s]
            e = np.abs(a[f"pred::{s}"].to_numpy(float) - a[f"phys::{s}"].to_numpy(float)) / scale
            hvec = a["horizon_s"].to_numpy(float)
            for h in cfg.horizons_s:
                target = min(h, 599)
                mask = (hvec >= 0) & (hvec <= target + 1.0) & np.isfinite(e)
                if not mask.any():
                    continue
                metric = float(np.quantile(e[mask], cfg.contract_quantile))
                rows.append(
                    {
                        "dataset": "MAGNET",
                        "domain": "thermal_heat_pipe",
                        "subsystem": "10_thermowell_forecast",
                        "segment": f"forecast_window_{wid}",
                        "component": s,
                        "unit": "degC",
                        "horizon_s": h,
                        "n_points": int(mask.sum()),
                        "normalized_p95_error": metric,
                        "scale": scale,
                        "unit_id": f"{wid}:{s}",
                        "semantics": "forecast trajectory remains within normalized discrepancy over service horizon",
                    }
                )
    schema = _schema_rows("MAGNET_physical", physical_path, physical) + _schema_rows("MAGNET_forecast", forecast_path, forecast)
    meta = {
        "physical_file": str(physical_path),
        "forecast_file": str(forecast_path),
        "n_total_forecast_windows": len(strict_meta),
        "n_strict_nonoverlap_windows": len(selected),
        "components": len(MAGNET_SENSORS),
    }
    return pd.DataFrame(rows), schema, meta


# -----------------------------------------------------------------------------
# FreeTwinEV 1S4P loader
# -----------------------------------------------------------------------------

def _temperature_candidates(df: pd.DataFrame) -> List[str]:
    out = []
    for c in df.columns:
        n = _clean_name(c)
        if not any(tok in n for tok in ("temp", "temperature", "thermal", "cell_t", "plate_t")):
            continue
        x = _numeric(df[c])
        if x.notna().mean() < 0.5:
            continue
        med = float(x.median())
        if -100 < med < 500:
            out.append(c)
    return out


def _physical_freetwinev_aggregates(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = pd.DataFrame({"time_s": _time_to_seconds(df[time_col], time_col)})
    temp_cols = _temperature_candidates(df)
    if not temp_cols:
        # Conservative value-based fallback for exports whose temperature sensor
        # headers use channel IDs rather than the word ``temp``.  Explicit
        # electrical/flow names are excluded to avoid treating current/voltage
        # channels as Celsius merely because they share a numerical range.
        excluded = ("current", "volt", "power", "soc", "flow", "pressure", "speed", "velocity", "rpm", "mass", "density", "energy", "heat_flux", "htc")
        for c in df.columns:
            if c == time_col or any(tok in _clean_name(c) for tok in excluded):
                continue
            x = _numeric(df[c])
            if x.notna().mean() < 0.7:
                continue
            med = float(x.median())
            q05 = float(x.quantile(0.05))
            q95 = float(x.quantile(0.95))
            if (250 < med < 400) or (-50 < med < 130 and -100 < q05 < 200 and -100 < q95 < 220):
                temp_cols.append(c)
    cell_cols = [c for c in temp_cols if "cell" in _clean_name(c)]
    ambient_tokens = ("ambient", "inlet", "outlet", "liquid", "coolant")
    plate_cols = [c for c in temp_cols if c not in cell_cols and not any(t in _clean_name(c) for t in ambient_tokens)]

    def add_aggregates(prefix: str, cols: List[str]) -> None:
        if not cols:
            return
        mat = pd.concat([_to_kelvin(df[c]).rename(c) for c in cols], axis=1)
        out[f"{prefix}_mean_K"] = mat.mean(axis=1, skipna=True)
        out[f"{prefix}_max_K"] = mat.max(axis=1, skipna=True)
        if len(cols) >= 2:
            out[f"{prefix}_spread_K"] = mat.max(axis=1, skipna=True) - mat.min(axis=1, skipna=True)

    add_aggregates("cell", cell_cols)
    add_aggregates("plate", plate_cols)
    if not cell_cols and temp_cols:
        add_aggregates("thermal", temp_cols)
    return out


def _simulation_freetwinev_aggregates(df: pd.DataFrame, time_col: str) -> pd.DataFrame:
    out = pd.DataFrame({"sim_time_s": _time_to_seconds(df[time_col], time_col)})
    temp_cols = _temperature_candidates(df)
    # If explicit names fail, include numeric columns whose median is temperature-like.
    if not temp_cols:
        excluded = ("current", "volt", "power", "soc", "flow", "pressure", "speed", "velocity", "rpm", "mass", "density", "energy", "heat_flux", "htc", "residual")
        for c in df.columns:
            if c == time_col or any(tok in _clean_name(c) for tok in excluded):
                continue
            x = _numeric(df[c])
            if x.notna().mean() < 0.7:
                continue
            med = float(x.median())
            q05 = float(x.quantile(0.05))
            q95 = float(x.quantile(0.95))
            if (250 < med < 400) or (-50 < med < 130 and -100 < q05 < 200 and -100 < q95 < 220):
                temp_cols.append(c)
    if not temp_cols:
        raise ValueError("No plausible temperature outputs found in FreeTwinEV simulation CSV")

    cell_cols = [c for c in temp_cols if "cell" in _clean_name(c) or "battery" in _clean_name(c)]
    plate_cols = [c for c in temp_cols if any(t in _clean_name(c) for t in ("plate", "cool", "alum"))]

    def add(prefix: str, cols: List[str]) -> None:
        if not cols:
            return
        mat = pd.concat([_to_kelvin(df[c]).rename(c) for c in cols], axis=1)
        out[f"{prefix}_mean_K"] = mat.mean(axis=1, skipna=True)
        out[f"{prefix}_max_K"] = mat.max(axis=1, skipna=True)
        if len(cols) >= 2:
            out[f"{prefix}_spread_K"] = mat.max(axis=1, skipna=True) - mat.min(axis=1, skipna=True)

    add("cell", cell_cols)
    add("plate", plate_cols)
    # Universal fallback so at least mean/max thermal state can be paired.
    add("thermal", temp_cols)
    return out


def _align_sim_to_physical_segment(
    physical: pd.DataFrame,
    simulation: pd.DataFrame,
    start_s: float,
    end_s: float,
    segment_name: str = "segment",
) -> tuple[pd.DataFrame, dict]:
    """Robustly align a released FreeTwinEV simulation with its physical segment.

    The physical ID22 experiment uses the published segment ranges in seconds.
    Simulation exports may use a relative clock and may contain blank/duplicate
    solver rows.  We sanitize both clocks, infer only conventional time units,
    shift a relative simulation clock to the declared physical segment start,
    and perform explicit nearest-neighbor matching.
    """
    expected_span = float(end_s - start_s)
    if expected_span <= 0:
        raise ValueError(f"Invalid FreeTwinEV segment bounds {start_s}-{end_s}s")

    p = physical.copy()
    p["time_s"] = _numeric(p["time_s"]).astype(float)
    p = _collapse_numeric_time_duplicates(p, "time_s")
    if p.empty:
        raise ValueError("FreeTwinEV physical aggregate table has no finite timestamps")

    # Physical experiment filenames specify these exact ranges in seconds.  If
    # the full experiment clock was exported in another conventional unit, infer
    # the scale from the requirement that it covers the declared end time.
    pmax = float(p["time_s"].max())
    if pmax < end_s * 0.8 or pmax > end_s * 20.0:
        raw = p["time_s"].to_numpy(float)
        candidates = (1.0, 0.001, 60.0, 3600.0)
        viable = []
        for scale in candidates:
            scaled = raw * scale
            if np.nanmax(scaled) >= end_s * 0.98:
                # Prefer the smallest change and a plausible sample interval.
                dt = _median_positive_dt(scaled)
                penalty = abs(math.log(scale)) + (0 if np.isfinite(dt) and 0.001 <= dt <= 60 else 5)
                viable.append((penalty, scale, scaled))
        if viable:
            _, pscale, scaled = min(viable, key=lambda x: x[0])
            p["time_s"] = scaled
        else:
            pscale = 1.0
    else:
        pscale = 1.0

    p = p[(p["time_s"] >= start_s) & (p["time_s"] <= end_s)].copy()
    p = _collapse_numeric_time_duplicates(p, "time_s")
    if p.empty:
        raise ValueError(
            f"FreeTwinEV physical segment {start_s}-{end_s}s is empty after finite-time cleaning; "
            f"physical clock max before clipping={pmax:g}"
        )

    s = simulation.copy()
    sim_raw = _numeric(s["sim_time_s"]).astype(float)
    finite_raw = sim_raw[np.isfinite(sim_raw)]
    if len(finite_raw) < 2:
        raise ValueError(f"FreeTwinEV {segment_name} simulation has fewer than two finite timestamps")
    scaled_time, sscale = _choose_time_scale(sim_raw, expected_span, f"FreeTwinEV {segment_name} simulation")
    s["sim_time_s"] = scaled_time
    s = s[np.isfinite(s["sim_time_s"])].copy()
    s = s.sort_values("sim_time_s", kind="mergesort")
    # Collapse duplicate solver timestamps by averaging numeric aggregate outputs.
    tmp = s.rename(columns={"sim_time_s": "time_s"})
    tmp = _collapse_numeric_time_duplicates(tmp, "time_s")
    if tmp.empty:
        raise ValueError(f"FreeTwinEV {segment_name} simulation time is empty after cleaning")

    st = tmp["time_s"].to_numpy(float)
    sim_span = float(st.max() - st.min())
    # Determine whether simulation time is absolute or relative from overlap with
    # the declared physical segment.  No fit to temperature outcomes is used.
    overlap_absolute = max(0.0, min(float(st.max()), end_s) - max(float(st.min()), start_s))
    relative_like = (
        overlap_absolute < 0.25 * expected_span
        and sim_span <= expected_span * 1.5
    )
    sim_shift = 0.0
    if relative_like:
        sim_shift = start_s - float(st.min())
        tmp["time_s"] = tmp["time_s"] + sim_shift

    # Retain only simulation points reasonably associated with the target segment.
    pad = max(15.0, 0.02 * expected_span)
    tmp = tmp[(tmp["time_s"] >= start_s - pad) & (tmp["time_s"] <= end_s + pad)].copy()
    tmp = _collapse_numeric_time_duplicates(tmp, "time_s")
    if tmp.empty:
        raise ValueError(
            f"FreeTwinEV {segment_name} simulation does not overlap declared physical segment "
            f"{start_s}-{end_s}s after time normalization (scale={sscale}, shift={sim_shift})"
        )

    pdt = _median_positive_dt(p["time_s"])
    sdt = _median_positive_dt(tmp["time_s"])
    finite_dts = [x for x in (pdt, sdt) if np.isfinite(x) and x > 0]
    base_dt = max(finite_dts) if finite_dts else 1.0
    tolerance_s = min(max(base_dt * 1.75, 1.0), 30.0)
    aligned = _nearest_time_alignment(tmp, p, tolerance_s=tolerance_s)

    coverage = float(len(aligned) / max(len(tmp), 1))
    meta = {
        "segment": segment_name,
        "physical_start_s": float(start_s),
        "physical_end_s": float(end_s),
        "physical_time_scale": float(pscale),
        "simulation_time_scale": float(sscale),
        "simulation_time_shift_s": float(sim_shift),
        "relative_clock_detected": bool(relative_like),
        "physical_rows_clean": int(len(p)),
        "simulation_rows_clean": int(len(tmp)),
        "aligned_rows": int(len(aligned)),
        "alignment_coverage": coverage,
        "alignment_tolerance_s": float(tolerance_s),
        "alignment_delta_median_s": float(aligned["alignment_delta_s"].median()),
        "alignment_delta_p95_s": float(aligned["alignment_delta_s"].quantile(0.95)),
    }
    if coverage < 0.25:
        raise ValueError(
            f"FreeTwinEV {segment_name} alignment coverage is only {coverage:.1%}; "
            f"clock normalization likely failed. Diagnostics={meta}"
        )
    return aligned, meta


def load_freetwinev_contract_units(repo: Path, cfg: ProtocolConfig, allow_download: bool) -> Tuple[pd.DataFrame, List[dict], dict, pd.DataFrame]:
    data_root = repo / "public_datasets" / "freetwinev_1s4p"
    zip_path = data_root / FREETWINEV_ZIP
    _ensure_download(zip_path, FREETWINEV_URLS, "FreeTwinEV 1S4P identification/validation simulation package", FREETWINEV_MD5, allow_download)
    extract_root = _extract_zip_once(zip_path, data_root / "extracted")

    exp_files = list(extract_root.rglob("FreeTwinEV_1s4p_ID_22.csv"))
    cd_files = list(extract_root.rglob("CASE_ID22_CD_2DresultsFULL.csv"))
    dis_files = list(extract_root.rglob("CASE_ID22_DIS_2DresultsFULL.csv"))
    if len(exp_files) != 1 or len(cd_files) != 1 or len(dis_files) != 1:
        raise FileNotFoundError(
            "FreeTwinEV package layout mismatch. Expected exactly one ID22 experiment CSV and one CD/DIS 2DresultsFULL CSV."
        )
    exp_path, cd_path, dis_path = exp_files[0], cd_files[0], dis_files[0]
    exp = _read_csv_flexible(exp_path)
    cd = _read_csv_flexible(cd_path)
    dis = _read_csv_flexible(dis_path)
    exp_t = _time_column(exp)
    cd_t = _time_column(cd)
    dis_t = _time_column(dis)
    p = _physical_freetwinev_aggregates(exp, exp_t)
    cd_s = _simulation_freetwinev_aggregates(cd, cd_t)
    dis_s = _simulation_freetwinev_aggregates(dis, dis_t)

    audits = []
    pair_rows = []
    segments = [
        ("cooldown", cd_s, cfg.freetwinev_cd_physical_start_s, cfg.freetwinev_cd_physical_end_s),
        ("discharge", dis_s, cfg.freetwinev_dis_physical_start_s, cfg.freetwinev_dis_physical_end_s),
    ]
    for seg_name, sim, start_s, end_s in segments:
        m, align_meta = _align_sim_to_physical_segment(p, sim, start_s, end_s, segment_name=seg_name)
        candidate_pairs = _freetwinev_pair_candidates(m)
        if not candidate_pairs:
            sim_cols = [c for c in m.columns if c.startswith("sim__")]
            phys_cols = [c for c in m.columns if c.startswith("phys__")]
            raise RuntimeError(
                "FreeTwinEV simulation/measurement temperature aggregates could not be paired after alignment. "
                f"simulation aggregates={sim_cols}; physical aggregates={phys_cols}; alignment={align_meta}"
            )

        for comp, phys_col, sim_col, pair_rule in candidate_pairs:
            valid = m[["time_s", "physical_time_s", "alignment_delta_s", phys_col, sim_col]].copy()
            for c in (phys_col, sim_col):
                valid[c] = _numeric(valid[c])
            valid = valid.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s", phys_col, sim_col])
            coverage = len(valid) / max(len(m), 1)
            audits.append(
                {
                    **align_meta,
                    "component": comp,
                    "physical_column": phys_col,
                    "simulation_column": sim_col,
                    "pair_rule": pair_rule,
                    "n_component_pairs": int(len(valid)),
                    "component_pair_coverage": float(coverage),
                    "physical_median_K": float(valid[phys_col].median()) if len(valid) else np.nan,
                    "simulation_median_K": float(valid[sim_col].median()) if len(valid) else np.nan,
                }
            )
            if len(valid) < 2:
                continue
            for _, r in valid.iterrows():
                pair_rows.append(
                    {
                        "dataset": "FreeTwinEV_1S4P",
                        "domain": "battery_electrothermal",
                        "subsystem": "ID22_3D_CFD_validation",
                        "segment": seg_name,
                        "component": comp,
                        "unit": "K",
                        "time_s": float(r["time_s"]),
                        "physical": float(r[phys_col]),
                        "twin": float(r[sim_col]),
                    }
                )

    paired = pd.DataFrame(pair_rows)
    if paired.empty:
        raise RuntimeError("FreeTwinEV produced no aligned physical/simulation pairs")
    units = continuous_pairs_to_contract_units(paired, cfg)
    schema = (
        _schema_rows("FreeTwinEV_experiment_ID22", exp_path, exp)
        + _schema_rows("FreeTwinEV_simulation_cooldown", cd_path, cd)
        + _schema_rows("FreeTwinEV_simulation_discharge", dis_path, dis)
    )
    meta = {
        "record": FREETWINEV_ZENODO_RECORD,
        "zip": str(zip_path),
        "experiment_csv": str(exp_path),
        "cooldown_sim_csv": str(cd_path),
        "discharge_sim_csv": str(dis_path),
        "paired_components": sorted(paired["component"].unique().tolist()),
        "n_paired_rows": int(len(paired)),
    }
    return units, schema, meta, pd.DataFrame(audits)


def diagnose_freetwinev_inputs(repo: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Best-effort schema/time diagnostics that never affect scientific results."""
    extract_root = repo / "public_datasets" / "freetwinev_1s4p" / "extracted"
    specs = [
        ("experiment", "FreeTwinEV_1s4p_ID_22.csv"),
        ("cooldown_sim", "CASE_ID22_CD_2DresultsFULL.csv"),
        ("discharge_sim", "CASE_ID22_DIS_2DresultsFULL.csv"),
    ]
    schema_rows: List[dict] = []
    summary_rows: List[dict] = []
    for label, name in specs:
        matches = list(extract_root.rglob(name)) if extract_root.exists() else []
        if len(matches) != 1:
            summary_rows.append({"source": label, "file": name, "status": f"found_{len(matches)}"})
            continue
        path = matches[0]
        try:
            df = _read_csv_flexible(path)
            schema_rows.extend(_schema_rows(f"FreeTwinEV_diag_{label}", path, df))
            tc = _time_column(df)
            tv = _numeric(df[tc])
            finite = tv[np.isfinite(tv)]
            temp_cols = _temperature_candidates(df)
            row = {
                "source": label,
                "file": str(path),
                "status": "parsed",
                "rows": int(len(df)),
                "columns": int(len(df.columns)),
                "time_column": tc,
                "time_finite": int(finite.size),
                "time_null_or_nonfinite": int(len(df) - finite.size),
                "time_min": float(finite.min()) if finite.size else np.nan,
                "time_max": float(finite.max()) if finite.size else np.nan,
                "time_median_positive_dt": _median_positive_dt(finite),
                "temperature_candidate_count": int(len(temp_cols)),
                "temperature_candidates": ";".join(map(str, temp_cols[:50])),
            }
            summary_rows.append(row)
        except Exception as e:  # noqa: BLE001
            summary_rows.append({
                "source": label, "file": str(path), "status": "diagnostic_failed", "error": str(e)
            })
    return pd.DataFrame(schema_rows), pd.DataFrame(summary_rows)


# -----------------------------------------------------------------------------
# TU Wien biomass-to-SNG loader
# -----------------------------------------------------------------------------

def _ensure_sng_files(repo: Path, allow_download: bool) -> Dict[str, Path]:
    root = repo / "public_datasets" / "tuwien_sng"
    root.mkdir(parents=True, exist_ok=True)
    paths = {}
    for name, md5 in SNG_FILES.items():
        urls = [u.format(name=name) for u in SNG_BASE_URLS]
        p = root / name
        _ensure_download(p, urls, f"TU Wien SNG {name}", md5, allow_download)
        paths[name] = p
    return paths


def _align_sng_softsensor(syngas: pd.DataFrame, soft: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    syn = syngas.copy()
    ss = soft.copy()
    syn["time_s"] = _numeric(syn["Time"]) * 3600.0
    ss["time_s"] = _numeric(ss["Time"]) * 3600.0
    # Sanitize as-of keys explicitly; this also protects the SNG path from the
    # same blank-timestamp export pattern observed in FreeTwinEV.
    syn = syn.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s"]).sort_values("time_s")
    ss = ss.replace([np.inf, -np.inf], np.nan).dropna(subset=["time_s"]).sort_values("time_s")
    syn = syn.drop_duplicates(subset=["time_s"], keep="first")
    ss = ss.drop_duplicates(subset=["time_s"], keep="first")
    merged = pd.merge_asof(ss, syn, on="time_s", direction="nearest", tolerance=90.0, suffixes=("_soft", "_meas"))
    rows = []
    audit = []
    for gas in ("H2", "CO", "CO2", "CH4", "C2H4"):
        pcol = f"Plt5_PG_{gas}"
        tcol = f"y_{gas}_pg_wf_filter"
        if pcol not in merged.columns or tcol not in merged.columns:
            audit.append({"component": f"PG_{gas}", "status": "missing", "physical_column": pcol, "twin_column": tcol})
            continue
        d = merged[["time_s", pcol, tcol]].apply(pd.to_numeric, errors="coerce").dropna()
        # The online soft-sensor channels are fractions (e.g. ~0.41 H2),
        # whereas Plt5_PG_* measurements are reported on a percent scale
        # (e.g. ~38--55 H2).  Harmonize units before computing discrepancy.
        twin_scale_factor = 100.0
        audit.append({
            "component": f"PG_{gas}",
            "status": "paired",
            "physical_column": pcol,
            "twin_column": tcol,
            "n": len(d),
            "twin_scale_factor": twin_scale_factor,
            "unit_harmonization": "soft-sensor fraction -> percent",
        })
        for _, r in d.iterrows():
            rows.append(
                {
                    "dataset": "TUWien_SNG",
                    "domain": "biomass_to_sng_process",
                    "subsystem": "soft_sensor_product_gas",
                    "segment": "campaign_2023_excerpt",
                    "component": f"PG_{gas}",
                    "unit": "percent_scale",
                    "time_s": float(r["time_s"]),
                    "physical": float(r[pcol]),
                    "twin": float(r[tcol]) * twin_scale_factor,
                }
            )
    return pd.DataFrame(rows), audit


def _sng_dfb_pairs(dfb: pd.DataFrame) -> Tuple[pd.DataFrame, List[dict]]:
    d = dfb.copy()
    d["time_s"] = _numeric(d["Time"]) * 3600.0
    specs = [
        ("PG_volume_flow", "Plt1_PGVolFlow_Measurement", "Plt1_PGVolFlow_Estimate", "Nm3_per_h"),
        ("PG_temperature", "Plt3_GasTemp_Measurement", "Plt3_GasTemp_Estimate", "degC"),
    ]
    rows = []
    audit = []
    for comp, pcol, tcol, unit in specs:
        if pcol not in d.columns or tcol not in d.columns:
            audit.append({"component": comp, "status": "missing", "physical_column": pcol, "twin_column": tcol})
            continue
        z = d[["time_s", pcol, tcol]].apply(pd.to_numeric, errors="coerce").dropna()
        audit.append({"component": comp, "status": "paired", "physical_column": pcol, "twin_column": tcol, "n": len(z)})
        for _, r in z.iterrows():
            rows.append(
                {
                    "dataset": "TUWien_SNG",
                    "domain": "biomass_to_sng_process",
                    "subsystem": "mpc_kalman_state",
                    "segment": "campaign_2023_excerpt",
                    "component": comp,
                    "unit": unit,
                    "time_s": float(r["time_s"]),
                    "physical": float(r[pcol]),
                    "twin": float(r[tcol]),
                }
            )
    return pd.DataFrame(rows), audit


def load_sng_contract_units(repo: Path, cfg: ProtocolConfig, allow_download: bool) -> Tuple[pd.DataFrame, List[dict], dict, pd.DataFrame]:
    paths = _ensure_sng_files(repo, allow_download)
    dfb = _read_csv_flexible(paths["Data_MPC_DFB.csv"])
    syngas = _read_csv_flexible(paths["Data_MPC_Syngas.csv"])
    soft = _read_csv_flexible(paths["Data_SoftSensor.csv"])
    d1, a1 = _sng_dfb_pairs(dfb)
    d2, a2 = _align_sng_softsensor(syngas, soft)
    paired = pd.concat([d1, d2], ignore_index=True)
    if paired.empty:
        raise RuntimeError("TU Wien SNG produced no physical/virtual pairs")
    units = continuous_pairs_to_contract_units(paired, cfg)
    schema = (
        _schema_rows("SNG_DFB", paths["Data_MPC_DFB.csv"], dfb)
        + _schema_rows("SNG_Syngas", paths["Data_MPC_Syngas.csv"], syngas)
        + _schema_rows("SNG_SoftSensor", paths["Data_SoftSensor.csv"], soft)
    )
    meta = {
        "record": SNG_RECORD,
        "files": {k: str(v) for k, v in paths.items()},
        "paired_components": sorted(paired["component"].unique().tolist()),
        "n_paired_rows": int(len(paired)),
    }
    return units, schema, meta, pd.DataFrame(a1 + a2)


# -----------------------------------------------------------------------------
# Shared contract evaluation
# -----------------------------------------------------------------------------

def continuous_pairs_to_contract_units(paired: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    """Convert synchronized physical/twin time-series into the frozen service contract.

    For each component and non-overlapping horizon-length interval, compute the
    q95 normalized absolute physical-virtual discrepancy. A contract at tolerance
    tau passes iff this interval statistic <= tau.
    """
    d = paired.copy()
    rows = []
    group_cols = ["dataset", "domain", "subsystem", "segment", "component", "unit"]
    for keys, g in d.groupby(group_cols, dropna=False):
        meta = dict(zip(group_cols, keys))
        g = g[["time_s", "physical", "twin"]].replace([np.inf, -np.inf], np.nan).dropna().sort_values("time_s")
        if len(g) < 2:
            continue
        # Collapse exact duplicate timestamps before binning.
        g = g.groupby("time_s", as_index=False).mean(numeric_only=True)
        scale = _robust_scale(g["physical"].to_numpy(float), cfg)
        if not np.isfinite(scale) or scale <= 0:
            continue
        g["norm_abs_error"] = np.abs(g["twin"] - g["physical"]) / scale
        t0 = float(g["time_s"].min())
        for h in cfg.horizons_s:
            g2 = g.copy()
            g2["window_index"] = np.floor((g2["time_s"] - t0) / float(h)).astype(int)
            for wi, w in g2.groupby("window_index"):
                vals = w["norm_abs_error"].to_numpy(float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    continue
                rows.append(
                    {
                        **meta,
                        "horizon_s": h,
                        "n_points": int(vals.size),
                        "normalized_p95_error": float(np.quantile(vals, cfg.contract_quantile)),
                        "scale": scale,
                        "unit_id": f"{meta['segment']}:{meta['component']}:{h}:{int(wi)}",
                        "semantics": "synchronized twin remains within normalized discrepancy over service horizon",
                    }
                )
    return pd.DataFrame(rows)


def build_contract_grid(units: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    rows = []
    for (dataset, domain, subsystem, horizon), g in units.groupby(["dataset", "domain", "subsystem", "horizon_s"]):
        e = pd.to_numeric(g["normalized_p95_error"], errors="coerce").dropna().to_numpy(float)
        comps = int(g["component"].nunique())
        for tau in cfg.normalized_tolerances:
            rows.append(
                {
                    "dataset": dataset,
                    "domain": domain,
                    "subsystem": subsystem,
                    "horizon_s": int(horizon),
                    "normalized_tolerance": float(tau),
                    "n_contract_units": int(len(e)),
                    "n_components": comps,
                    "service_validity": float(np.mean(e <= tau)) if len(e) else np.nan,
                    "median_normalized_p95_error": float(np.median(e)) if len(e) else np.nan,
                }
            )
    return pd.DataFrame(rows)


def build_dataset_macro(grid: pd.DataFrame) -> pd.DataFrame:
    return (
        grid.groupby(["dataset", "domain", "horizon_s", "normalized_tolerance"], as_index=False)
        .agg(
            service_validity=("service_validity", "mean"),
            n_subsystems=("subsystem", "nunique"),
            n_contract_units=("n_contract_units", "sum"),
            n_components=("n_components", "sum"),
        )
    )


def build_grid_average(macro: pd.DataFrame) -> pd.DataFrame:
    return (
        macro.groupby(["dataset", "domain", "horizon_s"], as_index=False)
        .agg(
            grid_average_validity=("service_validity", "mean"),
            min_validity=("service_validity", "min"),
            max_validity=("service_validity", "max"),
            total_contract_units=("n_contract_units", "max"),
        )
    )


def transfer_gates(units: pd.DataFrame, grid: pd.DataFrame, cfg: ProtocolConfig) -> pd.DataFrame:
    rows = []
    for dataset, u in units.groupby("dataset"):
        g = grid[grid["dataset"] == dataset]
        components = int(u["component"].nunique())
        horizons_ok = 0
        horizon_details = []
        for h in cfg.horizons_s:
            n = int((u["horizon_s"] == h).sum())
            ok = n >= cfg.min_units_per_horizon
            horizons_ok += int(ok)
            horizon_details.append(f"{h}s:{n}")
        vals = g["service_validity"].dropna().to_numpy(float)
        nondegenerate = bool(len(vals) and (np.nanmax(vals) - np.nanmin(vals) > 0.05))
        passed = (
            components >= cfg.min_components_for_transfer
            and horizons_ok >= cfg.min_horizons_for_transfer
            and nondegenerate
        )
        rows.append(
            {
                "dataset": dataset,
                "n_components": components,
                "horizons_meeting_min_units": horizons_ok,
                "required_horizons": cfg.min_horizons_for_transfer,
                "contract_units_by_horizon": ";".join(horizon_details),
                "surface_nondegenerate": nondegenerate,
                "structural_transfer_pass": passed,
            }
        )
    return pd.DataFrame(rows)


def _make_figures(out: Path, gridavg: pd.DataFrame, macro: pd.DataFrame) -> None:
    if plt is None or gridavg.empty:
        return
    figdir = out / "figures"
    figdir.mkdir(parents=True, exist_ok=True)

    # One line per dataset: average across the complete frozen tolerance grid.
    plt.figure(figsize=(8.4, 5.2))
    for dataset, d in gridavg.groupby("dataset"):
        d = d.sort_values("horizon_s")
        plt.plot(d["horizon_s"] / 60.0, d["grid_average_validity"], marker="o", label=dataset)
    plt.xlabel("Service horizon (min)")
    plt.ylabel("Grid-average contract validity")
    plt.ylim(0, 1)
    plt.title("Same normalized service-contract structure across non-robot DT domains")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figdir / "cross_domain_horizon_profile.png", dpi=200)
    plt.close()

    # Dataset x horizon heatmap.
    p = gridavg.pivot(index="dataset", columns="horizon_s", values="grid_average_validity")
    if not p.empty:
        plt.figure(figsize=(7.2, 3.8))
        im = plt.imshow(p.to_numpy(), aspect="auto", vmin=0, vmax=1)
        plt.colorbar(im, label="Grid-average contract validity")
        plt.yticks(np.arange(len(p.index)), p.index)
        plt.xticks(np.arange(len(p.columns)), [f"{int(h/60)} min" for h in p.columns])
        plt.xlabel("Service horizon")
        plt.ylabel("Dataset/domain")
        plt.title("Cross-domain operational-fidelity contract map")
        plt.tight_layout()
        plt.savefig(figdir / "cross_domain_contract_heatmap.png", dpi=200)
        plt.close()

    # Tolerance surfaces as one panel per dataset, saved independently (no subplots).
    for dataset, d in macro.groupby("dataset"):
        plt.figure(figsize=(7.2, 4.8))
        for h, hdf in d.groupby("horizon_s"):
            hdf = hdf.sort_values("normalized_tolerance")
            plt.plot(hdf["normalized_tolerance"], hdf["service_validity"], marker="o", label=f"{int(h/60)} min")
        plt.xscale("log")
        plt.ylim(0, 1)
        plt.xlabel("Normalized tolerance (fraction of physical p95-p05 scale)")
        plt.ylabel("Contract validity")
        plt.title(f"{dataset}: service validity across tolerance and horizon")
        plt.legend(title="Horizon")
        plt.tight_layout()
        safe = re.sub(r"[^A-Za-z0-9]+", "_", dataset).strip("_")
        plt.savefig(figdir / f"{safe}_contract_surface.png", dpi=200)
        plt.close()


def _report(out: Path, cfg: ProtocolConfig, gates: pd.DataFrame, gridavg: pd.DataFrame, meta: dict, errors: dict) -> None:
    passed = int(gates["structural_transfer_pass"].sum()) if not gates.empty else 0
    expected_total = 3
    completed = len(gates)
    all_pass = completed == expected_total and passed == expected_total
    lines = [
        "# Cross-domain operational-fidelity contract audit",
        "",
        f"**Structural transfer verdict: {'PASS' if all_pass else 'INCOMPLETE / NO-GO'} ({passed}/{expected_total} required datasets passed; {completed}/{expected_total} completed)**",
        "",
        "## Frozen question",
        "",
        "Does the same *quantity × horizon × tolerance* fidelity-contract structure remain computable and informative in independent non-robot digital-twin domains without tuning thresholds to outcomes?",
        "",
        "## Frozen contract",
        "",
        "For each physical quantity, normalize absolute physical–virtual discrepancy by that quantity's physical p95−p05 range (with only a small scale floor for near-constant channels). For a service horizon h, compute the p95 normalized discrepancy inside each non-overlapping h-second service window. A contract passes at normalized tolerance τ when that window statistic is ≤ τ.",
        "",
        f"- Horizons: {', '.join(str(x) + ' s' for x in cfg.horizons_s)}.",
        f"- Normalized tolerance sweep: {', '.join(f'{x:g}' for x in cfg.normalized_tolerances)}.",
        f"- Within-window error statistic: q={cfg.contract_quantile:.2f}.",
        "- The numerical tolerance grid is dimensionless; no dataset-specific error threshold is tuned after observing results.",
        "",
        "## Dataset transfer gates",
        "",
    ]
    if gates.empty:
        lines.append("No datasets completed.")
    else:
        lines.append(gates.to_markdown(index=False))
    lines += ["", "## Grid-average descriptive validity", ""]
    if not gridavg.empty:
        show = gridavg[["dataset", "horizon_s", "grid_average_validity", "total_contract_units"]].copy()
        lines.append(show.to_markdown(index=False, floatfmt=".4f"))
    lines += ["", "## Dataset execution status", ""]
    if errors:
        for ds, err in errors.items():
            lines.append(f"- **{ds}: FAILED** — `{err}`")
    else:
        lines.append("- All three required datasets completed without loader errors.")
    lines += [
        "",
        "## Interpretation boundary",
        "",
        "A structural PASS means the unchanged contract abstraction produces complete, non-degenerate validity surfaces over all frozen horizons in that dataset. It is **not** a claim that the underlying domain model is superior, that normalized tolerances are safety standards, or that all domains should share identical physical-unit tolerances.",
        "",
        "MAGNET remains the strongest inferential non-robot transfer because its existing hardened study includes dependence-reduced forecast windows and horizon statistics. FreeTwinEV and TU Wien SNG are used here to test broader contract portability across battery electro-thermal and industrial-process digital-twin data.",
        "",
        "## Dataset-specific semantics",
        "",
        "- **MAGNET:** released 10-thermowell physical experiment versus digital-twin forecast windows; dependence-reduced forecast windows are used.",
        "- **FreeTwinEV 1S4P:** released ID22 experiment versus the released identification/validation 3D CFD cooldown/discharge simulation; thermal aggregate quantities are paired without calibrating the simulation in this script.",
        "- **TU Wien SNG:** measured versus Kalman-estimated DFB process states plus measured product-gas composition versus the released soft-sensor outputs from the same 9.5 h DT campaign.",
        "",
        "## Publication use",
        "",
        "If all three transfer gates pass, the defensible contribution is **cross-domain portability of the service-contract representation**. Do not claim universal fidelity, universal safety tolerances, or model transfer superiority.",
    ]
    (out / "CROSS_DOMAIN_GENERALIZATION_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def run(repo: Path, out: Path, allow_download: bool = True) -> dict:
    cfg = ProtocolConfig()
    # Results are fully regenerated on every run so stale files from an earlier
    # failed/partial audit cannot be mistaken for current evidence.
    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    units_all = []
    schemas: List[dict] = []
    metadata = {}
    errors = {}
    error_traces = {}

    print("[1/3] MAGNET contract transfer")
    try:
        u, s, m = load_magnet_contract_units(repo, cfg, allow_download)
        units_all.append(u)
        schemas += s
        metadata["MAGNET"] = m
    except Exception as e:  # noqa: BLE001
        errors["MAGNET"] = str(e)
        error_traces["MAGNET"] = traceback.format_exc()
        print(f"[MAGNET] FAILED: {e}")

    print("[2/3] FreeTwinEV 1S4P contract transfer")
    try:
        u, s, m, audit = load_freetwinev_contract_units(repo, cfg, allow_download)
        units_all.append(u)
        schemas += s
        metadata["FreeTwinEV_1S4P"] = m
        audit.to_csv(out / "freetwinev_pairing_audit.csv", index=False)
    except Exception as e:  # noqa: BLE001
        errors["FreeTwinEV_1S4P"] = str(e)
        error_traces["FreeTwinEV_1S4P"] = traceback.format_exc()
        print(f"[FreeTwinEV] FAILED: {e}")
        # Always emit a schema/time diagnostic for the real downloaded archive,
        # even when failure occurs after parsing.  This prevents opaque iteration.
        try:
            diag_schema, diag_summary = diagnose_freetwinev_inputs(repo)
            diag_schema.to_csv(out / "freetwinev_schema_diagnostic.csv", index=False)
            diag_summary.to_csv(out / "freetwinev_time_diagnostic.csv", index=False)
        except Exception as diag_e:  # noqa: BLE001
            error_traces["FreeTwinEV_diagnostic"] = traceback.format_exc()
            print(f"[FreeTwinEV diagnostic] FAILED: {diag_e}")

    print("[3/3] TU Wien SNG contract transfer")
    try:
        u, s, m, audit = load_sng_contract_units(repo, cfg, allow_download)
        units_all.append(u)
        schemas += s
        metadata["TUWien_SNG"] = m
        audit.to_csv(out / "sng_pairing_audit.csv", index=False)
    except Exception as e:  # noqa: BLE001
        errors["TUWien_SNG"] = str(e)
        error_traces["TUWien_SNG"] = traceback.format_exc()
        print(f"[SNG] FAILED: {e}")

    pd.DataFrame(schemas).to_csv(out / "input_schema_audit.csv", index=False)
    if units_all:
        units = pd.concat(units_all, ignore_index=True)
    else:
        units = pd.DataFrame(columns=["dataset", "domain", "subsystem", "segment", "component", "unit", "horizon_s", "n_points", "normalized_p95_error", "scale", "unit_id", "semantics"])
    units.to_csv(out / "cross_domain_contract_units.csv", index=False)
    grid = build_contract_grid(units, cfg) if not units.empty else pd.DataFrame()
    grid.to_csv(out / "cross_domain_contract_grid.csv", index=False)
    macro = build_dataset_macro(grid) if not grid.empty else pd.DataFrame()
    macro.to_csv(out / "cross_domain_contract_macro.csv", index=False)
    gridavg = build_grid_average(macro) if not macro.empty else pd.DataFrame()
    gridavg.to_csv(out / "cross_domain_horizon_grid_average.csv", index=False)
    gates = transfer_gates(units, grid, cfg) if not units.empty else pd.DataFrame()
    gates.to_csv(out / "cross_domain_transfer_gates.csv", index=False)
    _make_figures(out, gridavg, macro)
    pd.DataFrame([{"dataset": k, "error": v} for k, v in errors.items()]).to_csv(out / "dataset_errors.csv", index=False)
    if error_traces:
        (out / "dataset_error_tracebacks.txt").write_text(
            "\n\n".join(f"===== {k} =====\n{v}" for k, v in error_traces.items()),
            encoding="utf-8",
        )
    _report(out, cfg, gates, gridavg, metadata, errors)

    manifest = {
        "analysis": "cross_domain_operational_fidelity_contract",
        "protocol_version": "E3_cross_domain_contract_v4_hardened_alignment",
        "config": asdict(cfg),
        "claim_boundary": "contract-structure portability across independent DT domains; not universal fidelity or target-domain model superiority",
        "datasets": {
            "MAGNET": {
                "source": "Idaho National Laboratory MAGNET Heat Pipe Digital Twin",
                "official_urls": MAGNET_URLS,
            },
            "FreeTwinEV_1S4P": {
                "zenodo_record": FREETWINEV_ZENODO_RECORD,
                "doi": "10.5281/zenodo.19935693",
                "official_md5": FREETWINEV_MD5,
                "official_urls": list(FREETWINEV_URLS),
            },
            "TUWien_SNG": {
                "record": SNG_RECORD,
                "doi": "10.48436/6mmjq-1tj37",
                "official_md5": SNG_FILES,
            },
        },
        "metadata": metadata,
        "errors": errors,
        "completed_datasets": sorted(units["dataset"].unique().tolist()) if not units.empty else [],
        "structural_pass_count": int(gates["structural_transfer_pass"].sum()) if not gates.empty else 0,
        "all_three_pass": bool(len(gates) >= 3 and gates["structural_transfer_pass"].all()),
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print("\n============================================================")
    print("Cross-domain contract audit complete")
    print(f"Results: {out.resolve()}")
    if errors:
        print("Dataset errors:")
        for k, v in errors.items():
            print(f"  - {k}: {v}")
    if not gates.empty:
        print(gates[["dataset", "structural_transfer_pass"]].to_string(index=False))
    print("============================================================")
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description="Cross-domain operational-fidelity contract audit: MAGNET + FreeTwinEV 1S4P + TU Wien SNG")
    ap.add_argument("--repo-root", type=Path, default=Path.cwd())
    ap.add_argument("--output", type=Path, default=None)
    ap.add_argument("--no-download", action="store_true", help="Do not download missing public datasets")
    args = ap.parse_args()
    repo = args.repo_root.resolve()
    out = args.output.resolve() if args.output else (repo / "results" / "cross_domain_contract_generalization")
    manifest = run(repo, out, allow_download=not args.no_download)
    if not manifest.get("all_three_pass", False):
        # Required-dataset incompleteness is a failed publication audit even when
        # diagnostics were written successfully.
        raise SystemExit(2)


if __name__ == "__main__":
    main()
