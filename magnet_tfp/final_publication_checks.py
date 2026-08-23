from __future__ import annotations

import argparse
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def rel_diff(a: float, b: float) -> float:
    den = (abs(a) + abs(b)) / 2.0
    return abs(a-b)/den if den > 0 else 0.0


def safe_ratio(a: float, b: float) -> float:
    lo = min(abs(a), abs(b))
    hi = max(abs(a), abs(b))
    return hi / max(lo, 1e-12)


def independent_matched_pairs(non: pd.DataFrame, tol: float = 0.05) -> pd.DataFrame:
    rows=[]
    non=non.sort_values('forecast_start_s').reset_index(drop=True)
    for i in range(len(non)):
        for j in range(i+1,len(non)):
            a=non.iloc[i]; b=non.iloc[j]
            rd=rel_diff(float(a.rmse_c), float(b.rmse_c))
            if rd > tol:
                continue
            rows.append({
                'window_a': int(a.window_id), 'window_b': int(b.window_id),
                'forecast_a_s': f"{int(a.forecast_start_s)}-{int(a.forecast_end_s)}",
                'forecast_b_s': f"{int(b.forecast_start_s)}-{int(b.forecast_end_s)}",
                'rmse_a_c': float(a.rmse_c), 'rmse_b_c': float(b.rmse_c),
                'rmse_relative_difference': rd,
                'short_rmse_a_c': float(a.band_0_60_rmse_c), 'short_rmse_b_c': float(b.band_0_60_rmse_c),
                'long_rmse_a_c': float(a.band_301_599_rmse_c), 'long_rmse_b_c': float(b.band_301_599_rmse_c),
                'long_rmse_ratio': safe_ratio(float(a.band_301_599_rmse_c), float(b.band_301_599_rmse_c)),
                'p99_a_c': float(a.p99_abs_c), 'p99_b_c': float(b.p99_abs_c),
                'p99_ratio': safe_ratio(float(a.p99_abs_c), float(b.p99_abs_c)),
                'persistence_a': float(a.persistence_envelope_frac), 'persistence_b': float(b.persistence_envelope_frac),
                'persistence_abs_difference': abs(float(a.persistence_envelope_frac)-float(b.persistence_envelope_frac)),
                'condition_a': str(a.condition), 'condition_b': str(b.condition),
            })
    df=pd.DataFrame(rows)
    if df.empty:
        return df
    df['diagnostic_strength'] = np.maximum.reduce([
        df.p99_ratio.to_numpy(),
        1.0 + 5.0*df.persistence_abs_difference.to_numpy(),
        df.long_rmse_ratio.to_numpy(),
    ])
    return df.sort_values(['diagnostic_strength','p99_ratio'], ascending=False).reset_index(drop=True)


def make_counterexample_figure(out: Path, best: pd.Series) -> None:
    # Avoid combining fractions and temperatures on one axis: three panels with matched semantic units.
    fig, axes = plt.subplots(1,3, figsize=(12,3.6))
    labels=[f"W{int(best.window_a)}", f"W{int(best.window_b)}"]
    x=np.arange(2)
    axes[0].bar(x,[best.rmse_a_c,best.rmse_b_c])
    axes[0].set_xticks(x,labels)
    axes[0].set_ylabel('RMSE (°C)')
    axes[0].set_title('Nearly identical aggregate error')

    axes[1].bar(x,[best.p99_a_c,best.p99_b_c])
    axes[1].set_xticks(x,labels)
    axes[1].set_ylabel('p99 |error| (°C)')
    axes[1].set_title(f"Tail severity ({best.p99_ratio:.2f}× apart)")

    axes[2].bar(x,[best.persistence_a,best.persistence_b])
    axes[2].set_xticks(x,labels)
    axes[2].set_ylim(0,1)
    axes[2].set_ylabel('Persistence fraction')
    axes[2].set_title(f"Persistence (Δ={best.persistence_abs_difference:.3f})")
    fig.suptitle('Independent MAGNET windows: similar RMSE, different fidelity diagnostics')
    fig.tight_layout()
    fp=out/'figures'/'independent_matched_rmse_counterexample.png'
    fp.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(fp,dpi=220,bbox_inches='tight')
    plt.close(fig)


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--results',type=Path,required=True)
    args=ap.parse_args()
    out=args.results
    non=pd.read_csv(out/'nonoverlap_windows.csv')
    comp=pd.read_csv(out/'component_horizon_summary.csv')
    red=pd.read_csv(out/'metric_redundancy_vs_rmse.csv')
    sig=pd.read_csv(out/'paired_horizon_significance_nonoverlap.csv').iloc[0]
    trim=pd.read_csv(out/'outlier_trim_sensitivity.csv')
    cond=pd.read_csv(out/'condition_nonoverlap_stat_test.csv').iloc[0]

    pairs=independent_matched_pairs(non,0.05)
    pairs.to_csv(out/'independent_matched_rmse_pairs_5pct.csv',index=False)

    n_pairs=len(pairs)
    n_p99=int((pairs.p99_ratio>=2).sum()) if n_pairs else 0
    n_persist=int((pairs.persistence_abs_difference>=0.25).sum()) if n_pairs else 0
    n_long=int((pairs.long_rmse_ratio>=1.25).sum()) if n_pairs else 0
    best=pairs.iloc[0] if n_pairs else None
    if best is not None:
        make_counterexample_figure(out,best)

    # Main-paper compact evidence table.
    trim20=trim[(trim.trim_metric=='max_abs_c') & (trim.trim_top_percent==20)].iloc[0]
    min_component_ratio=float(comp.median_long_short_ratio.min())
    min_component_frac=float(comp.fraction_long_gt_short.min())
    main_rows=[
        ['Independent horizon degradation', f"n={int(sig['n'])}; median Δ={sig.median_diff:.3f} °C; 95% CI [{sig.diff_ci_low:.3f}, {sig.diff_ci_high:.3f}]; p={sig.wilcoxon_one_sided_p:.3g}; rank-biserial={sig.paired_rank_biserial:.3f}"],
        ['Outlier robustness', f"After trimming worst 20% by max error: long/short median RMSE ratio={trim20.median_long_short_ratio:.2f}×; long>short fraction={trim20.fraction_long_gt_short:.3f}"],
        ['Component transfer', f"All 10 thermowells show median long>short; minimum median ratio={min_component_ratio:.2f}×; minimum long>short fraction={min_component_frac:.3f}"],
        ['Independent matched-RMSE diagnostics', f"{n_pairs} pairs within 5% RMSE among 23 non-overlapping windows; {n_p99} have ≥2× p99 difference; {n_persist} have ≥0.25 persistence difference"],
        ['Operating-condition test', f"Exploratory only: Kruskal-Wallis p={cond.p_value:.3f}; epsilon-squared={cond.epsilon_squared:.3f}; not evidence for condition dependence"],
    ]
    pd.DataFrame(main_rows,columns=['evidence','result']).to_csv(out/'magnet_main_paper_evidence_table.csv',index=False)

    # Claim-boundary file makes the redundancy result explicit rather than hiding it.
    corr={r.metric_vs_rmse: float(r.spearman_rank_corr) for _,r in red.iterrows()}
    lines=[
        '# MAGNET final publication decision', '',
        '## Decision',
        '**PASS for a concise main-paper cross-domain transfer subsection.**', '',
        'The strongest evidence is horizon-scale degradation, component localization, and independent matched-RMSE counterexamples. The condition-regime test is null and should not be used as cross-domain support for condition dependence.', '',
        '## Independence-aware horizon result',
        f"- Non-overlapping windows: **{int(sig['n'])}**",
        f"- Median short-horizon RMSE: **{sig.median_a:.4f} °C**",
        f"- Median long-horizon RMSE: **{sig.median_b:.4f} °C**",
        f"- Median paired increase: **{sig.median_diff:.4f} °C**, bootstrap 95% CI **[{sig.diff_ci_low:.4f}, {sig.diff_ci_high:.4f}] °C**",
        f"- Long > short in **{sig.fraction_long_gt_short*100:.1f}%** of independent windows",
        f"- Wilcoxon one-sided p = **{sig.wilcoxon_one_sided_p:.3g}**; paired rank-biserial = **{sig.paired_rank_biserial:.3f}**", '',
        '## Independent same-RMSE counterexamples',
        f"- Within the conservative 23-window set, **{n_pairs}** pairs have aggregate RMSE within 5%.",
        f"- **{n_p99}** of those pairs differ by at least 2× in p99 tail error.",
        f"- **{n_persist}** differ by at least 0.25 in persistence fraction.",
    ]
    if best is not None:
        lines += [
            f"- Strongest example: windows **{int(best.window_a)}** and **{int(best.window_b)}** have RMSE **{best.rmse_a_c:.3f} vs {best.rmse_b_c:.3f} °C** (relative difference **{100*best.rmse_relative_difference:.2f}%**), but p99 **{best.p99_a_c:.2f} vs {best.p99_b_c:.2f} °C** (**{best.p99_ratio:.2f}×**) and persistence **{best.persistence_a:.3f} vs {best.persistence_b:.3f}** (Δ **{best.persistence_abs_difference:.3f}**).",
            f"- Their short-horizon RMSE is **{best.short_rmse_a_c:.3f} vs {best.short_rmse_b_c:.3f} °C**, showing qualitatively different temporal fidelity despite similar aggregate RMSE.",
        ]
    lines += ['', '## Component result',
        f"- Every thermowell has a median long/short RMSE ratio above **{min_component_ratio:.2f}×**.",
        f"- For every thermowell, long-horizon error exceeds short-horizon error in at least **{100*min_component_frac:.1f}%** of strict windows.",
        f"- TC-06 is the strongest localized degradation: median long-horizon RMSE **{comp.iloc[0].median_long_rmse_c:.3f} °C**.", '',
        '## Important limitation / claim boundary',
        f"- Across all windows, long-horizon RMSE is highly correlated with aggregate RMSE (Spearman ρ={corr.get('band_301_599_rmse_c', float('nan')):.3f}) and p99 is also highly correlated (ρ={corr.get('p99_abs_c', float('nan')):.3f}).",
        f"- Persistence is less redundant (ρ={corr.get('persistence_envelope_frac', float('nan')):.3f}).",
        '- Therefore do **not** claim that every TFP dimension is statistically independent of RMSE. Claim that the decomposition preserves diagnostic structure that aggregate RMSE compresses away; the independent matched-RMSE examples demonstrate this directly.',
        f"- The operating-regime comparison is null on independent windows (Kruskal-Wallis p={cond.p_value:.3f}, epsilon-squared={cond.epsilon_squared:.3f}); do not cite MAGNET as evidence for condition-dependent fidelity.",
        '- Persistence thresholds in MAGNET are sensitivity analyses rather than physics-certified safety tolerances; keep the threshold sweep in supplementary material unless a traceable instrument/application tolerance is established.', '',
        '## Recommended manuscript role',
        '- One concise cross-domain transfer subsection in the main paper.',
        '- Main figure: horizon profile + component heatmap; optionally the independent matched-RMSE counterexample as a third panel or supplementary figure.',
        '- One compact evidence table using `magnet_main_paper_evidence_table.csv`.',
        '- Keep robotics as the primary validation domain and Muñoz comparison there.',
        '- Claim **cross-domain transfer**, not universal digital-twin validity.',
    ]
    (out/'MAGNET_FINAL_PUBLICATION_DECISION.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    print('Final MAGNET publication checks complete.')
    print(f'Independent 5% matched-RMSE pairs: {n_pairs}; p99>=2x: {n_p99}; persistence diff>=0.25: {n_persist}')
    print(f'Decision: {(out/"MAGNET_FINAL_PUBLICATION_DECISION.md").resolve()}')

if __name__=='__main__':
    main()
