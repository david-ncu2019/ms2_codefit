"""
Two-regime GWL-driven per-layer compaction model: fit for any station+layer.

Reads station/layer file paths from data/ihmf_config.json — a centralized
registry generated from the assignment, well-info, and 2S-TOOL CSVs.

Usage:
    python scripts/10_ihmf/fit_ihm_f.py --station TUKU --layer F1
    python scripts/10_ihmf/fit_ihm_f.py --station TUKU --all

Modules:
    ihmf_io.py    — data loading and alignment
    ihmf_model.py — signal preparation, OLS fitting, walk-forward validation
    ihmf_plots.py — raw-fit and reconstruction figures
"""

import argparse
import json
import numpy as np
from pathlib import Path

from ihmf_io    import load_and_align
from ihmf_model import (prepare_signals, grid_search_tau,
                        run_walk_forward, build_diagnostics)
from ihmf_plots import plot_raw_fit, plot_reconstruction

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CONFIG_PATH  = PROJECT_ROOT / "data/ihmf_config.json"

# ── Parse args ──────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(
    description="Fit two-regime GWL-driven compaction model for a station+layer pair.")
parser.add_argument("--station", type=str, required=True, help="Station name (e.g. TUKU)")
parser.add_argument("--layer",   type=str, default=None,
                    help="Layer code (F1/T1/F2/T2/F3/F4). Omit with --all.")
parser.add_argument("--all", action="store_true",
                    help="Process all layers for the given station.")
args = parser.parse_args()

# ── Load config ─────────────────────────────────────────────────────────────
with open(CONFIG_PATH) as f:
    cfg = json.load(f)

shared  = cfg["shared"]
entries = cfg["entries"]
INSAR_CSV = PROJECT_ROOT / shared["insar_csv"]

station_entries = [e for e in entries if e["station"] == args.station]
if not station_entries:
    raise SystemExit(f"Station '{args.station}' not found in config. "
                     f"Available: {sorted(set(e['station'] for e in entries))}")

if args.all:
    targets = station_entries
elif args.layer:
    match = [e for e in station_entries if e["layer"] == args.layer]
    if not match:
        raise SystemExit(f"Layer '{args.layer}' not found for {args.station}. "
                         f"Available: {[e['layer'] for e in station_entries]}")
    targets = [match[0]]
else:
    raise SystemExit("Specify --layer LAYER or --all.")


# ── Per-entry fitting ────────────────────────────────────────────────────────

def run_one(entry: dict) -> dict:
    """Fit the model for a single config entry. Returns the result dict."""
    station  = entry["station"]
    layer    = entry["layer"]
    tau_max  = entry["tau_max"]
    hc_pct   = entry["hc_percentile"]
    warm_skv = entry["warmstart_skv"]
    warm_ske = entry["warmstart_ske"]
    out_dir  = PROJECT_ROOT / entry["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load & align
    merged, meta = load_and_align(entry, PROJECT_ROOT, INSAR_CSV)
    n_epochs = meta["n_epochs"]

    # 2. Prepare raw signals and regime classification
    (t_days, y_raw, x_raw, dh_raw, h_raw, is_elastic,
     h_c_eff, corr_yh, corr_yx, corr_hx) = prepare_signals(merged, hc_pct)

    n_elastic   = int(is_elastic.sum())
    n_inelastic = int((~is_elastic).sum())

    # 3. τ grid search → best OLS fit
    best, tau_results = grid_search_tau(y_raw, dh_raw, x_raw, is_elastic, tau_max)

    # 4. Walk-forward validation
    wf_results = run_walk_forward(merged, y_raw, dh_raw, x_raw, is_elastic, tau_max)

    # 5. Diagnostics
    concerns = build_diagnostics(corr_yh, best, n_inelastic)

    # 6. Console summary
    print(f"\n{'─' * 60}")
    print(f"  {station} {layer}  |  {n_epochs} epochs  "
          f"|  h_c={meta['h_c_depth_m']:.1f}m depth  "
          f"|  well={meta['wellcode']}  elev={meta['well_elev_m']:.0f}m")
    print(f"  GWL: {meta['gwl_feather_name']}")
    print(f"{'─' * 60}")
    print(f"  corr(y,dh)={corr_yh:+.3f}  corr(y,x)={corr_yx:+.3f}  "
          f"corr(dh,x)={corr_hx:+.3f}")
    print(f"  sigma_y={y_raw.std():.2f} mm  sigma_dh={dh_raw.std():.2f} m")
    print(f"  elastic={n_elastic}  inelastic={n_inelastic}  "
          f"({100*n_inelastic/n_epochs:.0f}%)")
    print(f"  tau={best['tau']}  R2={best['r2']:.3f}  RMSE={best['rmse']:.3f} mm")
    print(f"  intercept={best['intercept']:.4e}  "
          f"S_ske={best['S_ske']:.4e}  S_skv={best['S_skv']:.4e}  "
          f"beta={best['beta']:.4e}")
    print(f"  2S-TOOL: S_kv={warm_skv:.4e}  S_ke={warm_ske:.4e}")
    for c in concerns:
        print(f"  WARNING: {c}")
    if wf_results:
        wf_rmse = " | ".join(f"{w['rmse_mm']:.2f}" for w in wf_results)
        print(f"  Walk-forward RMSE: {wf_rmse} mm")

    # 7. Figures
    tau    = best["tau"]
    n      = len(y_raw)
    mask_e = is_elastic[:n - tau]
    mask_i = ~mask_e
    X_full = np.column_stack([
        np.ones(n - tau),
        dh_raw[tau:] * mask_e,
        dh_raw[tau:] * mask_i,
        x_raw[:n - tau],
    ])
    y_pred = X_full @ [best["intercept"], best["S_ske"], best["S_skv"], best["beta"]]
    dates_pred = merged["datetime"].iloc[:n - tau].values

    fig_path = plot_raw_fit(
        dates=dates_pred,
        y_obs_raw=y_raw[:n - tau],
        y_pred_raw=y_pred,
        dh_lag=dh_raw[tau:],
        x_raw_lag=x_raw[:n - tau],
        mask_i=mask_i,
        best=best,
        station=station,
        layer=layer,
        out_dir=out_dir,
    )

    fig2_path = plot_reconstruction(
        dates=dates_pred,
        y_obs_cum=merged["mlcw_mm"].values[:n - tau],
        y_pred_cum=y_pred,
        h_raw_lag=h_raw[tau:],
        insar_raw=merged["insar_mm"].values[:n - tau],
        best=best,
        station=station,
        layer=layer,
        out_dir=out_dir,
    )

    # 8. Save JSON result
    result = {
        "station": station, "layer": layer,
        "wellcode": meta["wellcode"],
        "well_elev_m": meta["well_elev_m"],
        "gwl_feather": meta["gwl_feather_name"],
        "h_c_depth_m": round(meta["h_c_depth_m"], 2),
        "h_c_head_m":  round(meta["h_c_head_m"],  2),
        "h_c_eff_m_msl": round(h_c_eff, 2),
        "n_epochs":    n_epochs,
        "n_elastic":   n_elastic,
        "n_inelastic": n_inelastic,
        "corr_y_dh":   round(corr_yh, 4),
        "corr_y_x":    round(corr_yx, 4),
        "corr_dh_x":   round(corr_hx, 4),
        "tau_opt":     best["tau"],
        "intercept_mm": best["intercept"],
        "S_ske_bulk_mm_per_m": best["S_ske"],
        "S_skv_bulk_mm_per_m": best["S_skv"],
        "beta":        best["beta"],
        "r2":          best["r2"],
        "rmse_mm":     best["rmse"],
        "warmstart_skv_2stool": warm_skv,
        "warmstart_ske_2stool": warm_ske,
        "walk_forward": wf_results,
        "tau_rss_curve": [
            {"tau": r["tau"], "rss": r["rss"], "r2": r["r2"]}
            for r in sorted(tau_results, key=lambda x: x["tau"])
        ],
        "diagnostics": concerns,
    }
    json_path = out_dir / f"{station}_{layer}_ihmf_results.json"
    with open(json_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    print(f"  -> {fig_path.name}  |  {fig2_path.name}  |  {json_path.name}")
    return result


# ══════════════════════════════════════════════════════════════════════════════
# Entry point
# ══════════════════════════════════════════════════════════════════════════════

print(f"Config: {CONFIG_PATH}  ({len(entries)} entries)")
print(f"Targets: {len(targets)} layer(s) for {args.station}")

for entry in targets:
    try:
        run_one(entry)
    except Exception as exc:
        print(f"\n  {entry['station']} {entry['layer']}: ERROR — {exc}")

print("\nDone.")
