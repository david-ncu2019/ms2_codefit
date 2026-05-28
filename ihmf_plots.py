"""
ihmf_plots.py — Figures for the two-regime GWL-driven per-layer compaction model.

Public API:
    plot_raw_fit(dates, y_obs_raw, y_pred_raw, dh_lag,
                 x_raw_lag, mask_i, best, station, layer, out_dir)
        -> Path   (saved PNG path)

    plot_reconstruction(dates, y_obs_cum, y_pred_cum, h_raw_lag,
                        insar_raw, best, station, layer, out_dir)
        -> Path   (saved PNG path)

Both functions close their figure after saving to avoid memory leaks when
processing all 195 station-layer pairs in a loop.

Sign conventions shown in plots:
  - MLCW / InSAR: negative = compaction / subsidence  (original sign kept)
  - GWL: higher value = GWL rises (piezometric head in m above MSL, NOT negated)
  - dh_lag: ΔH from reference epoch (m); negative = head fell (drought)
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def plot_raw_fit(
    dates,
    y_obs_raw:  np.ndarray,   # raw cumulative MLCW (mm, negative = compaction)
    y_pred_raw: np.ndarray,   # model prediction (mm, negative = compaction)
    dh_lag:     np.ndarray,   # ΔH from reference, lagged (m; negative = head fell)
    x_raw_lag:  np.ndarray,   # raw cumulative InSAR, lagged (mm, negative = subsidence)
    mask_i:     np.ndarray,   # True where inelastic regime
    best:       dict,
    station:    str,
    layer:      str,
    out_dir:    Path,
) -> Path:
    """Three-panel regression-diagnostic figure (raw signals)."""
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1 — raw MLCW vs model prediction
    ax = axes[0]
    ax.plot(dates, y_obs_raw, "k-", lw=1.5, label="Observed (raw MLCW)")
    ax.plot(dates, y_pred_raw, "r-", lw=1.5, alpha=0.8, label="Predicted")
    if mask_i.sum() > 0:
        ax.plot(dates[mask_i], np.full(mask_i.sum(), y_obs_raw.min() - 0.5),
                "r|", ms=10, alpha=0.5, label=f"inelastic ({mask_i.sum()})")
    ax.set_ylabel("Cumulative compaction\n(mm, negative = compaction)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10, ncol=2)
    ax.set_title(
        f"{station} {layer}  —  "
        f"S_ske={best['S_ske']:.4f}, S_skv={best['S_skv']:.4f} mm/m, "
        f"tau={best['tau']}, R²={best['r2']:.2f}",
        fontsize=12,
    )
    ax.axhline(y=0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.3)

    # Panel 2 — ΔH from reference (negative = head fell = drought)
    ax = axes[1]
    ax.plot(dates, dh_lag, "b-", lw=1.5)
    ax.fill_between(dates, 0, dh_lag,
                    where=(dh_lag < 0), color="red",  alpha=0.10, label="head fell (drought)")
    ax.fill_between(dates, 0, dh_lag,
                    where=(dh_lag > 0), color="blue", alpha=0.08, label="head rose (recharge)")
    ax.set_ylabel("ΔH from reference\n(m, positive = head rose)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.axhline(y=0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.3)

    # Panel 3 — raw InSAR cumulative (negative = subsidence)
    ax = axes[2]
    ax.plot(dates, x_raw_lag, "g-", lw=1.5)
    ax.fill_between(dates, 0, x_raw_lag,
                    where=(x_raw_lag < 0), color="red",  alpha=0.08, label="subsidence")
    ax.fill_between(dates, 0, x_raw_lag,
                    where=(x_raw_lag > 0), color="blue", alpha=0.08, label="uplift")
    ax.set_ylabel("InSAR cumulative\n(mm, negative = subsidence)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.axhline(y=0, color="gray", lw=0.5, ls=":")
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    out_path = out_dir / f"{station}_{layer}_raw_fit.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


def plot_reconstruction(
    dates,
    y_obs_cum:  np.ndarray,   # raw cumulative MLCW (mm, negative = compaction)
    y_pred_cum: np.ndarray,   # model prediction (mm, negative = compaction)
    h_raw_lag:  np.ndarray,   # raw GWL head (m MSL), higher = rising
    insar_raw:  np.ndarray,   # raw InSAR cumulative (mm, negative = subsidence)
    best:       dict,
    station:    str,
    layer:      str,
    out_dir:    Path,
) -> Path:
    """Three-panel reconstruction figure (cumulative, original sign convention)."""
    yr_arr   = (dates.astype("datetime64[Y]").astype(int) + 1970)
    val_mask = yr_arr >= 2022

    fig2, axes2 = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # Panel 1 — cumulative observed vs predicted
    ax = axes2[0]
    ax.plot(dates, y_obs_cum, "k-",  lw=1.5, label="Observed (MLCW reconstr)")
    ax.plot(dates, y_pred_cum, "r--", lw=1.5, alpha=0.85, label="Predicted")
    if val_mask.any():
        ax.axvspan(dates[val_mask][0], dates[-1],
                   color="pink", alpha=0.15, label="validation (2022+)")
    ax.set_ylabel("Cumulative compaction\n(mm, negative = compaction)", fontsize=12)
    ax.legend(loc="lower left", fontsize=10, ncol=3)
    ax.set_title(
        f"{station} {layer} Reconstruction  —  "
        f"S_ske={best['S_ske']:.4f}, S_skv={best['S_skv']:.4f} mm/m, "
        f"tau={best['tau']}, R²={best['r2']:.2f}",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)

    # Panel 2 — raw GWL head (positive = rising)
    h_mean = float(h_raw_lag.mean())
    ax = axes2[1]
    ax.plot(dates, h_raw_lag, "b-", lw=1.5)
    ax.fill_between(dates, h_mean, h_raw_lag,
                    where=(h_raw_lag < h_mean),
                    color="red",  alpha=0.10, label="below mean (drought)")
    ax.fill_between(dates, h_mean, h_raw_lag,
                    where=(h_raw_lag >= h_mean),
                    color="blue", alpha=0.08, label="above mean (wet)")
    ax.axhline(y=h_mean, color="gray", lw=0.6, ls="--")
    ax.set_ylabel("GWL piezometric head\n(m above MSL, + = rising)", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    # Panel 3 — raw InSAR cumulative (negative = subsidence)
    ax = axes2[2]
    ax.plot(dates, insar_raw, "g-", lw=1.5, label="InSAR cumulative")
    ax.set_ylabel("InSAR displacement\n(mm, negative = subsidence)", fontsize=12)
    ax.set_xlabel("Date", fontsize=12)
    ax.legend(loc="upper left", fontsize=10)
    ax.grid(True, alpha=0.3)

    fig2.tight_layout()
    out_path = out_dir / f"{station}_{layer}_reconstruction.png"
    fig2.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig2)
    return out_path
