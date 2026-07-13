"""make_figures.py — regenerate every paper figure from the frozen artifacts.

Every figure is authored at its TRUE on-page width so that matplotlib's
absolute point sizes render at their nominal value in the compiled PDF.
The ACL column is 3.03 in wide and the full text block 6.30 in; drawing
on a wider canvas and letting \\includegraphics shrink the figure scales
every label down (a 6.5-10.5 in canvas lands at ~3.5-5.5 pt).
Column figures are drawn at COL_W and placed with
width=\\columnwidth; the two full-width figures are drawn at their
0.78/0.88 x textwidth targets. Nothing is scaled by more than the small
tight-bbox crop, so on-page fonts land at ~7.5-9.5 pt next to the ~10 pt
caption text.

Figures (all read real artifacts; output filenames keep their v4_* names):
  v4_fig0_teaser                    → paper Fig. 1 (page-1 overview)
  v4_fig1_six_condition_lollipop    → Sec. 4.1 (companion to Table 1)
  v4_fig2_polysemy_waterfall        → Sec. 4.2 (companion to Table 3)
  v4_fig6_layer_trajectory_gradient → paper Fig. 2 (Sec. 4.5, App. E)
  v4_fig7_l0_sweep                  → paper Fig. 3 (Sec. 4.5, App. E)
  v4_fig5_shell_trajectory          → paper Fig. 5 (Sec. 7 Limitations)
  v4_appC_threshold_streamgraph     → paper Fig. 6 (App. D)
  v4_fig_confusion_genflip          → paper Fig. 4 (Sec. 4.4)

Run: python make_figures.py
"""
import json
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from scipy.ndimage import gaussian_filter1d


# ── On-page geometry (inches) ────────────────────────────────────
# Measured from the ACL class: \columnwidth = 219.086 pt, \textwidth =
# 455.244 pt (1 pt = 1/72.27 in).
COL_W  = 219.08614 / 72.27   # 3.031 in  — single column
TEXT_W = 455.24411 / 72.27   # 6.299 in  — full two-column block
FIG1_W = 0.78 * TEXT_W       # 4.913 in  — figure* at 0.78\textwidth
CONF_W = 0.88 * TEXT_W       # 5.543 in  — figure* at 0.88\textwidth


# ── Shared style (column-native point sizes) ─────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.serif":       ["Times", "Times New Roman", "Computer Modern Roman",
                         "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size":        8.5,
    "axes.titlesize":   9.5,
    "axes.titleweight": "bold",
    "axes.labelsize":   8.5,
    "axes.labelcolor":  "#1A202C",
    "axes.edgecolor":   "#2D3748",
    "axes.linewidth":   0.8,
    "xtick.color":      "#4A5568",
    "ytick.color":      "#4A5568",
    "xtick.labelsize":  7.5,
    "ytick.labelsize":  7.5,
    "legend.fontsize":  7.5,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "savefig.dpi":   300,
    "savefig.bbox":  "tight",
    "savefig.pad_inches": 0.02,
})

INK    = "#1A365D"   # primary blue
RED    = "#C53030"   # signal red (large negative deltas)
GOLD   = "#B7791F"   # warm gold (small negative deltas)
TEAL   = "#2C7A7B"   # secondary teal
SLATE  = "#4A5568"   # mid grey (refs, baselines, null)
MUTED  = "#A0AEC0"   # soft grey
CLOUD  = "#EDF2F7"   # near-white grey (shaded regions)

A   = (Path(__file__).resolve().parent.parent / "artifacts")
FIG = Path(__file__).resolve().parent / "rendered"
FIG.mkdir(parents=True, exist_ok=True)


def save_both(fig, name):
    fig.savefig(FIG / f"{name}.pdf")
    fig.savefig(FIG / f"{name}.png", dpi=200)
    plt.close(fig)
    print(f"  wrote {name}.{{pdf,png}}")


# ── Shared headline loader (keeps fig1 and the teaser in lock-step) ──
def headline_deltas():
    """Per-pair mean hit@1 for the six conditions at L41.
    Returns (baseline, [(condition, hit@1, delta_pp), ...] most-impactful
    first)."""
    p = json.load(open(A / "layer_bookends" / "L41" / "results_main.json"))

    def per_pair_mean(rows, key, val):
        d = defaultdict(list)
        for r in rows:
            d[(r["pair_id"], r[key])].append(r[val])
        return {k: float(np.mean(v)) for k, v in d.items()}

    sib = per_pair_mean(p["cross_rows"],            "target_idx", "ablate_hit1")
    wt  = per_pair_mean(p["wikitext_shuffled_rows"], "target_idx", "wt_shuffled_hit1")
    ash = per_pair_mean(p["ambigqa_shuffled_rows"],  "target_idx", "shuffled_hit1")
    rnd = per_pair_mean(p["random_rows"],            "ablate_idx", "random_hit1")

    keys = sorted(sib.keys())
    base_l = {(r["pair_id"], r["target_idx"]): r["base_hit1"]   for r in p["self_rows"]}
    targ_l = {(r["pair_id"], r["target_idx"]): r["ablate_hit1"] for r in p["self_rows"]}
    _m = lambda d: float(np.mean([d[k] for k in keys if k in d]))
    base = _m(base_l)
    order = [("Targeted", targ_l), ("Sibling", sib),
             ("Shuffled (AmbigQA)", ash), ("WikiText shuffled", wt),
             ("Random", rnd)]
    return base, [(nm, _m(d), (_m(d) - base) * 100) for nm, d in order]


# ═════════════════════════════════════════════════════════════════
# PAPER FIG 1 — Page-1 overview (teaser)
# ═════════════════════════════════════════════════════════════════
def fig0_teaser():
    """One-glance summary: an ambiguous question yields sibling answers,
    and features selected along a specificity gradient (answer-specific →
    corpus-generic) carry a monotonically fading causal effect. Numbers
    are the real L41 headline deltas."""
    base, rows = headline_deltas()
    conds = [("Targeted",      "features for the\ncommitted answer"),
             ("Sibling",       "features for a\nsibling answer"),
             ("Shuf. AmbigQA", "features for an\nunrelated question"),
             ("WikiText",      "features from\nrandom prose"),
             ("Random",        "random\nfeatures")]
    deltas = [r[2] for r in rows]                    # -13.24 .. 0.00

    fig, ax = plt.subplots(figsize=(TEXT_W, 2.4))
    ax.set_xlim(0, 100); ax.set_ylim(0, 100); ax.axis("off")

    # ── Left: the substrate (AmbigQA sibling structure) ──────────
    ax.text(14.5, 97, "AmbigQA: one question,\nmultiple valid answers",
            ha="center", va="top", fontsize=7.8, color=SLATE,
            fontweight="bold")
    qbox = FancyBboxPatch((2.0, 68), 25, 16,
                          boxstyle="round,pad=0.5,rounding_size=2.5",
                          linewidth=1.0, edgecolor=INK, facecolor=CLOUD)
    ax.add_patch(qbox)
    ax.text(14.5, 76, "“When did the Simpsons\nfirst air on television?”",
            ha="center", va="center", fontsize=8.0, color="#1A202C",
            style="italic")
    ax.add_patch(FancyArrowPatch((14.5, 67), (14.5, 58.5),
                 arrowstyle="-|>", mutation_scale=9, linewidth=0.9,
                 color=SLATE))
    for (yy, txt, col) in [(48, "1987 · Tracey Ullman short", INK),
                           (35, "1989 · prime-time premiere", TEAL)]:
        sb = FancyBboxPatch((2.0, yy), 25, 9,
                            boxstyle="round,pad=0.4,rounding_size=2",
                            linewidth=0.9, edgecolor=col, facecolor="white")
        ax.add_patch(sb)
        ax.text(14.5, yy + 4.5, txt, ha="center", va="center",
                fontsize=7.5, color=col)
    ax.text(14.5, 30, "siblings", ha="center", va="top",
            fontsize=7.8, color=SLATE, style="italic")

    # divider
    ax.plot([32, 32], [8, 88], color=MUTED, linewidth=0.7, linestyle=(0, (3, 3)))

    # ── Right: the specificity gradient ──────────────────────────
    x0, x1 = 41, 96
    xs = np.linspace(x0, x1, len(conds))
    base_y = 40
    mmax = max(abs(d) for d in deltas) or 1.0

    ax.text(68.5, 97, "Targeted ablation vs. a graded control ladder",
            ha="center", va="top", fontsize=8.8, color=INK,
            fontweight="bold")
    ax.text(68.5, 90, r"bar = causal effect on the answer  ($|\Delta$ hit@1$|$, pp)",
            ha="center", va="top", fontsize=7.5, color=SLATE)

    ax.plot([x0 - 2.5, x1 + 2.5], [base_y, base_y],
            color=SLATE, linewidth=0.8, alpha=0.6, zorder=1)
    for x, (name, how), d in zip(xs, conds, deltas):
        col = RED if d < -2 else (GOLD if d < -0.5 else MUTED)
        h = max(36 * abs(d) / mmax, 0.8)             # bar height ∝ |Δ|
        ax.add_patch(plt.Rectangle((x - 3.1, base_y), 6.2, h,
                     facecolor=col, edgecolor="white", linewidth=0.8,
                     alpha=0.92, zorder=3))
        ax.plot(x, base_y, "o", markersize=5.5, color=col,
                markeredgecolor="white", markeredgewidth=1.1, zorder=4)
        ax.text(x, base_y + h + 1.6, f"{d:+.1f}", ha="center", va="bottom",
                fontsize=7.8, color=col, fontweight="bold")
        ax.text(x, base_y - 2.0, name, ha="center", va="top",
                fontsize=7.6, color="#1A202C", fontweight="bold")
        ax.text(x, base_y - 8.5, how, ha="center", va="top",
                fontsize=6.8, color=SLATE, linespacing=1.15)

    # selection axis at the bottom
    ax.add_patch(FancyArrowPatch((x0 - 2, 15), (x1 + 2, 15),
                 arrowstyle="-|>", mutation_scale=9, linewidth=1.0,
                 color=SLATE))
    ax.text(x0 - 2, 10.5, "answer-specific", ha="left", va="top",
            fontsize=7.2, color=SLATE, style="italic")
    ax.text(x1 + 2, 10.5, "corpus-generic", ha="right", va="top",
            fontsize=7.2, color=SLATE, style="italic")
    ax.text(68.5, 10.5, "how features are selected", ha="center", va="top",
            fontsize=7.2, color=SLATE)
    save_both(fig, "v4_fig0_teaser")


# ═════════════════════════════════════════════════════════════════
# Six-condition control gradient (Sec. 4.1)  →  Cleveland lollipop
# ═════════════════════════════════════════════════════════════════
def fig1_six_condition_lollipop():
    """Per-pair mean hit@1 for each condition vs Baseline reference.
    Stem length = Δ. Companion to Table 1."""
    base, rows = headline_deltas()
    conditions = [r[0] for r in rows]
    hits       = [r[1] for r in rows]
    deltas     = [r[2] for r in rows]

    fig, ax = plt.subplots(figsize=(FIG1_W, 2.3))
    y = np.arange(len(conditions))[::-1]

    ax.axvline(base, color=SLATE, linestyle="--", linewidth=1.0,
               alpha=0.7, zorder=1)

    for yi, h, d in zip(y, hits, deltas):
        col = RED if d < -2 else (GOLD if d < -0.5 else SLATE)
        ax.plot([base, h], [yi, yi], color=col, alpha=0.5,
                linewidth=2.0, solid_capstyle="round", zorder=2)
        ax.plot(h, yi, "o", markersize=9, color=col,
                markeredgecolor="white", markeredgewidth=1.4, zorder=3)
        sign = "+" if d > 0.005 else ""
        label = f"{sign}{d:.2f} pp"
        if h < base:
            ax.text(h - 0.006, yi, label, ha="right", va="center",
                    fontsize=8.0, color=col, fontweight="medium")
        else:
            ax.text(h + 0.006, yi, label, ha="left", va="center",
                    fontsize=8.0, color=col, fontweight="medium")

    ax.text(base + 0.004, -0.72, f"baseline = {base:.3f}",
            ha="left", va="center", color=SLATE,
            fontsize=8.0, style="italic")

    ax.set_yticks(y); ax.set_yticklabels(conditions)
    ax.set_xlabel("hit@1")
    pad = 0.014
    ax.set_xlim(min(hits + [base]) - 0.050, max(hits + [base]) + pad)
    ax.set_ylim(-1.15, len(conditions) - 0.35)
    ax.set_title(r"Six-condition control gradient at L41 ($n = 1{,}103$ self-pairs)",
                 loc="left", pad=10)
    ax.grid(axis="x", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    save_both(fig, "v4_fig1_six_condition_lollipop")


# ═════════════════════════════════════════════════════════════════
# Content-vs-position decomposition (Sec. 4.2)  →  Waterfall
# ═════════════════════════════════════════════════════════════════
def fig2_polysemy_waterfall():
    """Content + Position + Interaction = Targeted, shown as a waterfall."""
    poly = json.load(open(A / "reference_layer" / "L41_summary_table.json"))["polysemy_at_080"]
    targ = poly["delta_targeted_pp"]
    cont = poly["delta_content_pp"]
    pos  = poly["delta_position_pp"]
    inter = poly["residual_pp"]
    pcts = [poly["pct_targeted_content"],
            poly["pct_targeted_position"],
            poly["pct_targeted_residual"]]

    components = ["Content", "Position", "Interaction"]
    deltas     = [cont, pos, inter]
    cols       = [INK, GOLD, TEAL]

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    bar_w = 0.62
    cumul = 0.0
    ends  = []

    for i, (comp, d, pct, col) in enumerate(zip(components, deltas, pcts, cols)):
        start = cumul
        end   = cumul + d
        ax.bar(i, d, bottom=start, color=col, width=bar_w,
               edgecolor="white", linewidth=0.8, zorder=3, alpha=0.92)
        midy = (start + end) / 2
        ax.text(i, midy, f"{d:+.2f}", ha="center", va="center",
                color="white", fontweight="bold", fontsize=8.0)
        ax.text(i, end - 0.45, f"{pct:.0f}%", ha="center", va="top",
                fontsize=7.0, color=SLATE)
        ends.append(end)
        cumul = end

    for i in range(len(components)):
        ax.plot([i + bar_w / 2, (i + 1) - bar_w / 2],
                [ends[i], ends[i]],
                color=SLATE, linestyle=":", linewidth=0.9, alpha=0.7, zorder=2)

    ax.bar(len(components), targ, color=RED, width=bar_w,
           edgecolor="white", linewidth=0.8, zorder=3, alpha=0.92)
    ax.text(len(components), targ / 2, f"{targ:+.2f}",
            ha="center", va="center", color="white",
            fontweight="bold", fontsize=8.0)

    ax.axhline(0, color=SLATE, linewidth=0.8, alpha=0.7, zorder=1)
    ax.set_xticks(range(len(components) + 1))
    ax.set_xticklabels(components + ["Targeted"], fontsize=7.6)
    ax.set_ylabel(r"$\Delta$ pp vs. baseline")
    ax.set_ylim(targ - 2.2, 1.6)
    ax.set_title(r"Content-vs-position decomposition at L41 ($n = 1{,}103$)",
                 loc="left", pad=10)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    save_both(fig, "v4_fig2_polysemy_waterfall")


# ═════════════════════════════════════════════════════════════════
# PAPER FIG 2 — Layer trajectory  →  Gradient line + CI ribbon
# ═════════════════════════════════════════════════════════════════
def fig6_layer_trajectory_gradient():
    """7 measured layers, depth-encoded gradient line, CI ribbon."""
    traj = json.load(open(A / "layer_bookends" / "trajectory_summary.json"))
    s = traj["summaries"]
    layers = [x["layer"] for x in s]
    t3_d   = [x["tests"]["T3_sibling_vs_wtshuf"]["delta_pp"] for x in s]
    t3_lo  = [x["tests"]["T3_sibling_vs_wtshuf"]["ci_pp"][0] for x in s]
    t3_hi  = [x["tests"]["T3_sibling_vs_wtshuf"]["ci_pp"][1] for x in s]

    L_dense = np.linspace(min(layers), max(layers), 200)
    d_dense  = np.interp(L_dense, layers, t3_d)
    lo_dense = np.interp(L_dense, layers, t3_lo)
    hi_dense = np.interp(L_dense, layers, t3_hi)

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))

    ax.fill_between(L_dense, lo_dense, hi_dense,
                    color=INK, alpha=0.13, linewidth=0, zorder=2,
                    label="95% bootstrap CI")

    cmap = LinearSegmentedColormap.from_list("depth", ["#C8D6EE", "#5A7AB6", INK])
    norm = Normalize(min(layers), max(layers))
    pts  = np.array([L_dense, d_dense]).T.reshape(-1, 1, 2)
    segs = np.concatenate([pts[:-1], pts[1:]], axis=1)
    lc = LineCollection(segs, cmap=cmap, norm=norm, linewidth=2.4,
                        capstyle="round", zorder=3)
    lc.set_array(L_dense[:-1])
    ax.add_collection(lc)

    for L, d in zip(layers, t3_d):
        ax.plot(L, d, "o", markersize=6.5, color=cmap(norm(L)),
                markeredgecolor="white", markeredgewidth=1.3, zorder=4)
        ax.text(L, d + 0.7, f"{d:+.1f}", ha="center", va="bottom",
                fontsize=6.8, color=INK)

    ax.axhline(0, color=SLATE, linewidth=0.7, alpha=0.6, zorder=1)
    ax.set_xlabel("residual-stream layer")
    ax.set_ylabel(r"T3 $\Delta$ pp (Sib. $-$ WikiText-shuf.)")
    ax.set_title("T3 trajectory across seven layers",
                 loc="left", pad=8)
    ax.set_xticks(layers)
    ax.set_xticklabels([f"L{l}" for l in layers], rotation=45, ha="right",
                       fontsize=7.0)
    ax.set_xlim(min(layers) - 1, max(layers) + 2.0)
    ax.set_ylim(min(t3_lo) - 1.0, max(t3_hi) + 1.8)
    ax.grid(axis="y", linestyle=":", linewidth=0.6, alpha=0.5)
    ax.legend(loc="upper left", frameon=False)
    save_both(fig, "v4_fig6_layer_trajectory_gradient")


# ═════════════════════════════════════════════════════════════════
# PAPER FIG 3 — L0 sweep at L37 vs L41
# ═════════════════════════════════════════════════════════════════
def fig7_l0_sweep():
    """T3 vs SAE sparsity (average L_0) across Gemma Scope's published
    family, at L37 and L41."""
    sweep = json.load(open(A / "l0_sweep" / "l0_sweep_summary.json"))
    canonical = sweep["canonical_l0_per_layer"]
    canon_t3 = {37: 4.49, 41: 9.05}     # canonical-L0 T3 (main run)
    colors = {"37": TEAL, "41": RED}

    fig, ax = plt.subplots(figsize=(COL_W, 2.15))
    for L_str, cells in sweep["summaries"].items():
        L = int(L_str)
        if not cells:
            continue
        cells_sorted = sorted(cells, key=lambda c: c["l0"])
        l0_x = [c["l0"] for c in cells_sorted]
        t3_y = [c["T3"]["delta_pp"] for c in cells_sorted]
        t3_lo = [c["T3"]["ci_pp"][0] for c in cells_sorted]
        t3_hi = [c["T3"]["ci_pp"][1] for c in cells_sorted]
        yerr = [[d - lo for d, lo in zip(t3_y, t3_lo)],
                [hi - d for hi, d in zip(t3_hi, t3_y)]]
        canon = canonical[L_str]
        all_l0 = [canon] + l0_x
        all_t3 = [canon_t3[L]] + t3_y
        order = sorted(range(len(all_l0)), key=lambda k: all_l0[k])
        sorted_l0 = [all_l0[k] for k in order]
        sorted_t3 = [all_t3[k] for k in order]
        ax.plot(sorted_l0, sorted_t3, "-", color=colors[L_str], linewidth=1.4,
                label=fr"L{L} ($\star$ canonical $L_0={canon}$)")
        ax.errorbar(l0_x, t3_y, yerr=yerr, fmt="o", capsize=3,
                    color=colors[L_str], ecolor="#999", elinewidth=0.7,
                    markersize=5, markeredgecolor="white",
                    markeredgewidth=0.6, linewidth=0)
        ax.scatter([canon], [canon_t3[L]], marker="*", s=150,
                   color=colors[L_str], edgecolor="white", linewidth=1.0,
                   zorder=5)

    ax.axhline(0, color=SLATE, linewidth=0.7, alpha=0.7)
    ax.set_xlabel(r"SAE average $L_0$ (lower = sparser)")
    ax.set_ylabel(r"T3 $\Delta$ pp (Sib. $-$ WikiText-shuf.)")
    ax.set_title(r"T3 vs SAE $L_0$ (Gemma Scope family)", loc="left", pad=8)
    ax.set_xscale("log")
    ax.legend(loc="best", frameon=False)
    ax.grid(True, which="both", alpha=0.22, linewidth=0.5)
    save_both(fig, "v4_fig7_l0_sweep")


# ═════════════════════════════════════════════════════════════════
# PAPER FIG 5 — Overlap-shell T3 trajectory (Sec. 7 Limitations), column-native
# ═════════════════════════════════════════════════════════════════
def fig5_shell_trajectory():
    """Per-shell T3 (max overlap with any sibling) at L37, with 95% CIs."""
    shells = json.load(open(A / "overlap_shells" / "d" / "per_shell_results.json"))["shells"]
    keep = [s for s in shells if s["T3"]["delta_pp"] is not None]
    ks  = [s["k"] for s in keep]
    ds  = [s["T3"]["delta_pp"] for s in keep]
    los = [s["T3"]["ci_pp"][0] for s in keep]
    his = [s["T3"]["ci_pp"][1] for s in keep]
    ns  = [s["n"] for s in keep]
    yerr = [[d - lo for d, lo in zip(ds, los)],
            [hi - d for hi, d in zip(his, ds)]]

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.errorbar(ks, ds, yerr=yerr, fmt="o", capsize=3, color=INK,
                ecolor="#999", elinewidth=0.8, markersize=5,
                markerfacecolor=INK, markeredgecolor="white",
                markeredgewidth=0.7, zorder=3)
    for k, d, n in zip(ks, ds, ns):
        ax.annotate(f"{n}", (k, d), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=6.3, color=SLATE)
    ax.axhline(0, color=SLATE, linewidth=0.7, alpha=0.7)
    ax.axhline(4.49, color=RED, linestyle="--", linewidth=0.9,
               label=r"full-sample T3 $= +4.49$ pp")
    ax.set_xlabel("max overlap with any sibling (top-10)")
    ax.set_ylabel(r"T3 $\Delta$ pp (signed)")
    ax.set_title("Per-shell T3 with 95% CIs (L37)", loc="left", pad=8)
    ax.set_xticks(range(11))
    ax.legend(loc="upper left", frameon=False)
    ax.grid(axis="y", alpha=0.22, linewidth=0.5)
    save_both(fig, "v4_fig5_shell_trajectory")


# ═════════════════════════════════════════════════════════════════
# PAPER FIG 6 (App. D) — Polysemy threshold sweep  →  Streamgraph
# ═════════════════════════════════════════════════════════════════
def appC_threshold_streamgraph():
    """Smooth stacked areas: content/position/interaction shares across
    threshold."""
    ts = json.load(open(A / "reference_layer" / "polysemy" / "threshold_sweep_L41.json"))
    cells = ts["cells"]
    thresholds = np.array([c["threshold"] for c in cells])
    cont_pct = np.array([c["pct_targeted_content"]  for c in cells])
    pos_pct  = np.array([c["pct_targeted_position"] for c in cells])
    res_pct  = np.array([c["pct_targeted_residual"] for c in cells])
    tot = cont_pct + pos_pct + res_pct
    cont_pct, pos_pct, res_pct = cont_pct / tot * 100, pos_pct / tot * 100, res_pct / tot * 100

    t_dense = np.linspace(thresholds[0], thresholds[-1], 200)
    cont_d = gaussian_filter1d(np.interp(t_dense, thresholds, cont_pct), sigma=2.5)
    pos_d  = gaussian_filter1d(np.interp(t_dense, thresholds, pos_pct),  sigma=2.5)
    res_d  = gaussian_filter1d(np.interp(t_dense, thresholds, res_pct),  sigma=2.5)

    fig, ax = plt.subplots(figsize=(COL_W, 2.05))
    ax.fill_between(t_dense, 0, cont_d, color=INK,  alpha=0.88, linewidth=0)
    ax.fill_between(t_dense, cont_d, cont_d + pos_d, color=GOLD, alpha=0.88, linewidth=0)
    ax.fill_between(t_dense, cont_d + pos_d, cont_d + pos_d + res_d,
                    color=TEAL, alpha=0.88, linewidth=0)

    mid = len(t_dense) // 2
    ax.text(t_dense[mid], cont_d[mid] / 2, "Content",
            ha="center", va="center", color="white",
            fontweight="bold", fontsize=8.5)
    ax.text(t_dense[mid], cont_d[mid] + pos_d[mid] / 2, "Position",
            ha="center", va="center", color="white",
            fontweight="bold", fontsize=8.5)
    inter_y = cont_d[mid] + pos_d[mid] + res_d[mid] / 2
    if res_d[mid] > 6:
        ax.text(t_dense[mid], inter_y, "Interaction",
                ha="center", va="center", color="white",
                fontweight="bold", fontsize=8.0)
    else:
        ax.annotate("Interaction", xy=(t_dense[mid], inter_y),
                    xytext=(t_dense[mid], 106), color=TEAL,
                    fontweight="medium", fontsize=7.5, ha="center",
                    arrowprops=dict(arrowstyle="-", color=TEAL, lw=0.7))

    ax.axvline(0.80, color="white", linestyle="--",
               linewidth=1.3, alpha=0.95, zorder=4)
    ax.text(0.795, 111, "published = 0.80",
            ha="right", va="bottom", fontsize=7.2, color=SLATE,
            style="italic")

    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, 116)
    ax.set_xticks(thresholds)
    ax.set_xticklabels([f"{t:.2f}" for t in thresholds])
    ax.set_xlabel(r"$\mathrm{pct}_{\mathrm{pos}0}$ threshold")
    ax.set_ylabel("% of targeted total")
    ax.set_title(r"Polysemy shares vs. threshold ($n = 1{,}103$)",
                 loc="left", pad=8)
    save_both(fig, "v4_appC_threshold_streamgraph")


# ═════════════════════════════════════════════════════════════════
# PAPER FIG 4 — Generation-flip confusion: small-multiples heatmap
# ═════════════════════════════════════════════════════════════════
def fig_confusion_genflip():
    """Three 3x3 confusion matrices (Targeted, Shared-only, Unique-only),
    row-normalized to transition probabilities."""
    cm = json.load(open(A / "reference_layer" / "multimetric" /
                        "gen_confusion_matrices_L41.json"))

    slots = ["D_i", "D_j", "no-match"]
    slot_labels = [r"$D_i$", r"$D_j$", "no-match"]
    panels = [
        ("targeted",     "Targeted",     RED),
        ("shared_only",  "Shared-only",  INK),
        ("unique_only",  "Unique-only",  TEAL),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(CONF_W, 1.75),
                             gridspec_kw=dict(wspace=0.30))

    cmap = LinearSegmentedColormap.from_list(
        "cream_ink", ["#F7FAFC", "#C8D6EE", "#5A7AB6", INK])

    for ax, (key, title, accent) in zip(axes, panels):
        m = np.array([[cm[key][r][c] for c in slots] for r in slots], dtype=float)
        row_sums = m.sum(axis=1, keepdims=True)
        prob = np.divide(m, row_sums, out=np.zeros_like(m), where=row_sums > 0)

        im = ax.imshow(prob, cmap=cmap, vmin=0, vmax=1, aspect="equal")

        for k in range(len(slots) + 1):
            ax.axhline(k - 0.5, color="white", linewidth=1.8, zorder=2)
            ax.axvline(k - 0.5, color="white", linewidth=1.8, zorder=2)

        for i in range(len(slots)):
            for j in range(len(slots)):
                v = prob[i, j]
                n = int(m[i, j])
                if v > 0.55:
                    color, weight = "white", "medium"
                elif v < 0.05:
                    color, weight = MUTED, "normal"
                else:
                    color, weight = "#1A202C", "medium"
                ax.text(j, i - 0.10, f"{v:.2f}", ha="center", va="center",
                        color=color, fontsize=8.0, fontweight=weight, zorder=3)
                ax.text(j, i + 0.28, f"n={n}", ha="center", va="center",
                        color=color, fontsize=6.0, alpha=0.85, zorder=3)

        ax.set_xticks(range(len(slots))); ax.set_xticklabels(slot_labels, fontsize=7.0)
        ax.set_yticks(range(len(slots)))
        ax.set_title(title, loc="left", pad=6, color=accent, fontsize=8.8)
        ax.set_xlabel("post-ablation slot", fontsize=7.4)
        if ax is axes[0]:
            ax.set_yticklabels(slot_labels, fontsize=7.0)
            ax.set_ylabel("baseline slot", fontsize=7.4)
        else:
            ax.set_yticklabels([])
        ax.tick_params(axis="both", length=0)
        for s in ax.spines.values():
            s.set_visible(False)

    cbar = fig.colorbar(im, ax=axes, shrink=0.82, pad=0.02,
                        fraction=0.022, location="right")
    cbar.set_label("transition probability", labelpad=8, fontsize=7.4)
    cbar.outline.set_visible(False)
    cbar.ax.tick_params(length=0, labelsize=6.5)

    fig.suptitle(r"Generation-flip confusion at L41 (row-normalized; $n = 1{,}103$)",
                 fontsize=9.5, fontweight="bold", x=0.012, ha="left", y=1.01)
    save_both(fig, "v4_fig_confusion_genflip")


# ─── run them all ─────────────────────────────────────────────────
if __name__ == "__main__":
    print("Generating paper figures (column-native sizes)...")
    fig0_teaser()
    fig1_six_condition_lollipop()
    fig2_polysemy_waterfall()
    fig6_layer_trajectory_gradient()
    fig7_l0_sweep()
    fig5_shell_trajectory()
    appC_threshold_streamgraph()
    fig_confusion_genflip()
    print(f"Done. Output: {FIG}")
