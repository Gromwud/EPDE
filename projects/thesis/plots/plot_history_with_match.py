"""Per-system candidate-history plot, one figure per pipeline.

Combines the all-candidates-throughout-search view of
``plot_objectives_density.py`` (every candidate from every Pareto
level across every epoch, read from the sidecar
``<rep>.history.json``) with the truth-matching overlay used in
``plot_pareto_correct.py``.

LEGACY and NEW live in different objective spaces:
  LEGACY: (L2 discrepancy, complexity) — set by
          ``use_legacy_multiobjective_function`` in main_structures.py.
  NEW:    (WAPE discrepancy, instability) — set by
          ``use_new_multiobjective_function``.
Each pipeline gets its own figure so the two objective spaces don't
have to share axes. For each system we pick the first seed where
both pipelines have a history sidecar (when available), preferring
seeds where NEW matched truth so the star overlay is meaningful.

Output: per-system PNGs split by pipeline at
``projects/thesis/figures/history_match_<pipeline>_<system>.png``
(one for LEGACY, one for NEW) plus two grid montages at
``projects/thesis/figures/history_match_grid_<pipeline>.png``.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import warnings

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Seaborn theme: clean ticks, soft grid, sans-serif. ``whitegrid`` keeps
# the log-log gridlines visible against the candidate cloud while
# softening the surrounding chrome.
sns.set_theme(style='whitegrid', context='notebook',
              rc={'axes.spines.right': False, 'axes.spines.top': False})
# Bright, perceptually-uniform candidate colour (Crest mid-tone) for NEW
# pipeline; warm flare mid-tone for LEGACY so the two clouds separate
# without becoming muddy when they overlap.
_CLOUD_COLOR_NEW = sns.color_palette('crest', n_colors=5)[2]
_CLOUD_COLOR_LEGACY = sns.color_palette('flare', n_colors=5)[2]
# High-contrast truth markers: Set1 red star for NEW, Set1 blue diamond
# for LEGACY.
_MATCH_COLOR_NEW = sns.color_palette('Set1', n_colors=9)[0]
_MATCH_COLOR_LEGACY = sns.color_palette('Set1', n_colors=9)[1]

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_DIR = os.path.abspath(os.path.join(_THIS_DIR, '..'))
_RESULTS_DIR = os.path.join(_PROJECT_DIR, 'results')
_FIG_DIR = os.path.join(_PROJECT_DIR, 'figures')

SYSTEM_ORDER = [
    'ode', 'vdp', 'lorenz', 'lv',
    'ac', 'burgers_inviscid', 'burgers_viscous',
    'kdv', 'kdv_cossin', 'ks',
    'wave', 'pde_compound', 'pde_divide',
    'ns',
]


def _load_history(rep_dir: str, history_basename: str):
    hist_path = os.path.join(rep_dir, history_basename)
    try:
        with open(hist_path, 'r', encoding='utf-8') as fh:
            payload = json.load(fh)
        return payload.get('candidate_history')
    except (OSError, json.JSONDecodeError) as exc:
        warnings.warn(f"history load failed: {hist_path}: {exc!r}")
        return None


def _iter_reps_with_history(system: str, pipeline: str):
    """Yield (rec, rep_path) for every rep of ``pipeline`` whose sidecar
    history exists, in seed-sorted order."""
    pattern = os.path.join(_RESULTS_DIR, system, f'{pipeline}_rep*.json')
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        hist = rec.get('history_path')
        if not hist:
            continue
        if not os.path.exists(os.path.join(os.path.dirname(path), hist)):
            continue
        yield rec, path


def _find_first_success(system: str, pipeline: str = 'new'):
    """First rep of ``pipeline`` with structural_success AND a sidecar."""
    for rec, path in _iter_reps_with_history(system, pipeline):
        if rec.get('structural_success'):
            return rec, path
    return None, None


def _find_first_with_history(system: str, pipeline: str = 'new'):
    """First rep of ``pipeline`` with a sidecar (success not required)."""
    for rec, path in _iter_reps_with_history(system, pipeline):
        return rec, path
    return None, None


def _find_paired_with_history(system: str):
    """Return ((legacy_rec, legacy_path), (new_rec, new_path)) for a single
    seed where BOTH pipelines have a history sidecar. Prefers seeds where
    NEW matched truth so the truth-match overlay is meaningful; falls back
    to the lowest shared seed otherwise. Either side may be None if that
    pipeline never produced a sidecar for the system.
    """
    legacy_by_seed = {rec.get('seed'): (rec, path)
                      for rec, path in _iter_reps_with_history(system, 'legacy')}
    new_by_seed = {rec.get('seed'): (rec, path)
                   for rec, path in _iter_reps_with_history(system, 'new')}
    shared = sorted(set(legacy_by_seed) & set(new_by_seed),
                    key=lambda s: (s is None, s))
    # Preferred: shared seed where NEW matched truth.
    for seed in shared:
        rec_new, _ = new_by_seed[seed]
        if rec_new.get('structural_success'):
            return legacy_by_seed[seed], new_by_seed[seed]
    # Fallback: any shared seed.
    if shared:
        seed = shared[0]
        return legacy_by_seed[seed], new_by_seed[seed]
    # No shared seed — pick each side independently so the figure still
    # shows whichever cloud(s) exist.
    legacy = next(iter(legacy_by_seed.values()), (None, None))
    new = next(iter(new_by_seed.values()), (None, None))
    # Prefer NEW reps with truth match if available.
    new_success = _find_first_success(system, 'new')
    if new_success[0] is not None:
        new = new_success
    return legacy, new


def _gather_history_points(rec: dict, rep_path: str) -> np.ndarray:
    """Return the SET of unique candidate objective vectors seen across
    the whole search (not the time-stacked history).

    MOEA/D writes the full population once per epoch into ``_hist``, so
    a candidate that survives K epochs would contribute K identical
    rows. Deduplicating collapses those to a single point and removes
    the alpha-stacking artefact (some dots darker than others) that
    otherwise misled readers into thinking the population was non-
    uniform. Each retained point corresponds to a distinct objective
    vector that MOEA/D produced at some point in the search.
    """
    rep_dir = os.path.dirname(rep_path)
    hist_basename = rec.get('history_path')
    if not hist_basename:
        return np.empty((0, 0))
    candidate_history = _load_history(rep_dir, hist_basename)
    if not candidate_history:
        return np.empty((0, 0))
    chunks = []
    for snap in candidate_history:
        if not snap:
            continue
        arr = np.asarray(snap, dtype=float)
        if arr.ndim != 2 or arr.size == 0:
            continue
        chunks.append(arr)
    if not chunks:
        return np.empty((0, 0))
    arr = np.vstack(chunks)
    # Round to 6 sig figs so floating-point jitter doesn't dodge dedup.
    _, idx = np.unique(np.round(arr, 6), axis=0, return_index=True)
    # ``np.unique`` returns lex-sorted indices; restoring original order
    # keeps the cloud's first-appearance temporal hint without
    # duplicating points.
    return arr[sorted(idx)]


def _final_match_points(rec: dict) -> np.ndarray:
    """Return (n_match_solutions × n_objectives) for hamming==0 Pareto-0 sols."""
    objs = rec.get('objectives_per_solution') or []
    hams = rec.get('hamming_per_solution') or []
    rows = []
    for i, obj in enumerate(objs):
        if i >= len(hams) or hams[i] != 0:
            continue
        if obj is None:
            continue
        arr = np.asarray(obj, dtype=float).reshape(-1)
        if arr.size == 0 or arr.size % 2 != 0:
            continue
        rows.append(arr)
    if not rows:
        return np.empty((0, 0))
    # Pad to the same width if needed (shouldn't happen — objectives shape is fixed per system).
    width = max(r.size for r in rows)
    rows = [r if r.size == width else np.concatenate([r, np.full(width - r.size, np.nan)])
            for r in rows]
    return np.vstack(rows)


def _cohort_match_points(system: str, pipeline: str = 'new') -> np.ndarray:
    """Aggregate hamming==0 Pareto-0 objective vectors from ALL
    structurally-successful reps of ``system`` for ``pipeline``. Used
    when the rep providing the history cloud didn't itself match truth —
    the truth coordinates still live in the cohort and are meaningful to
    overlay because all reps share the system's objective-space scale.
    """
    all_rows = []
    width = 0
    pattern = os.path.join(_RESULTS_DIR, system, f'{pipeline}_rep*.json')
    for path in sorted(glob.glob(pattern)):
        if path.endswith('.history.json'):
            continue
        try:
            with open(path, 'r', encoding='utf-8') as fh:
                rec = json.load(fh)
        except (OSError, json.JSONDecodeError):
            continue
        if not rec.get('structural_success'):
            continue
        m = _final_match_points(rec)
        if m.size == 0:
            continue
        width = max(width, m.shape[1])
        all_rows.append(m)
    if not all_rows:
        return np.empty((0, 0))
    padded = []
    for m in all_rows:
        if m.shape[1] < width:
            pad = np.full((m.shape[0], width - m.shape[1]), np.nan)
            m = np.concatenate([m, pad], axis=1)
        padded.append(m)
    return np.vstack(padded)


def _pipeline_payload(system: str, rec, rep_path):
    """Bundle the cloud + match points + provenance flag for one pipeline.
    Returns None if no history is available for that pipeline.
    """
    if rec is None or rep_path is None:
        return None
    history = _gather_history_points(rec, rep_path)
    if history.size == 0:
        return None
    matches = _final_match_points(rec)
    matches_from_cohort = False
    if matches.size == 0:
        # Currently this rec is for a known pipeline (legacy/new); read the
        # pipeline label off the rec so we cohort-match within the same
        # pipeline (mixing legacy and new truth-match stars would mislead).
        pipeline = rec.get('pipeline', 'new')
        cohort = _cohort_match_points(system, pipeline=pipeline)
        if cohort.size:
            matches = cohort
            matches_from_cohort = True
    return {
        'rec': rec,
        'rep_path': rep_path,
        'history': history,
        'matches': matches,
        'matches_from_cohort': matches_from_cohort,
    }


# Per-pipeline plotting style: cloud colour, truth-match colour, marker.
_PIPELINE_STYLES = {
    'legacy': dict(cloud=_CLOUD_COLOR_LEGACY, match=_MATCH_COLOR_LEGACY,
                   marker='D', size=70,
                   y_label='complexity', x_label='discrepancy (L2)'),
    'new':    dict(cloud=_CLOUD_COLOR_NEW,    match=_MATCH_COLOR_NEW,
                   marker='*', size=220,
                   y_label='instability', x_label='discrepancy (WAPE)'),
}


def _draw_single(ax, payload, eq_idx: int, pipeline: str) -> int:
    """Draw one pipeline's cloud + truth-match into ``ax``. Returns the
    candidate count drawn. Does not set axis scales/labels — that's the
    caller's job so we can avoid setting them on a hidden panel."""
    ix, iy = 2 * eq_idx, 2 * eq_idx + 1
    if payload is None:
        return 0
    history = payload['history']
    matches = payload['matches']
    if iy >= history.shape[1]:
        return 0
    style = _PIPELINE_STYLES[pipeline]
    lx = np.log10(np.maximum(history[:, ix], 1e-12))
    ly = np.log10(np.maximum(history[:, iy], 1e-12))
    m = np.isfinite(lx) & np.isfinite(ly)
    cloud_label = f'{pipeline} candidates (n={int(m.sum())})'
    ax.scatter(10 ** lx[m], 10 ** ly[m],
               color=style['cloud'], alpha=0.55, s=16, edgecolors='none',
               label=cloud_label)
    drawn = int(m.sum())
    if matches.size and iy < matches.shape[1]:
        mx = matches[:, ix]; my = matches[:, iy]
        mm = np.isfinite(mx) & np.isfinite(my)
        if mm.any():
            ax.scatter(mx[mm], my[mm], color=style['match'],
                       edgecolors='black', linewidths=0.7,
                       marker=style['marker'], s=style['size'], zorder=5,
                       label=f'{pipeline} truth-match (n={int(mm.sum())})')
    return drawn


def _match_count(payload):
    if payload is None or payload['matches'].size == 0:
        return 0
    first = (payload['matches'][:, :2] if payload['matches'].shape[1] >= 2
             else payload['matches'])
    return int(np.isfinite(first).all(axis=1).sum())


def _seed_blurb(payload, label):
    if payload is None:
        return f"{label}: (no history)"
    seed = payload['rec'].get('seed')
    n_cands = payload['history'].shape[0]
    return f"{label}: seed {seed}, {n_cands} unique cands"


def plot_one(legacy, new_, system: str, out_dir: str, show: bool,
             pipelines: tuple = ('legacy', 'new'),
             out_suffix: str = ''):
    """Plot one system with LEGACY (top row) and NEW (bottom row) on
    separate subplots because their objective spaces differ.

    ``legacy`` and ``new_`` are payload dicts from ``_pipeline_payload``
    or ``None`` if that pipeline has no history. ``pipelines`` selects
    which rows to render (e.g. ``('legacy',)`` or ``('new',)`` for the
    split variants). ``out_suffix`` lands inside the output filename
    so the three variants don't overwrite each other.
    """
    payload_by_pipeline = {'legacy': legacy, 'new': new_}
    rows_spec = [(p, payload_by_pipeline[p]) for p in pipelines]
    # If every requested pipeline is missing, skip — nothing to draw.
    if all(p is None for _, p in rows_spec):
        print(f"  [{system}{out_suffix}] no usable history "
              f"for requested pipelines; skipping")
        return None

    n_equations = max(
        (max(1, p['history'].shape[1] // 2)
         for _, p in rows_spec if p is not None),
        default=1,
    )

    fig, axes = plt.subplots(
        len(rows_spec), n_equations,
        figsize=(5.2 * n_equations, 4.6 * len(rows_spec)),
        squeeze=False,
    )

    for row_idx, (pipeline, payload) in enumerate(rows_spec):
        style = _PIPELINE_STYLES[pipeline]
        row_axes = axes[row_idx]
        # Determine this pipeline's per-equation count separately so we
        # only enable panels that exist for the pipeline that's present.
        pipe_n_eq = (max(1, payload['history'].shape[1] // 2)
                     if payload is not None else 0)
        first_drawn_ax = None
        for eq_idx, ax in enumerate(row_axes):
            if payload is None or eq_idx >= pipe_n_eq:
                if payload is None and eq_idx == 0:
                    # Annotate the empty row so the figure shows that
                    # the pipeline has no history available.
                    ax.text(0.5, 0.5,
                            f"{pipeline.upper()}: no sidecar history available",
                            ha='center', va='center',
                            transform=ax.transAxes, fontsize=11,
                            color='dimgray')
                ax.axis('off')
                continue
            _draw_single(ax, payload, eq_idx, pipeline)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.set_xlabel(style['x_label'])
            ax.set_ylabel(style['y_label'])
            ax.grid(alpha=0.3, which='both', linewidth=0.4)
            n_match = _match_count(payload)
            if first_drawn_ax is None:
                # Push the title further up on the legend-host panel so
                # the legend (which sits just above the axes spine) has
                # room between the data plane and the title.
                title_pad = 24
                first_drawn_ax = ax
            else:
                title_pad = 6
            ax.set_title(
                f'{pipeline.upper()} — Equation {eq_idx} '
                f'(truth-match n={n_match})',
                fontsize=10, pad=title_pad,
            )
        # One legend per pipeline row, anchored ABOVE the leftmost drawn
        # panel — outside the objectives plane, centred, framed by the
        # raised title above and the axes spine below.
        if first_drawn_ax is not None:
            first_drawn_ax.legend(loc='lower center',
                                  bbox_to_anchor=(0.5, 1.02),
                                  ncol=2, fontsize=8)
            sns.move_legend(first_drawn_ax, loc='lower center',
                            bbox_to_anchor=(0.5, 1.02),
                            ncol=2, frameon=False, fontsize=8)

    cohort_blurbs = []
    if 'new' in pipelines and new_ and new_['matches_from_cohort']:
        cohort_blurbs.append('NEW stars from cohort')
    if 'legacy' in pipelines and legacy and legacy['matches_from_cohort']:
        cohort_blurbs.append('LEGACY stars from cohort')
    cohort_note = f"   ({'; '.join(cohort_blurbs)})" if cohort_blurbs else ''

    # Pick a payload that exists to read epoch metadata.
    ref_payload = next((p for _, p in rows_spec if p is not None), None)
    n_epochs = ref_payload['rec'].get('n_epochs', '?') if ref_payload else '?'
    blurbs = [_seed_blurb(payload_by_pipeline[p], p.upper())
              for p in pipelines]
    fig.suptitle(
        f"{system.upper()}  —  " + '   |   '.join(blurbs)
        + f"   |   {n_epochs} epochs{cohort_note}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.94])

    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f'history_match{out_suffix}_{system}.png')
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'  [{system}{out_suffix}] wrote {out_path}')
    return out_path


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--out-dir', default=_FIG_DIR)
    args = p.parse_args(argv)
    os.makedirs(args.out_dir, exist_ok=True)

    plotted = []
    for system in SYSTEM_ORDER:
        (legacy_rec, legacy_path), (new_rec, new_path) = _find_paired_with_history(system)
        legacy_payload = _pipeline_payload(system, legacy_rec, legacy_path)
        new_payload = _pipeline_payload(system, new_rec, new_path)
        if legacy_payload is None and new_payload is None:
            print(f'  [skip] {system}: no sidecar history available for '
                  f'either pipeline (re-run with history recording to populate)')
            continue
        if legacy_payload is None:
            print(f'  [{system}] no legacy history; plotting NEW only')
        elif new_payload is None:
            print(f'  [{system}] no new history; plotting LEGACY only')
        # Per-system PNGs split by pipeline:
        #   history_match_legacy_<system>.png  — LEGACY only.
        #   history_match_new_<system>.png     — NEW only.
        # Each pipeline's panels stay in its own objective space.
        out_legacy = plot_one(legacy_payload, new_payload, system,
                              args.out_dir, show=False,
                              pipelines=('legacy',), out_suffix='_legacy')
        out_new = plot_one(legacy_payload, new_payload, system,
                           args.out_dir, show=False,
                           pipelines=('new',), out_suffix='_new')
        if out_legacy or out_new:
            plotted.append((system, legacy_payload, new_payload))

    # Two grid montages, one per pipeline. Each stays in a single
    # objective space (LEGACY: discrepancy(L2) vs complexity;
    # NEW: discrepancy(WAPE) vs instability).
    for pipeline in ('legacy', 'new'):
        _write_grid_montage(plotted, pipeline, args.out_dir)

    print(f'\nDone. {len(plotted)} per-system figures written.')
    return 0 if plotted else 1




def _write_grid_montage(plotted: list, pipeline: str, out_dir: str):
    """Write a per-pipeline grid montage:
    ``history_match_grid_<pipeline>.png`` — one row per system with
    history for ``pipeline``, columns indexed by equation. All panels
    share the pipeline's objective space (LEGACY: discrepancy(L2) /
    complexity; NEW: discrepancy(WAPE) / instability).
    """
    style = _PIPELINE_STYLES[pipeline]
    # Filter to systems that actually have a payload for this pipeline.
    rows = []
    for system, legacy_payload, new_payload in plotted:
        payload = legacy_payload if pipeline == 'legacy' else new_payload
        if payload is None:
            continue
        rows.append((system, payload))
    if not rows:
        print(f'  [grid:{pipeline}] no systems with history; skipping')
        return None

    ncols = max(1, max(max(1, payload['history'].shape[1] // 2)
                       for _, payload in rows))
    nrows = len(rows)
    fig, axes = plt.subplots(nrows, ncols,
                              figsize=(5.0 * ncols, 3.6 * nrows),
                              squeeze=False)
    first_drawn_ax = None
    for row_idx, (system, payload) in enumerate(rows):
        pipe_n_eq = max(1, payload['history'].shape[1] // 2)
        for eq_idx in range(ncols):
            ax = axes[row_idx][eq_idx]
            if eq_idx >= pipe_n_eq:
                ax.axis('off')
                continue
            _draw_single(ax, payload, eq_idx, pipeline)
            ax.set_xscale('log')
            ax.set_yscale('log')
            ax.grid(alpha=0.3, which='both', linewidth=0.4)
            if eq_idx == 0:
                ax.set_ylabel(f"{system}\n{style['y_label']}", fontsize=9)
            else:
                ax.set_ylabel(style['y_label'], fontsize=9)
            if row_idx == nrows - 1:
                ax.set_xlabel(style['x_label'], fontsize=9)
            if row_idx == 0:
                ax.set_title(f'Equation {eq_idx}', fontsize=10)
            if first_drawn_ax is None:
                first_drawn_ax = ax
    if first_drawn_ax is not None:
        first_drawn_ax.legend(loc='lower center',
                              bbox_to_anchor=(0.5, 1.02),
                              ncol=2, fontsize=8)
        sns.move_legend(first_drawn_ax, loc='lower center',
                        bbox_to_anchor=(0.5, 1.02),
                        ncol=2, frameon=False, fontsize=8)
    fig.suptitle(
        f"Candidate-objective history per system — {pipeline.upper()} "
        f"({style['x_label']} vs {style['y_label']})",
        fontsize=12,
    )
    fig.tight_layout(rect=[0, 0, 1.0, 0.97])
    grid_path = os.path.join(out_dir, f'history_match_grid_{pipeline}.png')
    fig.savefig(grid_path, dpi=160, bbox_inches='tight')
    plt.close(fig)
    print(f'  wrote {grid_path}')
    return grid_path


if __name__ == '__main__':
    sys.exit(main())
