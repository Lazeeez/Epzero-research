"""Paper figures from verified analysis outputs (aggregate numbers only; no LongEval data).

Usage: python figures.py BUILD_DIR OUTPUT_DIR
Inputs: longeval-heldout-analysis-01/report.json, longeval-heldout-exploratory-01/cohort-ratios.json,
longeval-development-order-01/report.json, longeval-heldout-extension-analysis-01/report.json and the committed
insertion-pilot-summary.json. Writes fig-gap-recall, fig-cohort, fig-recall-gap and fig-insertion-pilot (pdf and png).
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Validated categorical slots 1-3 (all-pairs CVD-safe); aqua needs direct labels, which every series has.
COLOR = {'epzero': '#2a78d6', 'faiss_hnsw': '#eb6834', 'faiss_ivf': '#1baf7a'}
MARKER = {'epzero': 'o', 'faiss_hnsw': 's', 'faiss_ivf': '^'}
INK, MUTED, GRID = '#0b0b0b', '#52514e', '#e4e3df'
NAME = {'epzero': 'Epzero HNSW', 'faiss_hnsw': 'FAISS HNSW', 'faiss_ivf': 'IVF (frozen)'}
plt.rcParams.update({'font.family': 'serif', 'font.size': 8, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': MUTED, 'ytick.color': MUTED, 'axes.linewidth': .6, 'lines.linewidth': 1.5,
                     'pdf.fonttype': 42})


def style(ax):
    ax.grid(axis='y', color=GRID, linewidth=.6)
    ax.set_axisbelow(True)
    for side in ('top', 'right'):
        ax.spines[side].set_visible(False)


def label_end(ax, x, y, text, dy=0):
    ax.annotate(text, (x, y), xytext=(5, dy), textcoords='offset points', va='center', fontsize=7, color=INK)


def gap_recall(build, out):
    r = json.loads((build / 'longeval-heldout-analysis-01/report.json').read_text())
    block = r['secondary']['gpt-oss-120b']
    recall = defaultdict(list)
    for key, value in r['ann'].items():
        backend, mode, seed, step, regime = key.split()
        recall[(backend, mode, regime, int(step))].append(value['mean'])
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.2, 2.1), constrained_layout=True)
    # (backend, primary name, mode, regime, label offset in panel a, label offset in panel b)
    arms = (('epzero', 'epzero_hnsw', 'incremental', 'matched_recall', 0, -7),
            ('faiss_hnsw', 'faiss_hnsw', 'incremental', 'matched_recall', 4, -6),
            ('faiss_ivf', 'ivf_frozen', 'frozen_incremental', 'current_calibration', -4, 6))
    steps = [1, 2, 3]
    for backend, name, mode, regime, dy, dyb in arms:
        gaps = [block[name]['gap_by_snapshot'][str(s)] for s in steps]
        y = [g['mean_difference'] for g in gaps]
        err = [[g['mean_difference'] - g['percentile_interval'][0] for g in gaps],
               [g['percentile_interval'][1] - g['mean_difference'] for g in gaps]]
        offset = {'epzero': -.08, 'faiss_hnsw': 0, 'faiss_ivf': .08}[backend]
        xs = [s + offset for s in steps]
        a.errorbar(xs, y, yerr=err, color=COLOR[backend], marker=MARKER[backend], markersize=4, capsize=0,
                   elinewidth=.8)
        label_end(a, xs[-1], y[-1], NAME[backend], dy)
        ys = [sum(recall[(backend, mode, regime, s)]) / 3 for s in steps]
        b.plot(steps, ys, color=COLOR[backend], marker=MARKER[backend], markersize=4)
        label_end(b, steps[-1], ys[-1], NAME[backend] + (' (ef 200)' if backend == 'epzero' else ''), dyb)
    ys = [sum(recall[('faiss_hnsw', 'incremental', 'frozen', s)]) / 3 for s in steps]
    b.plot(steps, ys, color=COLOR['faiss_hnsw'], marker=MARKER['faiss_hnsw'], markersize=4, linestyle='--',
           markerfacecolor='white')
    label_end(b, steps[-1], ys[-1], 'FAISS HNSW (ef 200)', 5)
    a.axhline(0, color=MUTED, linewidth=.6)
    a.set_ylim(-.0195, .002)
    a.set_ylabel('ANN − exact nDCG@10')
    a.set_title('(a) Effectiveness gap, calibrated', fontsize=8, color=INK, loc='left')
    b.set_ylabel('Neighbour recall@100')
    b.set_ylim(.925, .97)
    b.set_title('(b) Neighbour recall', fontsize=8, color=INK, loc='left')
    for ax in (a, b):
        style(ax)
        ax.set_xticks(steps, ['s1', 's2', 's3'])
        ax.set_xlim(.75, 4.3)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig-gap-recall.{ext}', dpi=300)
    plt.close(fig)


def cohort(build, out):
    held = json.loads((build / 'longeval-heldout-exploratory-01/cohort-ratios.json').read_text())['rows']
    dev = json.loads((build / 'longeval-development-order-01/report.json').read_text())['cohort_misses']
    fig, (a, b) = plt.subplots(1, 2, figsize=(5.2, 1.9), constrained_layout=True, gridspec_kw={'width_ratios': [1.2, 1]})
    rows = (('epzero/incremental/frozen', 'epzero', 'Epzero HNSW'),
            ('faiss_hnsw/incremental/frozen', 'faiss_hnsw', 'FAISS HNSW'),
            ('faiss_ivf/frozen_incremental/current_calibration', 'faiss_ivf', 'IVF, frozen'),
            ('faiss_ivf/refreshed_rebuild/current_calibration', 'faiss_ivf', 'IVF, refreshed'))
    for i, (arm, backend, text) in enumerate(rows):
        r = next(x for x in held if x['arm'] == arm and x['snapshot'] == 3)
        lo, hi = r['interval_95']
        a.errorbar(r['ratio'], i, xerr=[[r['ratio'] - lo], [hi - r['ratio']]], color=COLOR[backend],
                   marker=MARKER[backend], markersize=4, elinewidth=.9, capsize=0,
                   markerfacecolor='white' if 'refreshed' in text else COLOR[backend])
    a.set_yticks(range(len(rows)), [t for _, _, t in rows])
    a.invert_yaxis()
    a.axvline(1, color=MUTED, linewidth=.6)
    a.set_xlim(.9, 1.9)
    a.set_xlabel('Miss ratio, later vs snapshot-1 documents')
    a.set_title('(a) Held-out, snapshot 3', fontsize=8, color=INK, loc='left')

    def ratio(arm, step):
        cells = {(x['inserted_at'], x['exact_rank_band']): (x['exact_neighbours'], x['missed'])
                 for x in dev if x['arm'] == arm and x['snapshot'] == step}
        bands = ('1-10', '11-50', '51-100')
        obs = sum(cells[(c, band)][1] for c in (2, 3) for band in bands if (c, band) in cells)
        exp =sum(cells[(c, band)][0] * cells[(1, band)][1] / cells[(1, band)][0] for c in (2, 3) for band in bands if (c, band) in cells)
        return obs / exp

    labels = []
    for y, (backend, step) in enumerate((('epzero', 2), ('epzero', 3), ('faiss_hnsw', 2), ('faiss_hnsw', 3))):
        appended, shuffled = ratio(f'{backend}/incremental/frozen', step), ratio(f'{backend}/shuffled/frozen', step)
        b.plot([shuffled, appended], [y, y], color=COLOR[backend], linewidth=1.2)
        b.plot(appended, y, marker=MARKER[backend], color=COLOR[backend], markersize=4)
        b.plot(shuffled, y, marker=MARKER[backend], color=COLOR[backend], markersize=4, markerfacecolor='white')
        labels.append((y, f"{NAME[backend].split()[0]}, s{step}"))
    # Filled = appended (incremental) order, hollow = shuffled order; stated in the figure caption.
    b.set_yticks([y for y, _ in labels], [t for _, t in labels])
    b.invert_yaxis()
    b.axvline(1, color=MUTED, linewidth=.6)
    b.set_xlim(.9, 1.9)
    b.set_xlabel('Miss ratio, development')
    b.set_title('(b) Insertion order', fontsize=8, color=INK, loc='left')
    for ax in (a, b):
        style(ax)
        ax.grid(axis='y', visible=False)
        ax.grid(axis='x', color=GRID, linewidth=.6)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig-cohort.{ext}', dpi=300)
    plt.close(fig)


def recall_gap(build, out):
    """Effectiveness gap against neighbour recall across the ef sweep; snapshots on a one-hue blue ramp."""
    r = json.loads((build / 'longeval-heldout-extension-analysis-01/report.json').read_text())
    ramp = {1: '#6da7ec', 2: '#2a78d6', 3: '#104281'}
    fig, axes = plt.subplots(1, 2, figsize=(5.2, 2.1), constrained_layout=True, sharey=True)
    for ax, backend in zip(axes, ('epzero', 'faiss_hnsw')):
        for step in (1, 2, 3):
            pts = sorted((c['recall_mean'], c['mean_difference'], c['ef']) for c in r['curve']
                         if c['backend'] == backend and c['mode'] == 'incremental' and c['snapshot'] == step)
            ax.plot([x for x, _, _ in pts], [y for _, y, _ in pts], color=ramp[step], marker='o', markersize=3.5,
                    linewidth=1.3)
            x, y, _ = pts[0]
            ax.annotate(f's{step}', (x, y), xytext=(-4, 0), textcoords='offset points', ha='right', va='center',
                        fontsize=7, color=INK)
            if step == 3:
                for x, y, ef in pts[1:4]:
                    ax.annotate(f'ef {ef}', (x, y), xytext=(3, -8), textcoords='offset points', fontsize=6, color=MUTED)
        ax.axhline(0, color=MUTED, linewidth=.6)
        ax.set_xlim(.87, 1.0)
        ax.set_xlabel('Neighbour recall@100')
        ax.set_title(f"({'ab'[backend != 'epzero']}) {NAME[backend]}, incremental", fontsize=8, color=INK, loc='left')
        style(ax)
    axes[0].set_ylabel('ANN − exact nDCG@10')
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig-recall-gap.{ext}', dpi=300)
    plt.close(fig)


def insertion_pilot(summary, out):
    """Sample pilot: miss ratio of the last arrival group against the first, by insertion order."""
    r = json.loads(Path(summary).read_text())
    orders = (('appended', 'As arrived'), ('shuffled', 'Shuffled'), ('reversed', 'Reversed'))
    fig, ax = plt.subplots(figsize=(5.2, 1.9), constrained_layout=True)
    for i, (order, _) in enumerate(orders):
        for backend, dy in (('epzero', -.17), ('faiss_hnsw', .17)):
            mean = r['contrasts'][backend]['cohort3_vs_cohort1']['seed_mean'][order]
            lo, hi = mean['interval_95']
            y = i + dy
            seeds = [b['miss_ratio_cohort3_vs_cohort1']['ratio'] for b in r['builds']
                     if b['backend'] == backend and b['order'] == order]
            ax.plot(seeds, [y] * len(seeds), linestyle='none', marker='|', markersize=7, markeredgewidth=.9,
                    color=COLOR[backend], alpha=.55)
            ax.errorbar(mean['value'], y, xerr=[[mean['value'] - lo], [hi - mean['value']]], color=COLOR[backend],
                        marker=MARKER[backend], markersize=4.5, elinewidth=.9, capsize=0)
            if i == 0:
                ax.annotate(NAME[backend], (hi, y), xytext=(5, 0), textcoords='offset points', va='center',
                            fontsize=7, color=INK)
    ax.set_yticks(range(len(orders)), [text for _, text in orders])
    ax.set_ylim(len(orders) - .5, -.5)
    ax.axvline(1, color=MUTED, linewidth=.6)
    ax.set_xlim(.6, 2.0)
    # Sample size and search width are stated in the caption.
    ax.set_xlabel('Miss ratio, snapshot-3 arrivals vs snapshot-1 arrivals')
    style(ax)
    ax.grid(axis='y', visible=False)
    ax.grid(axis='x', color=GRID, linewidth=.6)
    for ext in ('pdf', 'png'):
        fig.savefig(out / f'fig-insertion-pilot.{ext}', dpi=300)
    plt.close(fig)


if __name__ == '__main__':
    build, out = Path(sys.argv[1]), Path(sys.argv[2])
    out.mkdir(parents=True, exist_ok=True)
    gap_recall(build, out)
    cohort(build, out)
    recall_gap(build, out)
    insertion_pilot(Path(__file__).resolve().parents[1] / 'insertion-pilot-summary.json', out)
    print('written', sorted(p.name for p in out.iterdir()))
