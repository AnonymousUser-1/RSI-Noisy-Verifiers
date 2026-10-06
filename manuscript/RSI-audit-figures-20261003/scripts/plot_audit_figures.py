"""Render source-complete audit map, transition detail and explanatory surface."""
from pathlib import Path
import argparse
import csv
import hashlib
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.lines import Line2D
from matplotlib.text import Text
import yaml

ROOT = Path(__file__).resolve().parents[1]


def read_rows(path):
    with path.open(encoding='utf-8', newline='') as f:
        return list(csv.DictReader(f))


def grid(rows):
    Ls = np.array(sorted({float(r['L']) for r in rows}))
    qs = np.array(sorted({float(r['q']) for r in rows}))
    values = {(float(r['L']), float(r['q'])): float(r['delta_P_pp']) for r in rows}
    if len(values) != len(rows) or len(rows) != len(Ls) * len(qs):
        raise ValueError('Incomplete or duplicate rectangular grid')
    z = np.array([[values[(L, q)] for L in Ls] for q in qs])
    if not np.all(np.isfinite(z)):
        raise ValueError('Nonfinite response')
    return Ls, qs * 100, z


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--data-dir', type=Path, default=ROOT / 'data/scan-v1')
    cli.add_argument('--out-dir', type=Path, required=True)
    args = cli.parse_args()
    if args.out_dir.exists():
        cli.error('Use a new render directory.')
    palette = yaml.safe_load((ROOT / '.plotting/palettes.yaml').read_text())
    brief = {name: json.loads((ROOT / f'.plotting/briefs/{name}.json').read_text())
             for name in ('audit_map', 'audit_detail', 'audit_surface')}
    report = json.loads((args.data_dir / 'scan_validation.json').read_text())
    if report['status'] != 'passed':
        raise ValueError('Unverified scan data')
    for filename, expected in report['data_hashes'].items():
        if hashlib.sha256((args.data_dir / filename).read_bytes()).hexdigest() != expected:
            raise ValueError('Changed scan file: ' + filename)
    L, q, z = grid(read_rows(args.data_dir / 'main_grid.csv'))
    Ld, qd, zd = grid(read_rows(args.data_dir / 'detail_grid.csv'))
    boundary = read_rows(args.data_dir / 'critical_boundary.csv')
    bx = np.array([float(r['L']) for r in boundary])
    by = np.array([float(r['q']) * 100 for r in boundary])
    uniform = read_rows(args.data_dir / 'uniform_threshold.csv')
    slices = read_rows(args.data_dir / 'parameter_slices.csv')
    vmin, vmax = palette['effect']['limits_pp']
    if z.min() < vmin or z.max() > vmax:
        raise ValueError('Color scale would clip data')
    cmap = plt.get_cmap(palette['effect']['colormap'])
    norm = Normalize(vmin=vmin, vmax=vmax)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 8,
        'axes.labelsize': 9, 'axes.linewidth': .65, 'axes.edgecolor': '#38434A',
        'xtick.color': '#38434A', 'ytick.color': '#38434A',
        'xtick.major.width': .65, 'ytick.major.width': .65,
        'xtick.major.size': 3, 'ytick.major.size': 3,
        'pdf.fonttype': 42, 'svg.fonttype': 'none', 'path.simplify': False,
        'savefig.facecolor': 'white', 'axes.spines.top': False, 'axes.spines.right': False})
    args.out_dir.mkdir(parents=True)
    records = {}

    def save(fig, name, data_note, inspect_2d=True):
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        w, h = fig.canvas.get_width_height()
        outside = []
        if inspect_2d:
            for text in fig.findobj(Text):
                if text.get_visible() and text.get_text():
                    box = text.get_window_extent(renderer)
                    if box.width and box.height and (box.x0 < -.1 or box.y0 < -.1 or box.x1 > w+.1 or box.y1 > h+.1):
                        outside.append(text.get_text())
            if outside:
                raise ValueError(f'{name}: text outside canvas: {outside}')
        for fmt in brief[name]['export']['formats']:
            meta = {'Creator': 'RSI audit theory; scripts/plot_audit_figures.py'} if fmt in ('pdf', 'svg') else None
            fig.savefig(args.out_dir / f'{name}.{fmt}', dpi=brief[name]['export']['dpi'], metadata=meta)
        records[name] = {'status': 'rendered', 'data_geometry': data_note, 'text_outside_canvas': outside,
                         'size_mm': brief[name]['figsize_mm'], 'color_limits_pp': [vmin, vmax]}
        plt.close(fig)

    # Manuscript map and exact response profiles; physical size is fixed first.
    fig = plt.figure(figsize=np.array(brief['audit_map']['figsize_mm']) / 25.4, layout='constrained')
    gs = fig.add_gridspec(1, 3, width_ratios=[1.28, .065, 1.0], wspace=.22)
    ax = fig.add_subplot(gs[0, 0]); cbax = fig.add_subplot(gs[0, 1]); cuts = fig.add_subplot(gs[0, 2])
    fig.get_layout_engine().set(w_pad=.045, h_pad=.055, wspace=.16)
    mesh = ax.pcolormesh(L, q, z, shading='nearest', cmap=cmap, norm=norm, rasterized=False,
                        edgecolors='face', linewidth=.35, antialiased=False)
    if not np.array_equal(np.asarray(mesh.get_array()), z):
        raise ValueError('Heatmap geometry changed')
    ax.plot(bx, by, color='#263640', linewidth=1.25)
    ax.axhline(25, color='#263640', linewidth=1.05, linestyle=(0, (4, 2.4)))
    ax.text(2, 27.4, r'Uniform guarantee: $q=25\%$', fontsize=7.3, color='#28343C')
    ax.annotate(r'$\Delta P=0$', (15, 20), xytext=(4, -15), textcoords='offset points', fontsize=8,
                color='#263640')
    ax.text(3, 43, 'Improvement', fontsize=9.2, color='#234C4E', fontweight='bold')
    ax.text(15.5, 4, 'Harm', fontsize=9.2, color='#4C3A29', fontweight='bold')
    ax.plot(10, 17.5, 'o', ms=3.8, mfc='white', mec='#263640', mew=1, zorder=8)
    ax.set(xlim=(1, 20), ylim=(0, 50), xticks=[1, 5, 10, 15, 20], yticks=[0, 10, 20, 30, 40, 50],
           xlabel='Input magnitude, $L$ (1)', ylabel='High-group audit probability, $q$ (%)')
    ax.text(-.20, 1.03, '(a)', transform=ax.transAxes, fontsize=10, fontweight='bold')
    cb = fig.colorbar(mesh, cax=cbax, ticks=[-20, -10, 0, 10, 20])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face'); cb.solids.set_linewidth(.35)
    cb.set_label('Accuracy change (pp)', fontsize=8, labelpad=6)
    cb.outline.set_visible(False)
    for value in (3, 10, 20):
        rows = [r for r in slices if float(r['L']) == value]
        sx = np.array([100 * float(r['q']) for r in rows])
        sy = np.array([float(r['delta_P_pp']) for r in rows])
        style = palette['slices'][str(value)]
        line, = cuts.plot(sx, sy, color=style['color'], linestyle=style['linestyle'], lw=1.6)
        if not np.array_equal(line.get_ydata(), sy):
            raise ValueError('Slice data changed')
        cuts.annotate(f'$L={value}$', (sx[-1], sy[-1]), xytext=(5, 0), textcoords='offset points',
                      ha='left', va='center', fontsize=8, color=style['color'], annotation_clip=False)
    cuts.axhline(0, color='#A0A8AE', linestyle=':', lw=.7, zorder=0)
    cuts.axvline(25, color='#CAD0D4', linestyle=(0, (4, 2.4)), lw=.8, zorder=0)
    cuts.plot(17.5, 0, 'o', ms=4, mfc='white', mec=palette['slices']['10']['color'], mew=1.2)
    cuts.plot([17.5, 17.5], [-22, 0], color=palette['slices']['10']['color'], linestyle=':', lw=.75, zorder=0)
    cuts.set(xlim=(0, 50), ylim=(-22, 29), xticks=[0, 17.5, 25, 50], xticklabels=['0', '17.5', '25', '50'], yticks=[-20, -10, 0, 10, 20],
             xlabel='Audit probability, $q$ (%)', ylabel=r'$\Delta P$ (pp)')
    cuts.get_xticklabels()[1].set_color(palette['slices']['10']['color'])
    cuts.text(-.25, 1.03, '(b)', transform=cuts.transAxes, fontsize=10, fontweight='bold')
    fig.suptitle(r'$a=0.75\quad b=0.50\quad \theta_0=0\quad \eta=0.10$', fontsize=8.5)
    save(fig, 'audit_map', {'grid_shape': list(z.shape), 'grid_cells': int(z.size), 'slices': 3, 'slice_points_each': 501})

    # A genuinely refined grid near the boundary, preserving the same color scale.
    fig, ax = plt.subplots(figsize=np.array(brief['audit_detail']['figsize_mm']) / 25.4, layout='constrained')
    fig.get_layout_engine().set(w_pad=.05, h_pad=.055)
    mesh = ax.pcolormesh(Ld, qd, zd, shading='nearest', cmap=cmap, norm=norm, rasterized=False,
                        edgecolors='face', linewidth=.35, antialiased=False)
    keep = (bx >= 6) & (bx <= 14)
    ax.plot(bx[keep], by[keep], color='#263640', lw=1.5)
    contour = ax.contour(Ld, qd, zd, levels=[-5, -2, 2, 5], colors='#7E898D', linewidths=.55)
    ax.clabel(contour, inline=True, fontsize=7, fmt=lambda x: f'{x:+g} pp')
    ax.plot([6, 10, 10], [17.5, 17.5, 10], linestyle=':', color='#77878D', lw=.8)
    ax.plot(10, 17.5, 'o', ms=5, mfc='white', mec='#263640', mew=1.3)
    ax.annotate(r'$L=10,\ q=17.5\%$', (10, 17.5), xytext=(6.45, 19.4), fontsize=8.2,
                arrowprops={'arrowstyle': '-', 'lw': .7, 'color': '#4F6068'}, ha='left')
    ax.text(6.4, 23.3, 'Improvement', fontsize=9, color='#294B4F')
    ax.text(12.4, 10.9, 'Harm', fontsize=9, color='#6B4F35')
    ax.set(xlim=(6, 14), ylim=(10, 25), xticks=[6, 8, 10, 12, 14], yticks=[10, 15, 20, 25],
           xlabel='Input magnitude, $L$ (1)', ylabel='High-group audit probability, $q$ (%)')
    cb = fig.colorbar(mesh, ax=ax, pad=.035, fraction=.05, ticks=[-20, 0, 20])
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face'); cb.solids.set_linewidth(.35)
    cb.set_label(r'$\Delta P$ (pp)', fontsize=9); cb.outline.set_visible(False)
    save(fig, 'audit_detail', {'grid_shape': list(zd.shape), 'grid_cells': int(zd.size), 'contours_pp': [-5, -2, 2, 5]})

    # True two-input response surface; no perspective is used to create evidence.
    fig = plt.figure(figsize=np.array(brief['audit_surface']['figsize_mm']) / 25.4)
    ax = fig.add_axes([.035, .055, .79, .875], projection='3d', computed_zorder=False)
    xx, yy = np.meshgrid(L, q)
    surf = ax.plot_surface(xx, yy, z, rcount=len(q), ccount=len(L), cmap=cmap, norm=norm,
                           shade=False, linewidth=0, antialiased=False, rasterized=False, zorder=1)
    surf.set_edgecolor('face'); surf.set_linewidth(.35)
    ax.plot(bx, by, np.zeros_like(bx), color='#172F37', lw=1.5, zorder=10)
    ux = np.array([float(r['L']) for r in uniform]); uz = np.array([float(r['delta_P_pp']) for r in uniform])
    ax.plot(ux, np.full_like(ux, 25), uz, color='#172F37', lw=1.2, linestyle=(0, (4, 2.5)), zorder=10)
    ax.set(xlim=(1, 20), ylim=(0, 50), zlim=(vmin, vmax), xticks=[1, 10, 20], yticks=[0, 25, 50], zticks=[-20, 0, 20])
    ax.set_xlabel('Input magnitude, $L$ (1)', labelpad=6, fontsize=9)
    ax.set_ylabel('High-group audit, $q$ (%)', labelpad=7, fontsize=9)
    ax.set_zlabel(r'$\Delta P$ (pp)', labelpad=3, fontsize=9)
    ax.view_init(elev=25, azim=-130)
    ax.set_box_aspect((1.25, 1.05, .72))
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.set_pane_color((.985, .988, .99, 1))
        axis._axinfo['grid'].update(color='#D6DDE0', linewidth=.45)
    cbax = fig.add_axes([.855, .235, .024, .53])
    cb = fig.colorbar(surf, cax=cbax, ticks=[-20, -10, 0, 10, 20]); cb.outline.set_visible(False)
    cb.solids.set_rasterized(False)
    cb.solids.set_edgecolor('face'); cb.solids.set_linewidth(.35)
    cb.set_label('Accuracy change (pp)', labelpad=7, fontsize=9)
    fig.legend([Line2D([0], [0], color='#172F37', lw=1.5), Line2D([0], [0], color='#172F37', lw=1.2, linestyle='--')],
               [r'Initial boundary: $\Delta P=0$', r'Uniform guarantee: $q=25\%$'], loc='upper left',
               bbox_to_anchor=(.09, 1), frameon=False, fontsize=8, handlelength=2.6)
    save(fig, 'audit_surface', {'grid_shape': list(z.shape), 'mesh_vertices': int(z.size),
                              'mesh_faces': int((len(q)-1)*(len(L)-1)), 'decimation': False, 'shading': False,
                              'boundary_render_order': 'Exact boundary coordinates drawn after the surface to prevent painter occlusion; no coordinate offset.'}, inspect_2d=False)
    (args.out_dir / 'render_checks.json').write_text(json.dumps({'status': 'rendered', 'figures': records,
        'data_hashes': report['data_hashes'], 'matplotlib_version': matplotlib.__version__}, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'status': 'rendered', 'figures': list(records), 'out': str(args.out_dir)}, indent=2))


if __name__ == '__main__':
    main()
