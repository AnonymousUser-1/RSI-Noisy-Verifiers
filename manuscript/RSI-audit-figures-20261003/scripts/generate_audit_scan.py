"""Deterministic audited initial-update scan, checked by finite-outcome enumeration.

The plotted response is 100*(P(theta0-eta*g_S,q(theta0))-P(theta0)):
percentage POINTS, not relative percent and not a language-model measurement.
q is the query probability in the accepted high-magnitude group. The global
expected initial query fraction is separately recorded as B(q).
"""
from pathlib import Path
import argparse
import csv
import hashlib
import importlib.util
import json
import math
import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out-dir', type=Path, required=True)
    args = cli.parse_args()
    if args.out_dir.exists():
        cli.error('Use a new output directory; existing data are preserved.')
    settings = json.loads((ROOT / 'scan_parameters.json').read_text())
    source = ROOT / 'source/theory_checks.py'
    spec = importlib.util.spec_from_file_location('theory_reference', source)
    theory = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(theory)
    a, b, eta = (settings[k] for k in ('a', 'b', 'eta'))
    if not (0 < b < a < 1 and eta > 0 and settings['theta0'] == 0):
        raise ValueError('This frozen scan requires 0<b<a<1, eta>0, theta0=0.')
    q_uniform = 1 - a
    tol = settings['oracle_absolute_tolerance']
    errors = {'gradient': 0., 'delta_P': 0., 'query_fraction': 0.}
    checked = 0

    def point(L, q):
        nonlocal checked
        # Eq. (26), with the POST-audit normalizer.
        g = (b * L * (1 - a - q) - a * (1 - b)) / (2 * (a + b * (1 - q)))
        theta1 = -eta * g
        p1 = (1 - b) * theory.sigmoid(theta1)[0] + b * theory.sigmoid(L * theta1)[0]
        delta = p1 - .5
        budget = q * b * (a + 1) / (a + b)
        initial = theory.enumeration(0, a, b, L, 'S', q)
        after = theory.enumeration(-eta * initial['gradient'], a, b, L, 'S')['accuracy']
        for name, err in [('gradient', abs(g - initial['gradient'])),
                          ('delta_P', abs(delta - (after - .5))),
                          ('query_fraction', abs(budget - initial['query_fraction']))]:
            errors[name] = max(errors[name], err)
            if not math.isfinite(err) or err > tol:
                raise ValueError(f'Independent oracle mismatch: {name}, L={L}, q={q}, error={err}')
        if not (0 <= p1 <= 1 and 0 <= budget <= 1):
            raise ValueError('Invalid probability')
        raw_threshold = 1 - a - a * (1 - b) / (b * L)
        if abs(q - raw_threshold) > 1e-10 and delta * (q - raw_threshold) <= 0:
            raise ValueError('Initial sign disagrees with analytical threshold')
        if q >= q_uniform and delta <= 0:
            raise ValueError('Violation of sufficient audit threshold')
        checked += 1
        return {'a': a, 'b': b, 'L': float(L), 'q': float(q), 'theta0': 0., 'eta': eta,
                'gradient_S_q': g, 'theta1': theta1, 'P_before': .5, 'P_after': p1,
                'delta_P': delta, 'delta_P_pp': 100 * delta, 'initial_query_fraction': budget,
                'q_critical_raw': raw_threshold,
                'initial_effect': 'zero' if abs(delta) <= settings['zero_tolerance'] else ('improvement' if delta > 0 else 'harm')}

    datasets = {}
    for name in ('main', 'detail'):
        domain = settings[name]
        Ls = np.linspace(domain['L_min'], domain['L_max'], domain['L_count'])
        qs = np.linspace(domain['q_min'], domain['q_max'], domain['q_count'])
        datasets[name + '_grid.csv'] = [point(float(L), float(q)) for q in qs for L in Ls]
    qs = np.linspace(settings['main']['q_min'], settings['main']['q_max'], settings['slice_q_count'])
    datasets['parameter_slices.csv'] = [point(L, float(q)) for L in settings['slices_L'] for q in qs]
    threshold_L = a * (1 - b) / (b * (1 - a))
    Ls = np.linspace(max(1., threshold_L), settings['main']['L_max'], settings['boundary_count'])
    boundary = []
    boundary_checks = 0
    for L in Ls:
        q = 1 - a - a * (1 - b) / (b * L)
        row = point(float(L), float(q))
        if abs(row['delta_P']) > 2e-13:
            raise ValueError('Critical contour is not zero')
        for offset in (-1e-5, 1e-5):
            if 0 <= q + offset <= 1:
                test = point(float(L), float(q + offset))
                if test['delta_P'] * offset <= 0:
                    raise ValueError('Threshold-side test failed')
                boundary_checks += 1
        boundary.append(row)
    datasets['critical_boundary.csv'] = boundary
    datasets['uniform_threshold.csv'] = [point(float(L), q_uniform) for L in np.linspace(1, 20, 121)]
    datasets['selected_readouts.csv'] = [point(10., q) for q in (0., .1, .175, .2, .25)]

    args.out_dir.mkdir(parents=True)
    for name, rows in datasets.items():
        with (args.out_dir / name).open('x', encoding='utf-8', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    z = [r['delta_P_pp'] for r in datasets['main_grid.csv']]
    report = {'status': 'passed', 'parameters': settings, 'dataset_rows': {k: len(v) for k, v in datasets.items()},
              'independently_checked_parameter_points': checked, 'threshold_side_checks': boundary_checks,
              'max_oracle_error': errors, 'main_delta_P_pp_range': [min(z), max(z)],
              'uniform_sufficient_q': q_uniform, 'uniform_initial_global_query_fraction': q_uniform * b * (a + 1) / (a + b),
              'L10_initial_q_critical': .175, 'reference_checker_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
              'generator_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
              'parameters_sha256': hashlib.sha256((ROOT / 'scan_parameters.json').read_bytes()).hexdigest(),
              'data_hashes': {n: hashlib.sha256((args.out_dir / n).read_bytes()).hexdigest() for n in datasets},
              'scope': 'Deterministic initial mathematical step. No Monte Carlo, GPU measurements, confidence intervals, or finite-time claim of convergence to accuracy one.'}
    (args.out_dir / 'scan_validation.json').write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ('status', 'dataset_rows', 'independently_checked_parameter_points', 'max_oracle_error', 'main_delta_P_pp_range')}, indent=2))


if __name__ == '__main__':
    main()
