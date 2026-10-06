"""Independently verify the approved round-eight and population readouts."""
import argparse
import csv
import hashlib
import json
import math
import statistics as st
from pathlib import Path

PAPER = Path(__file__).resolve().parent
TCRIT = 2.7764451051977987


def population(theta, rule):
    a, b, L = .75, .5, 10.
    p1, pL = 1 / (1 + math.exp(-theta)), 1 / (1 + math.exp(-L * theta))
    P = (p1 + pL) / 2
    Z = a * P + b * (1 - P)
    m1, mL = (1 - p1) / 2, (1 - pL) / 2
    Q = b * (m1 + mL)
    r1, rL = ((b, b) if rule == 'R' else (max(0, Q - mL) / m1, min(1, Q / mL)))
    g = (.5 * p1 * (1 - p1) * (r1 - a) + .5 * L * pL * (1 - pL) * (rL - a)) / Z
    return dict(theta=theta, P=P, Z=Z, precision=a * P / Z, gradient=g)


def integrate(rule, dt):
    theta = 0.
    f = lambda x: -population(x, rule)['gradient']
    for _ in range(round(20 / dt)):
        k1 = f(theta)
        k2 = f(theta + dt * k1 / 2)
        k3 = f(theta + dt * k2 / 2)
        k4 = f(theta + dt * k3)
        theta += dt * (k1 + 2 * k2 + 2 * k3 + k4) / 6
    return population(theta, rule)


def check(code):
    root = code / 'experiments/outputs/2026-10-03-h100'
    hashes, result = {}, []
    def read(path):
        hashes[str(path.relative_to(code))] = hashlib.sha256(path.read_bytes()).hexdigest()
        return json.loads(path.read_text())
    for model, suffix in [('Llama-3.2-3B', '_llama3.2-3b'), ('Qwen3-1.7B', '')]:
        for policy in ['unaudited', 'audited']:
            row = dict(model=model, policy=policy)
            for split, n in [('eval_id', 2000), ('eval_ood', 1000)]:
                values = []
                for b in range(5):
                    records = [read(root / f'graph_{policy}{suffix}/out/evaluation_greedy_{split}/b{b:02d}_{arm}_round_008.json') for arm in ['R', 'S']]
                    for record, arm in zip(records, ['R', 'S']):
                        assert record['block'] == f'b{b:02d}' and record['arm'] == arm
                        assert record['round'] == 8 and record['questions'] == n and record['split'] == split
                        assert abs(record['pass1'] - (1 - sum(record['error_counts'].values()) / n)) < 1e-12
                    values.append((sum(records[0]['error_counts'].values()) - sum(records[1]['error_counts'].values())) / n)
                mean = st.mean(values)
                half = TCRIT * st.stdev(values) / math.sqrt(5)
                csv_path = root / f'figures/graph_unaudited{suffix}_vs_graph_audited{suffix}_{split}_summary.csv'
                hashes[str(csv_path.relative_to(code))] = hashlib.sha256(csv_path.read_bytes()).hexdigest()
                matches = [r for r in csv.DictReader(csv_path.open()) if r['run'] == ('unaudited' if policy == 'unaudited' else 'audited B=16') and r['metric'] == 'pass1' and r['difficulty'] == 'all' and r['series'] == 'S-R' and r['round'] == '8']
                assert len(matches) == 1
                supplied = matches[0]
                assert int(supplied['n']) == 5
                assert abs(mean - float(supplied['mean'])) < 1e-12
                assert abs(half - float(supplied['ci95_half_width'])) < 1e-12
                row[split] = dict(block_differences=values, mean=mean, half_width=half, low=mean-half, high=mean+half)
            result.append(row)
    lo, hi = -.12, -.10
    assert population(lo, 'S')['gradient'] < 0 < population(hi, 'S')['gradient']
    for _ in range(60):
        mid = (lo + hi) / 2
        if population(mid, 'S')['gradient'] < 0:
            lo = mid
        else:
            hi = mid
    s_limit = population((lo + hi) / 2, 'S')
    r20, refined = integrate('R', .001), integrate('R', .0005)
    assert max(abs(r20[k] - refined[k]) for k in r20) < 1e-10
    saved = list(csv.DictReader((PAPER / 'figures/population_flow/trajectories.csv').open()))
    saved_r = next(q for q in saved if q['rule'] == 'R' and float(q['t']) == 20.)
    # Existing plot uses Euler integration; all quoted three-digit readouts agree.
    for key, saved_key in [('Z', 'acceptance_rate'), ('precision', 'purity')]:
        assert f'{r20[key]:.3f}' == f'{float(saved_r[saved_key]):.3f}'
    report = dict(status='passed', round_eight=result, source_sha256=hashes,
                  population=dict(parameters=dict(a=.75,b=.5,L=10,theta0=0), initial=population(0., 'R'), S_limit=s_limit, R_t20=r20,
                                  R_refinement_max_difference=max(abs(r20[k]-refined[k]) for k in r20)))
    table_path = PAPER / 'eight_round_contrasts_table.tex'
    if table_path.exists():
        rows = [line for line in table_path.read_text().splitlines() if '&' in line and ('Unaudited' in line or '$B=16$' in line)]
        assert len(rows) == 4
        for row, values in zip(rows, result):
            for split in ['eval_id', 'eval_ood']:
                q = values[split]
                assert f'${q["mean"]:.3f}\\pm{q["half_width"]:.3f}$' in row
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--record', action='store_true')
    args = parser.parse_args()
    report = check(args.code)
    if args.record:
        (PAPER / 'approved_guide_results.json').write_text(json.dumps(report, indent=2, sort_keys=True) + '\n')
    print(json.dumps({k:v for k,v in report.items() if k != 'source_sha256'}, indent=2))
