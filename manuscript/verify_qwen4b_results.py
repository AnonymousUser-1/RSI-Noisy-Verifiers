#!/usr/bin/env python3
"""Check displayed Qwen3-4B cells against the checked native export values."""
import argparse
import hashlib
import json
from pathlib import Path
import re
import math
import statistics
from decimal import Decimal, ROUND_HALF_UP
import optional_inputs

PAPER = Path(__file__).resolve().parent

def rows(name):
    text = (PAPER/name).read_text()
    body = text.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    return [[c.strip() for c in line[:-2].split('&')] for line in body.splitlines()
            if '&' in line and line.endswith('\\\\')]

def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code', type=Path, required=True)
    optional_inputs.add_argument(ap)
    args = ap.parse_args()
    d = json.loads((PAPER/'figures/qwen4b_multiround/validated_results.json').read_text())
    s = d['summary']
    # Recompute cross-policy error differences independently from integer counts.
    for split,n in [('eval_id',2000),('eval_ood',1000)]:
        for arm in ['R','S']:
            vals = [(sum(d['audited_evaluations'][split][f'b{i:02d}_{arm}_4']['error_counts'].values())-
                     sum(d['unaudited_evaluations'][split][f'b{i:02d}_{arm}_4']['error_counts'].values()))/n for i in range(5)]
            q = d['audit_error_change'][split][arm]
            m = statistics.mean(vals)
            half = 2.7764451051977987*statistics.stdev(vals)/math.sqrt(5)
            for k,v in [('mean',m),('half_width',half),('low',m-half),('high',m+half)]:
                assert math.isclose(q[k],v,abs_tol=1e-12)
    count = 0
    def error_display(split, policy, arm, t):
        ev = d['unaudited_evaluations' if policy == 'none' else 'audited_evaluations'][split]
        values = [ev['base']]*5 if t == 0 else [ev[f'b{i:02d}_{arm}_{t}'] for i in range(5)]
        n = 2000 if split == 'eval_id' else 1000
        fraction = Decimal(sum(sum(q['error_counts'].values()) for q in values)) / Decimal(5*n)
        return str(fraction.quantize(Decimal('.001'), rounding=ROUND_HALF_UP))
    def check(text, expected):
        nonlocal count
        assert text == expected, (text, expected)
        count += 1
    rr = rows('h100_endpoint_table.tex')[5:]
    assert len(rr) == 6
    for i, policy in enumerate(['none','audit']):
        check(rr[i][0], 'Base, unaudited' if policy == 'none' else 'Base, audited')
        for col,split in [(4,'eval_id'),(5,'eval_ood')]:
            check(rr[i][col], error_display(split, policy, 'R', 0))
    for r,(policy,arm) in zip(rr[2:],[(p,a) for p in ['none','audit'] for a in ['R','S']]):
        check(r[0], 'Unaudited' if policy == 'none' else 'Adaptive, B=16')
        check(r[1], arm)
        check(r[2], '0' if policy == 'none' else '64')
        steps=[d['audited_steps'][f'b{i:02d}_{arm}'] for i in range(5)]
        check(r[3], '64' if policy == 'none' else f'{statistics.mean(steps):.1f} ({min(steps)}--{max(steps)})')
        for col,split in [(6,'eval_id'),(7,'eval_ood')]:
            q=d['audit_error_change'][split][arm]
            expected = '$'+f'{q["mean"]:.3f}'+r'\;['+f'{q["low"]:.3f},{q["high"]:.3f}'+']$'
            check(r[col], '---' if policy == 'none' else expected)
        for col,split in [(4,'eval_id'),(5,'eval_ood')]:
            check(r[col], error_display(split, policy, arm, 4))
    rr = rows('qwen4b_multiround_endpoint_table.tex')
    assert len(rr) == 4
    for r,(split,policy) in zip(rr,[(sp,p) for sp in ['eval_id','eval_ood'] for p in ['none','audit']]):
        check(r[2], f"{s[split][policy]['pass1']['R']['0']['mean']:.4f}")
        for col,metric in [(3,'pass1'),(5,'target_error_rate')]:
            q = s[split][policy][metric]
            check(r[col], '/'.join(f"{q[a]['4']['mean']:.4f}" for a in ['R','S']))
            v=q['S-R']['4']
            check(r[col+1], '$'+f"{v['mean']:.4f}"+r'\pm'+f"{v['half_width']:.4f}"+'$')
    for rel, expected in d['source_files'].items():
        source = optional_inputs.resolve(args.code, rel, args.inputs)
        assert hashlib.sha256(source.read_bytes()).hexdigest() == expected, rel
    # Inspect the discrete checkpoint scale for unintentional clipping.
    for split in s:
        for policy in s[split]:
            for metric, upper in [('pass1',1),('target_error_rate',.5)]:
                for a in ['R','S']:
                    for t,q in s[split][policy][metric][a].items():
                        assert 0 <= q['low'] <= q['high'] <= upper, (split,policy,metric,a,t)
    print(json.dumps(dict(status='passed',display_cells_checked=count,
        csv_rows_checked=d['csv_rows_checked'],source_hashes_verified=len(d['source_files']),
        audited_checkpoint_records=d['independent_audited_checkpoints'],
        audited_responses_rejudged=d['audited_responses_rejudged'],
        scope='Displayed endpoint cells, independently recomputed paired audit changes, audit effort, source hashes'),indent=2))

if __name__ == '__main__':
    main()
