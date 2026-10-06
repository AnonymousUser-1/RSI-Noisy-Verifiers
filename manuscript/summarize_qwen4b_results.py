#!/usr/bin/env python3
"""Validate the available four-round Qwen3-4B exports and render vector charts.

Run with --code ... Reconstruct both policies from checkpoint counts,
audit/training records, and the available audited held-out responses.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics
import re
import sys
from collections import Counter

import summarize_graph_extensions as ext
import summarize_h100_results as theme
import optional_inputs
from pypdf import PdfReader, PdfWriter

PAPER = Path(__file__).resolve().parent
BLOCKS = [f'b{i:02d}' for i in range(5)]
ARMS = ['R', 'S']
TCRIT = 2.7764451051977987


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code', type=Path, required=True)
    optional_inputs.add_argument(ap)
    args = ap.parse_args()
    code = args.code.resolve()
    ext.CODE = code
    roots = {
        'none': code / 'experiments/outputs/2026-10-05-qwen3-4b-multiround-unaudited',
        'audit': code / 'experiments/outputs/2026-10-05-qwen3-4b-graph-audit-b16',
    }
    source_hashes = {}
    def read(p):
        raw = p.read_bytes()
        source_hashes[str(p.relative_to(code))] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)
    def same(a, b):
        assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (a, b)
    summaries, native = {}, {}
    rows_checked = 0
    for policy, root in roots.items():
        native[policy] = {}
        for split in ['eval_id', 'eval_ood']:
            p = next((root / 'figures').glob('*' + split + '_summary.json'))
            d = read(p)
            labels = list(d['pass1'])
            assert len(labels) == 1
            label = labels[0]
            native[policy][split] = d
            summary = {}
            for metric in ['pass1', 'target_error_rate']:
                q = d[metric][label]
                summary[metric] = {}
                for arm in ARMS + ['S-R']:
                    saved = q['s_minus_r'] if arm == 'S-R' else q['arms'][arm]
                    assert sorted(saved) == ['0', '1', '2', '3', '4']
                    summary[metric][arm] = {}
                    for t, v in saved.items():
                        assert v['blocks'] == BLOCKS and v['n'] == 5
                        assert 0 <= v['sd'] and math.isfinite(v['mean'])
                        if arm != 'S-R':
                            assert 0 <= v['mean'] <= 1
                        same(v['half_width'], TCRIT * v['sd'] / math.sqrt(5))
                        if t == '0':
                            same(v['sd'], 0)
                        summary[metric][arm][t] = dict(v, low=v['mean']-v['half_width'], high=v['mean']+v['half_width'])
                for t in map(str, range(5)):
                    r, s, diff = [summary[metric][a][t] for a in ['R', 'S', 'S-R']]
                    same(diff['mean'], s['mean'] - r['mean'])
                    assert abs(s['sd'] - r['sd']) <= diff['sd'] + 1e-12
                    assert diff['sd'] <= s['sd'] + r['sd'] + 1e-12
            for a in ARMS:
                for t in map(str, range(5)):
                    assert summary['target_error_rate'][a][t]['mean'] <= 1-summary['pass1'][a][t]['mean']+1e-12
            summaries.setdefault(split, {})[policy] = summary
            csv_path = p.with_suffix('.csv')
            raw = csv_path.read_bytes()
            source_hashes[str(csv_path.relative_to(code))] = hashlib.sha256(raw).hexdigest()
            rows = list(csv.DictReader(raw.decode().splitlines()))
            assert len(rows) == 75
            assert len({(r['metric'],r['difficulty'],r['series'],r['round']) for r in rows}) == 75
            for row in rows:
                assert row['run'] == label
                group = d[row['metric']][label] if row['difficulty'] == 'all' else d['by_difficulty'][label][row['difficulty']]
                val = group['s_minus_r'][row['round']] if row['series'] == 'S-R' else group['arms'][row['series']][row['round']]
                for col, key in [('mean','mean'),('sd','sd'),('n','n'),('ci95_half_width','half_width')]:
                    same(float(row[col]), val[key])
                if row['paired_t_p']:
                    same(float(row['paired_t_p']), val['paired_t_p'])
                same(val['half_width'], TCRIT * val['sd']/math.sqrt(5))
                rows_checked += 1
    # Original audited image and summary hashes are independently checked.
    audit_root = roots['audit']
    for line in (audit_root / 'SHA256SUMS').read_text().splitlines():
        if not line.strip():
            continue
        expected, rel = line.split(maxsplit=1)
        asset = audit_root / rel.lstrip('*')
        raw = asset.read_bytes()
        assert hashlib.sha256(raw).hexdigest() == expected, rel
        source_hashes[str(asset.relative_to(code))] = expected
    source_hashes[str((audit_root/'README.md').relative_to(code))] = hashlib.sha256((audit_root/'README.md').read_bytes()).hexdigest()
    # The unaudited export has per-block counts and completion records.
    run = roots['none'] / 'graph_unaudited_qwen3-4b'
    data, protocols = ext.evaluations(run, 'out', 4)
    recomputed = ext.summarize(data, 4)
    supplied = {s:{m:native['none'][s][m]['unaudited'] for m in ['pass1','target_error_rate']} for s in data}
    ext.compare_summary(recomputed, supplied, 4)
    for split, n in [('eval_id',2000),('eval_ood',1000)]:
        assert re.fullmatch(r'[0-9a-f]{64}', protocols[split]['question_ids'])
        for b, t in itertools.product(BLOCKS, range(1,5)):
            r, s = [data[split][f'{b}_{a}_{t}'] for a in ARMS]
            same(s['pass1']-r['pass1'], (sum(r['error_counts'].values())-sum(s['error_counts'].values()))/n)
    matching = read(run / 'out/matching/matched_subsets.json')
    ext.matchcheck(matching)
    steps = {}
    for b, a in itertools.product(BLOCKS, ARMS):
        steps[b+'_'+a] = 0
        for t in range(1,5):
            q = read(run / f'out/{b}/{a}/round_{t:03d}/complete.json')
            assert (q['block_id'], q['branch'], q['round']) == (b,a,t)
            assert q['training']['steps'] == q['expected_optimizer_steps'] == 16
            assert q['examples'] == 64 and q['audit_queries'] == 0
            steps[b+'_'+a] += q['training']['steps']
        assert steps[b+'_'+a] == 64
    audited, audit_protocols = ext.evaluations(audit_root, 'out', 4)
    ext.compare_summary(ext.summarize(audited, 4),
        {s:{m:next(iter(native['audit'][s][m].values())) for m in ['pass1','target_error_rate']} for s in audited}, 4)
    audit_matching = read(audit_root/'out/matching/matched_subsets.json')
    ext.matchcheck(audit_matching)
    assert audit_matching == matching
    manifests = {p:read(r/'out/experiment.json') for p,r in [('none',run),('audit',audit_root)]}
    for q in manifests.values():
        assert q['status'] == 'complete' and q['exit_code'] == 0 and not q['DEMO_ONLY']
        assert len(q['completed']) == 40
    for split in protocols:
        for k in ['question_ids','dataset_hash','base','decoding','questions']:
            assert protocols[split][k] == audit_protocols[split][k], (split,k)
    audit_steps, training = {}, {}
    for b,a in itertools.product(BLOCKS, ARMS):
        key = b+'_'+a
        audit_steps[key] = 0
        training[key] = {}
        for t in range(1,5):
            folder = audit_root/f'out/{b}/{a}/round_{t:03d}'
            q = read(folder/'complete.json')
            expected_profile = 'legacy' if b == 'b00' or (b == 'b01' and t < 4) else 'new'
            assert q['protocol_profile'] == expected_profile
            au = read(folder/'audit.json')
            pre = ext.jsonl(folder/'pre_audit.jsonl')
            retained = ext.jsonl(folder/'training.jsonl')
            assert (q['block_id'],q['branch'],q['round']) == (b,a,t)
            assert q['audit_queries'] == au['queries'] == len(au['labels']) == 16
            assert q['cumulative_audit_queries'] == 16*t
            assert len(pre) == q['selected_examples'] == au['before']['K'] == 64
            assert len({x['id'] for x in pre}) == 64
            pre_ids = {x['id'] for x in pre}
            assert {x['id'] for x in retained} <= pre_ids
            assert len({x['id'] for x in retained}) == len(retained)
            labels = {x['candidate_id']:x for x in au['labels']}
            assert len(labels) == 16 and labels.keys() <= pre_ids
            removed = {i for i,x in labels.items() if not x['correct']}
            assert pre_ids-{x['id'] for x in retained} == removed
            comp = au['after']
            assert comp['C'] == 48 and comp['E'] == 16-len(removed)
            assert len(retained) == q['examples'] == comp['K'] == comp['C']+comp['E']
            weights = [x['weight'] for x in retained]
            assert all(math.isfinite(w) and w >= 0 for w in weights)
            positive = sum(w > 0 for w in weights)
            assert positive == comp['positive_weight_examples'] == q['training']['examples']
            same(sum(weights),comp['weight_sum'])
            same(sum(weights)**2/sum(w*w for w in weights),comp['effective_sample_size'])
            same(comp['final_TPR'],48/comp['N_plus'])
            same(comp['positive_weight_TPR'],comp['positive_weight_correct']/comp['N_plus'])
            assert q['training']['steps'] == q['expected_optimizer_steps'] == math.ceil(positive/4)
            audit_steps[key] += q['training']['steps']
            training[key][str(t)] = dict(completion=q,audit=au)
    contrasts = {s:{a:dict(ext.stats(vals),blocks=BLOCKS,values=vals)
        for a in ARMS for vals in [[(sum(audited[s][f'{b}_{a}_4']['error_counts'].values())-
                                    sum(data[s][f'{b}_{a}_4']['error_counts'].values()))/n for b in BLOCKS]]}
        for s,n in [('eval_id',2000),('eval_ood',1000)]}
    # Independently rejudge every uploaded audited answer against the paired task set.
    sys.path.insert(0,str(code))
    from rsi.tasks import judge
    from rsi.common import digest as value_digest
    for source in [code/'rsi/tasks.py',code/'rsi/common.py']:
        source_hashes[str(source.relative_to(code))] = hashlib.sha256(source.read_bytes()).hexdigest()
    rejudged = 0
    for split,n in [('eval_id',2000),('eval_ood',1000)]:
        task_rel = f'RSI-Qwen3-4B-Graph-paired-inputs-20261004/graph/{split}.jsonl'
        task_path = optional_inputs.resolve(code, task_rel, args.inputs)
        raw = task_path.read_bytes()
        source_hashes[optional_inputs.source_key(code, task_path, args.inputs)] = hashlib.sha256(raw).hexdigest()
        tasks = {q['id']:q for q in (json.loads(line) for line in raw.decode('utf-8').splitlines() if line.strip())}
        assert len(tasks) == n and value_digest(sorted(tasks)) == audit_protocols[split]['question_ids']
        for key,metrics in audited[split].items():
            filename = 'round_000' if key == 'base' else key.rsplit('_',1)[0]+'_round_'+f'{int(key.rsplit("_",1)[1]):03d}'
            answers = ext.jsonl(audit_root/f'out/evaluation_greedy_{split}/{filename}.jsonl')
            assert len(answers) == n and {q['task_id'] for q in answers} == tasks.keys()
            counts = Counter()
            for answer in answers:
                task = tasks[answer['task_id']]
                result = judge(task,answer['response'])
                assert result['correct'] == answer['correct'] and result['error'] == answer['error']
                assert answer['sample'] == 0 and answer['difficulty'] == task['difficulty']
                if not result['correct']:
                    counts[result['error']] += 1
            assert dict(counts) == metrics['error_counts']
            assert sum(q['truncated'] for q in answers) == metrics['truncated_answers']
            same(sum(q['correct'] for q in answers)/n,metrics['pass1'])
            rejudged += n
    source_hashes.update(ext.READ)
    out = PAPER / 'figures/qwen4b_multiround'
    out.mkdir(parents=True, exist_ok=True)
    theme.TCRIT = TCRIT
    theme.chart(out/'qwen4b_accuracy.pdf', summaries, [
        ('eval_id','pass1',(0,1),(0,.25,.5,.75,1),'(a) ID: 2,000 questions','Greedy Pass@1'),
        ('eval_ood','pass1',(0,1),(0,.25,.5,.75,1),'(b) OOD: 1,000 questions','Greedy Pass@1')])
    theme.chart(out/'qwen4b_target_errors.pdf', summaries, [
        ('eval_id','target_error_rate',(0,.5),(0,.1,.2,.3,.4,.5),'(a) ID target errors','Fraction of all answers'),
        ('eval_ood','target_error_rate',(0,.5),(0,.1,.2,.3,.4,.5),'(b) OOD target errors','Fraction of all answers')])
    for p in out.glob('*.pdf'):
        writer = PdfWriter(clone_from=PdfReader(p))
        writer.add_metadata({'/Title':'Qwen3-4B graph: four-round greedy evaluation'})
        with p.open('wb') as f:
            writer.write(f)
    report = dict(status='passed',summary=summaries,unaudited_evaluations=data,
        unaudited_protocols=protocols,unaudited_matching=matching,
        unaudited_steps=steps,csv_rows_checked=rows_checked,
        audited_evaluations=audited,audited_protocols=audit_protocols,
        audited_matching=audit_matching,audited_training=training,
        audited_steps=audit_steps,audit_error_change=contrasts,
        independent_audited_checkpoints=82,audited_responses_rejudged=rejudged,
        audited_training_completions_checked=40,
        independent_unaudited_checkpoints=82,audited_source_hashes_checked=8,
        source_files=source_hashes,
        limitations=['Baseline predictions differ across policies; cross-run paired changes are descriptive, not clean causal audit effects.',
                    'Unaudited raw answers are omitted; its evaluations are checked from counts only.',
                    'Audited adapters are available but not reloaded; shared initialisation and reference tensors are omitted.'],
        chart_contract={'surface':'LaTeX/vector PDF','question':'Compare the four recorded rounds of R/S accuracy within each policy',
                        'family':'Discrete checkpoint line with pointwise intervals',
                        'palette':{'R':'#127F83','S':'#805677'},
                        'non_color':'Solid/dashed R/S, filled/open policy markers',
                        'footprint_points':[486,184],'horizon':[0,1,2,3,4],
                        'baseline':'Distinct policy baselines; no independent replication at round zero'})
    (out/'validated_results.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print(json.dumps({k:report[k] for k in ['status','csv_rows_checked','independent_unaudited_checkpoints','independent_audited_checkpoints','audited_responses_rejudged','audited_training_completions_checked','audited_steps','audit_error_change','limitations']},indent=2))


if __name__ == '__main__':
    main()
