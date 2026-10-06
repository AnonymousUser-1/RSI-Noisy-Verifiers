#!/usr/bin/env python3
"""Validate compact graph exports and regenerate the manuscript additions.

python summarize_graph_extensions.py --code .. --out figures/graph_extensions
No training, network access, or model weights are required. Numerical checks
use the standard library; figures use ReportLab and pypdf. Omitted adapters and
held-out answers preclude adapter reloads and independent held-out rejudging.
"""
import argparse
import csv
import hashlib
import itertools
import json
import math
from pathlib import Path
import statistics as st
import sys
from table_typography import numeric_math

from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from pypdf import PdfReader, PdfWriter
import summarize_h100_results as old

BLOCKS = [f'b{i:02d}' for i in range(5)]
ARMS = ['R', 'S']
MODELS = [('graph_unaudited', 'Qwen3-1.7B'),
          ('graph_unaudited_qwen3-4b', 'Qwen3-4B'),
          ('graph_unaudited_llama3.2-3b', 'Llama-3.2-3B')]
TCRIT = 2.7764451051977987
READ = {}
TABLES = {}
CODE = None
OUT = None
PAPER = Path(__file__).resolve().parent


def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()


def read(p):
    raw = p.read_bytes()
    READ[str(p.relative_to(CODE))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def jsonl(p):
    raw = p.read_bytes()
    READ[str(p.relative_to(CODE))] = hashlib.sha256(raw).hexdigest()
    return [json.loads(s) for s in raw.decode().splitlines() if s.strip()]


def close(a, b):
    assert math.isclose(a, b, rel_tol=1e-9, abs_tol=1e-12), (a, b)


def stats(v):
    n = len(v)
    m = st.mean(v)
    sd = st.stdev(v) if n > 1 else None
    half = TCRIT * sd / math.sqrt(n) if n == 5 else None
    return dict(mean=m, sd=sd, n=n, half_width=half,
                low=m-half if half is not None else None,
                high=m+half if half is not None else None)


def cell(v, places=3):
    return f'{v:.{places}f}'


def interval(v, places=3):
    return '$' + cell(v['mean'], places) + r'\pm' + cell(v['half_width'], places) + '$'


def pair(v, places=3):
    return '/'.join(cell(x, places) for x in v)


def table(name, caption, label, headers, rows):
    rows = [[numeric_math(c) for c in row] for row in rows]
    # Exact rendered-cell contract is saved alongside the validated raw records.
    TABLES[name] = {'headers': headers, 'rows': rows}
    environment, placement = ('table*','!t') if name == 'one_step_heldout_table.tex' else ('table','H')
    lines = [r'\begin{'+environment+'}['+placement+']', r'\caption{' + caption + '}',
             r'\label{' + label + '}', r'\centering\small\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}'+'l'*len(headers)+r'@{}}', r'\toprule',
             ' & '.join(headers)+r'\\', r'\midrule']
    lines += [' & '.join(r)+r'\\' for r in rows]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{'+environment+'}', '']
    (PAPER/name).write_text('\n'.join(lines))


def matchcheck(m):
    assert sorted(m['per_block']) == BLOCKS and m['K'] == 64
    for b, q in m['per_block'].items():
        a = q['audit']
        assert all(a[k] for k in ('K_equal', 'final_TPR_equal', 'final_FPR_equal',
                   'same_correct_ids', 'same_task_ids', 'task_token_counts_within_tolerance', 'weights_all_one'))
        assert a['C_R'] == a['C_S'] == 48 and a['E_R'] == a['E_S'] == 16
        assert a['max_token_mismatch'] == 0 and a['S_target_hits'] == 16
        close(a['final_TPR'], 48/a['N_plus'])
        close(a['final_FPR'], 16/a['N_minus'])
        close(a['yield'], 64/(a['N_plus']+a['N_minus']))
        for arm in ARMS:
            assert len(q[arm]) == len(set(q[arm])) == 64


def evalcheck(q, n, t):
    assert q['questions'] == n and q['round'] == t and 0 <= q['pass1'] <= 1
    close(1-q['pass1'], sum(q['error_counts'].values())/n)
    close(q['target_error_rate'], q['error_counts'].get('nonshortest', 0)/n)
    assert sum(q['questions_by_difficulty'].values()) == n
    close(q['pass1'], sum(q['pass1_by_difficulty'][d]*k for d, k in q['questions_by_difficulty'].items())/n)


def evaluations(e, outname, rounds):
    data, protocols = {}, {}
    for split, n in [('eval_id', 2000), ('eval_ood', 1000)]:
        d = e/outname/('evaluation_greedy_'+split)
        pr = read(d/'protocol.json')
        assert pr['questions'] == n and pr['answers_per_question'] == 1
        assert pr['task'] == 'graph' and pr['decoding']['temperature'] == 0
        assert pr['decoding']['top_p'] == 1
        data[split] = {'base': read(d/'round_000.json')}
        evalcheck(data[split]['base'], n, 0)
        for b, arm, t in itertools.product(BLOCKS, ARMS, range(1, rounds+1)):
            q = read(d/f'{b}_{arm}_round_{t:03d}.json')
            assert q['block'] == b and q['arm'] == arm
            evalcheck(q, n, t)
            data[split][f'{b}_{arm}_{t}'] = q
        protocols[split] = pr
    return data, protocols


def summarize(data, rounds):
    result = {}
    for split, rows in data.items():
        result[split] = {}
        for metric in ['pass1', 'target_error_rate']:
            result[split][metric] = {}
            for arm in ['R', 'S', 'S-R']:
                result[split][metric][arm] = {'0': stats([0 if arm == 'S-R' else rows['base'][metric]])}
                for t in range(1, rounds+1):
                    vals = [rows[f'{b}_{arm}_{t}'][metric] if arm != 'S-R' else
                            rows[f'{b}_S_{t}'][metric]-rows[f'{b}_R_{t}'][metric] for b in BLOCKS]
                    result[split][metric][arm][str(t)] = stats(vals)
    return result


def compare_summary(computed, supplied, rounds):
    for split in computed:
        for metric in ['pass1', 'target_error_rate']:
            for arm in ['R', 'S', 'S-R']:
                saved = supplied[split][metric]
                for t in range(1, rounds+1):
                    q = saved['s_minus_r'][str(t)] if arm == 'S-R' else saved['arms'][arm][str(t)]
                    for k in ['mean', 'sd', 'half_width']:
                        close(computed[split][metric][arm][str(t)][k], q[k])


def multiround(root):
    data, summary, protocols, training, matching, manifests = {}, {}, {}, {}, {}, {}
    for policy, name in [('none', 'graph_unaudited'), ('audit', 'graph_audited')]:
        e = root/name
        exp = read(e/'out/experiment.json')
        assert exp['status'] == 'complete' and exp['exit_code'] == 0 and not exp['DEMO_ONLY']
        assert exp['rounds'] >= 4 and exp['blocks'] == BLOCKS and exp['K'] == 64
        original_code = exp['extensions'][0]['previous_code'] if exp.get('extensions') else exp['code']
        assert original_code['git_commit'].startswith('a742f6f')
        manifests[policy] = exp
        matching[policy] = read(e/'out/matching/matched_subsets.json')
        matchcheck(matching[policy])
        data[policy], protocols[policy] = evaluations(e, 'out', 4)
        summary[policy] = summarize(data[policy], 4)
        supplied = {s: read(e/f'figures/{name}_rounds0-4_{s}_summary.json') for s in data[policy]}
        plot_label = 'unaudited' if policy == 'none' else 'audited B=16'
        compare_summary(summary[policy], {s: {m: supplied[s][m][plot_label] for m in ['pass1', 'target_error_rate']} for s in supplied}, 4)
        training[policy] = {}
        for b, arm in itertools.product(BLOCKS, ARMS):
            training[policy][b+'_'+arm] = {}
            for t in range(1, 5):
                d = e/f'out/{b}/{arm}/round_{t:03d}'
                q = read(d/'complete.json')
                assert (q['round'], q['block_id'], q['branch']) == (t, b, arm)
                assert q['training']['steps'] == q['expected_optimizer_steps']
                assert q['audit_queries'] == (16 if policy == 'audit' else 0)
                if t == 1:
                    m = matching[policy]['per_block'][b]['audit']
                    pre = dict(C=48, E=16, K=64, N_plus=m['N_plus'], N_minus=m['N_minus'],
                               final_TPR=m['final_TPR'], final_FPR=m['final_FPR'])
                else:
                    selection = read(d/'selection.json')
                    pre = selection['audit']
                    assert pre['C'] == 48 and pre['E'] == 16 and selection['K'] == 64
                    close(pre['final_TPR'], 48/pre['N_plus'])
                q['pre'] = pre
                if policy == 'audit':
                    audit = read(d/'audit.json')
                    assert audit['queries'] == len(audit['labels']) == 16
                    comp = audit['after']
                    assert comp['K'] == q['examples'] == comp['C']+comp['E']
                    assert comp['C'] == 48 and comp['N_plus'] == pre['N_plus']
                    assert comp['positive_weight_examples'] == comp['positive_weight_correct']+comp['positive_weight_errors']
                    close(comp['final_TPR'], comp['C']/comp['N_plus'])
                    close(comp['positive_weight_TPR'], comp['positive_weight_correct']/comp['N_plus'])
                    assert q['training']['steps'] == math.ceil(comp['positive_weight_examples']/4)
                else:
                    assert q['examples'] == 64 and q['training']['steps'] == 16
                    comp = dict(K=64, C=48, E=16, positive_weight_examples=64,
                                effective_sample_size=64, positive_weight_TPR=pre['final_TPR'])
                q['composition'] = comp
                training[policy][b+'_'+arm][str(t)] = q
            assert training[policy][b+'_'+arm]['4']['cumulative_audit_queries'] == (64 if policy == 'audit' else 0)
    assert matching['none'] == matching['audit']
    for k in ['dataset_hash', 'shared_adapter', 'pools', 'source_hash']:
        assert manifests['none']['identity'][k] == manifests['audit']['identity'][k]
    for split in protocols['none']:
        for k in ['question_ids', 'dataset_hash', 'base', 'decoding', 'questions']:
            assert protocols['none'][split][k] == protocols['audit'][split][k]
        for key in ['pass1','target_error_rate','error_counts','pass1_by_difficulty']:
            assert data['none'][split]['base'][key] == data['audit'][split]['base'][key]
    contrasts = {s: {a: stats([data['audit'][s][f'{b}_{a}_4']['pass1']-data['none'][s][f'{b}_{a}_4']['pass1'] for b in BLOCKS]) for a in ARMS} for s in data['none']}
    # Verify that the previously published Llama four-round records did not change.
    prior = json.loads((PAPER/'figures/h100_runs/validated_results.json').read_text())
    for policy, name in old.NAMES.items():
        for split in ['eval_id', 'eval_ood']:
            for b, arm, t in itertools.product(BLOCKS, ARMS, range(5)):
                q = read(root/name/'out'/('evaluation_greedy_'+split)/('round_000.json' if t == 0 else f'{b}_{arm}_round_{t:03d}.json'))
                assert q == prior['evaluations'][split][policy][f'{b}_{arm}_{t}']
    return dict(evaluations=data, summary=summary, protocols=protocols, training=training,
                matching=matching['none'], audit_accuracy_change=contrasts, manifests=manifests)


def one_step(root, multi_root):
    result = {}
    for folder, model in MODELS:
        e = root/folder
        exp = read(e/'out_one_step/experiment.json')
        assert exp['status'] == 'complete' and exp['exit_code'] == 0 and not exp['DEMO_ONLY']
        assert exp['blocks'] == BLOCKS and exp['arms'] == ['R', 'S', 'null'] and exp['K'] == 64
        assert exp['code']['git_commit'].startswith('a908fb7') and not exp['code']['git_dirty']
        checks = read(e/'out_one_step/checks.json')
        assert len(checks['checks']) == 43 and all(c['ok'] for c in checks['checks'])
        cfg = read(e/'one_step_config.json')
        assert cfg['rounds'] == 1 and cfg['training']['learning_rate'] == 5e-5
        assert cfg['training']['lora_rank'] == 8 and cfg['training']['lora_alpha'] == 16
        m = read(e/'out_one_step/matching/matched_subsets.json')
        matchcheck(m)
        if model != 'Qwen3-4B':
            assert m == read(multi_root/folder/'out/matching/matched_subsets.json')
        h = read(e/'reference_one_step/h.json')
        data, protocols = evaluations(e, 'out_one_step', 1)
        assert all(p['base']['model'] == cfg['model'] and p['base']['revision'] == cfg['revision'] for p in protocols.values())
        records, prompts = {}, {}
        for b, arm in itertools.product(BLOCKS, ['R', 'S', 'null']):
            d = e/f'out_one_step/{b}/{arm}'
            q = read(d/'round_001/complete.json')
            diag = read(d/'round_001/diagnostics.json')
            run = read(d/'run.json')
            bindings = run['bindings']
            hb = h['bindings']
            for k in ['config_hash', 'shared_adapter_parameter_hash']:
                if k == 'shared_adapter_parameter_hash':
                    assert bindings['shared_adapter']['parameter_hash'] == hb[k]
                else:
                    assert bindings[k] == hb[k]
            # Git normalizes Windows CRLF to LF in the checked-out export.
            raw_config = (e/'one_step_config.json').read_bytes()
            config_hashes = {hashlib.sha256(v).hexdigest() for v in
                             [raw_config, raw_config.replace(b'\r\n', b'\n').replace(b'\n', b'\r\n')]}
            assert bindings['config_file']['sha256'] == hb['config_file']['sha256']
            assert hb['config_file']['sha256'] in config_hashes
            assert bindings['data']['dataset_hash'] == hb['data']['dataset_hash']
            assert q['training']['steps'] == (0 if arm == 'null' else 1)
            assert q['training']['clip_threshold'] == 1 and q['training']['weight_uniform']
            assert diag['validation']['composition_valid']
            assert diag['validation']['relative_error'] < 1e-4
            assert diag['model_params']['param_names'] == h['h']['parameter_names']
            rows = jsonl(d/'round_001/training.jsonl')
            assert len(rows) == 64 and len({r['id'] for r in rows}) == 64
            assert all(r['weight'] == 1 for r in rows)
            selected = m['per_block'][b]['R' if arm == 'null' else arm]
            assert sorted(r['id'] for r in rows) == sorted(selected)
            ps = diag['per_sample']
            assert sorted(r['id'] for r in ps) == sorted(selected)
            assert sum(r['correct'] for r in ps) == 48
            for correct, key in [(True, 'sum_correct_sample_norms'), (False, 'sum_error_sample_norms')]:
                close(sum(r['norm'] for r in ps if r['correct'] == correct), diag['gradient_norms'][key])
            if arm == 'null':
                assert diag['delta_theta']['norm'] == diag['main_diagnostic']['h_T_delta_theta'] == 0
            records[b+'_'+arm] = dict(complete=q, diagnostic=diag, bindings=bindings)
            if arm == 'R':
                prompts[b] = {r['task_id'] for r in rows}
        max_overlap = max(len(prompts[a]&prompts[b])/len(prompts[a]) for a,b in itertools.combinations(BLOCKS,2))
        assert max_overlap < .5
        for b in BLOCKS:
            r=records[b+'_R']['diagnostic'];s=records[b+'_S']['diagnostic']
            assert {v['id'] for v in r['per_sample'] if v['correct']} == {v['id'] for v in s['per_sample'] if v['correct']}
            close(r['gradient_norms']['G_C_vector_norm'],s['gradient_norms']['G_C_vector_norm'])
        if model != 'Qwen3-4B':
            for split in protocols:
                pr=read(multi_root/folder/'out'/('evaluation_greedy_'+split)/'protocol.json')
                for key in ['question_ids','dataset_hash','decoding','base']:
                    assert pr[key]==protocols[split][key],(model,split,key)
        summary = summarize(data, 1)
        supplied = {s: read(e/f'figures/{folder}_one_step_{s}_summary.json') for s in data}
        compare_summary(summary, {s: {m: supplied[s][m]['one-step'] for m in ['pass1', 'target_error_rate']} for s in data}, 1)
        diag_summary = {}
        for metric, section, key in [('projection','main_diagnostic','h_T_delta_theta'), ('norm','delta_theta','norm')]:
            vs = {a: [records[b+'_'+a]['diagnostic'][section][key] for b in BLOCKS] for a in ARMS}
            diag_summary[metric] = {a: stats(vs[a]) for a in ARMS}
            diag_summary[metric]['S-R'] = stats([s-r for s,r in zip(vs['S'],vs['R'])])
            saved = read(e/('figures/'+folder+'_one_step_'+('projection.json' if metric == 'projection' else 'displacement.json')))
            for b in BLOCKS:
                for a in ARMS:
                    close(saved['per_block'][b][a], records[b+'_'+a]['diagnostic'][section][key])
            for k in ['mean','sd','half_width']:
                close(saved['paired']['s_minus_r'][k], diag_summary[metric]['S-R'][k])
        result[model] = dict(config=cfg, manifest=exp, matching=m, reference=h,
                            records=records, protocols=protocols, evaluations=data,
                            summary=summary, diagnostics=diag_summary, max_prompt_overlap=max_overlap)
    return result


def make_combined_endpoint_table(multi):
    """Retain the Qwen3-1.7B policy endpoints in the appendix."""
    llama = json.loads((PAPER/'figures/h100_runs/validated_results.json').read_text())
    headers = ['Policy', 'Arm', '$Q$', 'Steps', 'ID error', 'OOD error',
               r'\shortstack{$\Delta$ID\\(95\% CI)}',
               r'\shortstack{$\Delta$OOD\\(95\% CI)}']
    rows, groups = [], []
    for model, data in [('Qwen3-1.7B', multi)]:
        is_qwen = model == 'Qwen3-1.7B'
        summary = lambda p,s: data['summary'][p][s] if is_qwen else data['summary'][s][p]
        group = [['Base', '---', '0', '0'] +
                 [cell(1-summary('none',s)['pass1']['R']['0']['mean'],4 if is_qwen else 3)
                  for s in ['eval_id','eval_ood']] + ['---','---']]
        for policy, arm in itertools.product(['none','audit'], ARMS):
            records = [data['training'][policy][b+'_'+arm] for b in BLOCKS]
            steps = [sum(q[str(t)]['training']['steps'] for t in range(1,5)) for q in records]
            queries = [q['4']['cumulative_audit_queries'] for q in records]
            assert len(set(queries)) == 1
            row = ['Unaudited' if policy == 'none' else 'Adaptive, B=16', arm,
                   str(queries[0]), str(steps[0]) if policy == 'none' else
                   f'{st.mean(steps):.1f} ({min(steps)}--{max(steps)})']
            row += [cell(1-summary(policy,s)['pass1'][arm]['4']['mean'])
                    for s in ['eval_id','eval_ood']]
            for split in ['eval_id','eval_ood']:
                if policy == 'none':
                    row.append('---')
                else:
                    if is_qwen:
                        v = data['audit_accuracy_change'][split][arm]
                        mean, low, high = -v['mean'], -v['high'], -v['low']
                    else:
                        v = data['audit_minus_none'][split][arm]['error']
                        mean, low, high = v['mean'], v['low'], v['high']
                    row.append('$'+cell(mean)+r'\;['+cell(low)+','+cell(high)+']$')
            group.append(row)
        groups.append((model,group))
        rows.extend(group)
    name = 'qwen17_policy_endpoints_table.tex'
    TABLES[name] = {'headers': headers, 'rows': rows}
    caption = ('Qwen3-1.7B four-round graph policy endpoints, with five paired blocks. '
               'Errors are mean fractions. $Q$ counts correctness-audit queries per arm/block, '
               'excluding matching and evaluation labels; steps are cumulative updates '
               '(mean and range). $\\Delta$ is audited-minus-unaudited error with a '
               'descriptive paired 95\\% block-$t$ interval on fixed questions '
               '(negative favors auditing). Models are not pooled. One-step outcomes '
               'are separate (Table~\\ref{tab:risk}).')
    lines = [r'\begin{table}[H]', r'\caption{'+caption+'}',
             r'\label{tab:qwen17-policy-endpoints}', r'\centering\small\setlength{\tabcolsep}{3pt}',
             r'\begin{tabular}{@{}llcccccc@{}}', r'\toprule',
             ' & '.join(headers)+r'\\', r'\midrule']
    for index,(model,group) in enumerate(groups):
        if index:
            lines.append(r'\addlinespace')
        lines.append(r'\multicolumn{8}{@{}l}{\textbf{'+model+r'}}\\')
        lines += [' & '.join(row)+r'\\' for row in group]
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    (PAPER/name).write_text('\n'.join(lines))


def make_tables(multi, one):
    make_combined_endpoint_table(multi)
    ss, tr, ev = multi['summary'], multi['training'], multi['evaluations']
    headers = ['Split', 'Policy', 'Base', 'Pass@1 $R/S$', '$S-R$ Pass@1', 'Target $R/S$', '$S-R$ target']
    rows = []
    for s, label in [('eval_id','ID'),('eval_ood','OOD')]:
        for p in ['none','audit']:
            rows.append([label, 'None' if p == 'none' else 'B=16', cell(ev[p][s]['base']['pass1'],4),
                pair([ss[p][s]['pass1'][a]['4']['mean'] for a in ARMS]), interval(ss[p][s]['pass1']['S-R']['4']),
                pair([ss[p][s]['target_error_rate'][a]['4']['mean'] for a in ARMS]), interval(ss[p][s]['target_error_rate']['S-R']['4'])])
    table('qwen_four_round_endpoint_table.tex', 'Qwen3-1.7B graph endpoints after four rounds. Target rates count nonshortest answers among all evaluated questions. Paired differences use five blocks, with descriptive 95\\% block-$t$ half-widths; no multiplicity adjustment is applied.', 'tab:qwen-endpoints', headers, rows)
    rows = []
    for s, label in [('eval_id','ID'),('eval_ood','OOD')]:
        base = ev['none'][s]['base']
        rows.append([label,'Base','0',cell(base['pass1'],4),cell(base['pass1'],4),'---',cell(base['target_error_rate']),cell(base['target_error_rate']),'---'])
        for p,t in itertools.product(['none','audit'],range(1,5)):
            rows.append([label,'None' if p == 'none' else 'B=16',str(t)]+[interval(ss[p][s][metric][a][str(t)]) for metric in ['pass1','target_error_rate'] for a in ['R','S','S-R']])
    table('qwen_four_round_trajectory_table.tex', 'Qwen3-1.7B graph trajectories restricted to the original four-round protocol. Entries are means $\\pm$ pointwise 95\\% block-$t$ half-widths. The base is one common checkpoint, not five independent fits.', 'tab:qwen-rounds', ['Split','Policy','Round','Pass@1 $R$','Pass@1 $S$','Pass@1 $S-R$','Target $R$','Target $S$','Target $S-R$'], rows)
    rows = []
    for p,b in itertools.product(['none','audit'],BLOCKS):
        rows.append(['None' if p == 'none' else 'B=16',b]+[pair([ev[p][s][f'{b}_{a}_4'][metric] for a in ARMS]) for metric in ['pass1','target_error_rate'] for s in ['eval_id','eval_ood']])
    table('qwen_four_round_blocks_table.tex', 'Qwen3-1.7B four-round graph endpoint blocks. Each paired cell reports $R/S$; models and splits are not pooled as replicate blocks.', 'tab:qwen-blocks', ['Policy','Block','ID Pass@1','OOD Pass@1','ID target','OOD target'], rows)
    rows = []
    for b in BLOCKS:
        a = multi['matching']['per_block'][b]['audit']
        rows.append([b,pair([a['N_plus'],a['N_minus']],0),str(16384-a['N_plus']-a['N_minus']),pair([a['final_TPR'],a['final_FPR']],6),cell(a['yield'],6),pair([a['R_target_hits'],a['S_target_hits']],0),'pass'])
    table('qwen_four_round_matching_table.tex', 'Qwen3-1.7B graph round-one matching certificate, common to the audited and unaudited runs and their one-step reuse. $N^+/N^-$ exclude truncated answers; $R/S$ share correct IDs, task IDs, 48/16 correct/error quotas and exact supervised response lengths. Later checkpoint-dependent pools are not jointly matched.', 'tab:qwen-matching', ['Block','$N^+/N^-$','Cut','TPR/FPR','Yield','Target hits $R/S$','Controls'],rows)
    rows = []
    for t,p,a in itertools.product(range(1,5),['none','audit'],ARMS):
        q = [tr[p][b+'_'+a][str(t)] for b in BLOCKS]
        avg = lambda k: st.mean(v['composition'][k] for v in q)
        rows.append([str(t),'None' if p == 'none' else 'B=16',a,cell(st.mean(v['training']['h_T_delta_theta'] for v in q),4),cell(st.mean(v['training']['steps'] for v in q),1),cell(avg('K'),1),cell(avg('positive_weight_examples'),1),cell(avg('effective_sample_size'),2),cell(100*st.mean(v['pre']['final_TPR'] for v in q),4),cell(100*st.mean(v['composition']['positive_weight_TPR'] for v in q),4)])
    table('qwen_four_round_audit_table.tex', 'Qwen3-1.7B graph update and audit readouts, means over five blocks. $K\\prime$ counts retained rows, $n^+$ positive-weight rows, and ESS uses saved weights. TPR columns are percentages on each original nontruncated pool: pre-audit $48/N^+$ and positive-weight correct count divided by $N^+$. All 48 correct rows survive deletion, so retained-row post-audit TPR equals pre-audit TPR. Unaudited rows have unit weights. The projection uses the fixed initial reference gradient and is not measured task risk.', 'tab:qwen-audit', ['Round','Policy','Arm','$h^T\\Delta\\theta_t$','Steps','$K\\prime$','$n^+$','ESS','Pre TPR (\\%)','Positive TPR (\\%)'],rows)
    rows = []
    for a in ARMS:
        qs = [tr['audit'][b+'_'+a] for b in BLOCKS]
        steps = [sum(v['training']['steps'] for v in q.values()) for q in qs]
        rows.append([a,'64',cell(st.mean(steps),1)+f' [{min(steps)},{max(steps)}]',interval(multi['audit_accuracy_change']['eval_id'][a]),interval(multi['audit_accuracy_change']['eval_ood'][a])])
    table('qwen_four_round_audit_contrast_table.tex', 'Qwen3-1.7B same-arm audited-minus-unaudited accuracy changes at round four (positive favors auditing), with paired 95\\% block-$t$ half-widths. Each unaudited arm takes 64 steps. Auditing spends 16 correctness queries per arm-round but changes weights, positive-weight training volume, and actual optimizer effort.', 'tab:qwen-audit-change', ['Arm','Queries','Audited steps: mean [range]','ID accuracy change','OOD accuracy change'],rows)
    rows = []
    for model in [m for _,m in MODELS]:
        q = one[model]
        s = q['summary']
        rows.append([model,'Graph','5',pair([1-s['eval_id']['pass1'][a]['1']['mean'] for a in ARMS],4),pair([1-s['eval_ood']['pass1'][a]['1']['mean'] for a in ARMS],4),interval(s['eval_id']['pass1']['S-R']['1'],4),pair([q['diagnostics']['projection'][a]['mean'] for a in ARMS],5),pair([q['diagnostics']['norm'][a]['mean'] for a in ARMS],4)])
    rows += [[m,'Arithmetic','5']+[r'\TBD']*5 for m in ['Llama-3.2-3B','Llama-3.2-1B']]
    expanded = []
    for r in rows[:3]:
        for i,arm in enumerate(ARMS):
            expanded.append((r[:3] if i==0 else ['','','']) + ['$'+arm+'$',
                r[3].split('/')[i],r[4].split('/')[i],r[5] if i else '',
                r[6].split('/')[i],r[7].split('/')[i]])
    expanded += [r[:3]+['---']+r[3:] for r in rows[3:]]
    table('h100_pending_table.tex', 'Unaudited one-step matrix. The three graph comparisons are complete: five paired blocks each, one AdamW update per $R/S$ arm, and an un-updated shared null. Separate rows identify the two arms. The primary paired contrast is $S-R$ in ID greedy Pass@1 (positive favors $S$), with a descriptive 95\\% block-$t$ half-width, reported on the $S$ row. Multiround checkpoints are not substituted.', 'tab:risk', ['Model','Task','Blocks','Arm','ID error','OOD error',r'\shortstack{ID Pass@1\\$S-R$}',r'$h^T\Delta\theta$',r'$\|\Delta\theta\|_2$'],expanded)
    rows = []
    for model in [m for _,m in MODELS]:
        q = one[model]
        for split, label in [('eval_id','ID'),('eval_ood','OOD')]:
            s=q['summary'][split]
            rows.append([model,label,cell(q['evaluations'][split]['base']['pass1'],4),interval(s['pass1']['R']['1'],4),interval(s['pass1']['S']['1'],4),interval(s['pass1']['S-R']['1'],4),pair([s['target_error_rate'][a]['1']['mean'] for a in ARMS],4),interval(s['target_error_rate']['S-R']['1'],4)])
    table('one_step_heldout_table.tex', 'One-step graph held-out results from each run\'s own base/null. Pass@1 and paired $S-R$ entries report means $\\pm$ descriptive 95\\% block-$t$ half-widths. Target incidence is a fraction of all answers. ID contrasts are primary; OOD and target-rate contrasts are secondary, without multiplicity adjustment.', 'tab:one-step-heldout',['Model','Split','Null','Pass@1 $R$','Pass@1 $S$','Pass@1 $S-R$','Target $R/S$','Target $S-R$'],rows)
    rows = []
    for model in [m for _,m in MODELS]:
        q=one[model]
        for a in ARMS:
            d=[q['records'][b+'_'+a]['diagnostic'] for b in BLOCKS]
            rows.append([model,a,interval(q['diagnostics']['projection'][a],5),cell(q['diagnostics']['norm'][a]['mean'],5),cell(st.mean(v['gradient_norms']['G_C_vector_norm'] for v in d),4),cell(st.mean(v['gradient_norms']['G_E_vector_norm'] for v in d),4),cell(st.mean(v['gradient_norms']['G_preclip_norm'] for v in d),4),str(sum(v['clipping']['fired'] for v in d))+'/5'])
    table('one_step_diagnostics_table.tex', 'One-step graph diagnostics at the shared initial adapter. $G_C$ and $G_E$ are correct/error contributions to the common full-batch gradient, not separately renormalized class means. Projections are surrogate first-order diagnostics, not certified reference-loss changes. Norms use measured AdamW displacements, not $-\\eta G$. Clipping uses threshold 1.', 'tab:one-step-diagnostics',['Model','Arm','$h^T\\Delta\\theta$','$\\|\\Delta\\theta\\|_2$','$\\|G_C\\|_2$','$\\|G_E\\|_2$','$\\|G\\|_2$','Clipped'],rows)
    for folder,model in MODELS:
        q=one[model]
        rows=[]
        for b,a in itertools.product(BLOCKS,ARMS):
            d=q['records'][b+'_'+a]['diagnostic']
            rows.append([b,a]+[cell(q['evaluations'][s][f'{b}_{a}_1']['pass1'],4) for s in ['eval_id','eval_ood']]+[cell(d['main_diagnostic']['h_T_delta_theta'],5),cell(d['delta_theta']['norm'],5)])
        table(f'one_step_{folder}_blocks_table.tex', f'{model} one-step graph blocks. Each split uses the same fixed questions for its base/null and all ten trained checkpoints.', 'tab:one-step-blocks-'+folder,['Block','Arm','ID Pass@1','OOD Pass@1','$h^T\\Delta\\theta$','$\\|\\Delta\\theta\\|_2$'],rows)


def figures(multi, one):
    # Reuse Figure 2's exact palette, fonts, discrete checkpoints and intervals.
    old.TCRIT = TCRIT
    s = {split:{p:multi['summary'][p][split] for p in ['none','audit']} for split in ['eval_id','eval_ood']}
    old.chart(OUT/'qwen_accuracy.pdf',s,[('eval_id','pass1',(0,.75),(0,.25,.5,.75),'(a) ID: 2,000 questions','Greedy Pass@1'),('eval_ood','pass1',(0,.75),(0,.25,.5,.75),'(b) OOD: 1,000 questions','Greedy Pass@1')])
    old.chart(OUT/'qwen_target_errors.pdf',s,[('eval_id','target_error_rate',(0,.3),(0,.1,.2,.3),'(a) ID target errors','Fraction of all answers'),('eval_ood','target_error_rate',(0,.3),(0,.1,.2,.3),'(b) OOD target errors','Fraction of all answers')])
    for name in ['qwen_accuracy.pdf','qwen_target_errors.pdf']:
        p=OUT/name
        writer=PdfWriter(clone_from=PdfReader(p))
        writer.add_metadata({'/Title':'Qwen3-1.7B graph: four-round greedy evaluation'})
        with p.open('wb') as f:writer.write(f)
    c=canvas.Canvas(str(OUT/'one_step_contrasts.pdf'),pagesize=(486,230),invariant=1)
    c.setTitle('One-step graph: paired accuracy contrasts')
    c.setAuthor('Anonymous research manuscript')
    ink='#38434A'
    colors={'eval_id':'#127F83','eval_ood':'#805677'}
    c.setFillColor(ink);c.setFont('H100Bold',10)
    c.drawCentredString(243,216,'One-step graph: paired S - R accuracy')
    c.setFont('H100Sans',8.5);c.drawCentredString(243,201,'Five blocks; 95% block-t intervals; fixed greedy evaluation questions')
    models=[m for _,m in MODELS]
    for j,split in enumerate(['eval_id','eval_ood']):
        x0,y0,w,h=145+166*j,54,142,117
        low,high=-.007,.007
        px=lambda v:x0+(v-low)/(high-low)*w
        c.setFillColor(ink);c.setFont('H100Bold',9)
        c.drawCentredString(x0+w/2,181,'ID (primary)' if split=='eval_id' else 'OOD (secondary)')
        c.setStrokeColor('#D8DEDF');c.setLineWidth(.6);c.line(px(0),y0,px(0),y0+h)
        c.setStrokeColor(ink);c.line(x0,y0,x0+w,y0)
        for v in [-.005,0,.005]:
            c.line(px(v),y0,px(v),y0-3);c.setFont('H100Sans',8);c.drawCentredString(px(v),y0-14,f'{v:+.3f}' if v else '0')
        for i,model in enumerate(models):
            y=y0+h-14-40*i
            q=one[model]['summary'][split]['pass1']['S-R']['1']
            c.setStrokeColor(colors[split]);c.setFillColor(colors[split]);c.setLineWidth(1.2)
            c.line(px(q['low']),y,px(q['high']),y)
            for v in [q['low'],q['high']]:c.line(px(v),y-3,px(v),y+3)
            c.circle(px(q['mean']),y,3,stroke=1,fill=1 if split=='eval_id' else 0)
            c.setFillColor(ink);c.setFont('H100Sans',8)
            c.drawCentredString(x0+w/2,y-17,f"{q['mean']:+.4f} +/- {q['half_width']:.4f}")
            if j==0:
                c.setFont('H100Sans',9);c.drawRightString(134,y-3,model)
    c.setFont('H100Sans',8.5);c.drawCentredString(314,15,'Pass@1 difference (positive favors S)')
    c.showPage();c.save()


def main():
    global CODE,OUT
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code',type=Path,required=True)
    ap.add_argument('--out',type=Path,required=True)
    args=ap.parse_args();CODE=args.code.resolve();OUT=args.out.resolve();OUT.mkdir(parents=True,exist_ok=True)
    multi_root=CODE/'experiments/outputs/2026-10-03-h100'
    one_root=CODE/'experiments/outputs/2026-10-04-one-step'
    multi=multiround(multi_root);one=one_step(one_root,multi_root)
    make_tables(multi,one);figures(multi,one)
    result=dict(multiround=multi,one_step=one)
    (OUT/'validated_results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    (OUT/'table_contract.json').write_text(json.dumps(TABLES,indent=2,sort_keys=True)+'\n')
    provenance=dict(source_files=READ,script_sha256=digest(Path(__file__)),
        selected_rounds=[0,1,2,3,4],multiround_snapshot='2026-10-04 23:35 UTC',
        one_step_export='2026-10-04-one-step',
        checks='Count identities; five-block completeness; exact matching controls; steps/queries; config-bound reference; training IDs, unit weights, diagnostic label counts; exported means/SD/t intervals; unchanged original Llama held-out records.',
        limitations=['Raw held-out answers and adapter/reference tensors omitted: no held-out rejudging, adapter reload, or independent tensor-level gradient calculation.',
                     'Hardware/platform baselines are run-specific; no cross-hardware reproducibility claim.',
                     'Eight-round extension excluded: requested four-round view only.',
                     'Recorded original one-step acceptance checks pass 43/43; adapter-hash rechecking requires omitted adapter files.'],
        chart_contract=dict(surface='LaTeX manuscript / vector PDF',
            question='Four-round Qwen accuracy and error composition; three-model one-step paired contrasts',
            families=['discrete-checkpoint comparison with intervals','faceted dot and interval'],
            data_sufficiency='Five measured checkpoints for four-round comparison, the full available requested horizon; three intentionally labeled model-level contrasts, each with five paired blocks.',
            palette={'R':'#127F83','S':'#805677','ink':'#38434A'},
            non_color='solid/dashed arms, filled/open audit markers; faceted labeled ID/OOD contrasts',
            footprints_points=[[486,184],[486,230]],qa='Rendered manuscript pages and standalone PDF previews'),
        exports={p.name:digest(p) for p in OUT.iterdir() if p.is_file() and p.name != 'provenance.json'})
    (OUT/'README.md').write_text('''# Validated graph additions\n\nSources: the H100 export updated 2026-10-04 23:35 UTC, restricted to rounds 0--4, and the complete 2026-10-04-one-step export.\n\nFrom the manuscript directory:\n\n```sh\npython summarize_graph_extensions.py --code .. --out figures/graph_extensions\npython verify_graph_extensions.py --code ..\npython verify_h100_tables.py\n```\n\nDependencies: Python 3, reportlab, pypdf. No GPU, model weights, network, or training.\n\n`validated_results.json` stores the checked checkpoint/audit/diagnostic records and recomputed paired intervals; `table_contract.json` stores the rendered table-cell contract. `provenance.json` binds consumed source files and exports by SHA-256. The reference and held-out adapter tensors and raw held-out answers are not included: validation does not rejudge held-out responses or reconstruct gradients. Recorded original one-step acceptance reports passed 43/43 checks on the execution machines; adapter-hash rechecking needs their omitted adapters.\n\nThe one-step primary contrast is S-minus-R **Pass@1**, opposite in sign to the source export's S-minus-R error contrast. Intervals are descriptive pointwise block-t intervals (five blocks), unadjusted for multiple comparisons and conditional on fixed question sets. Each run retains its own base/null because platform-specific base predictions differ slightly.\n\nCharts use the manuscript's Figure 1/2 teal/plum palette and vector rendering. R/S differ by line style, auditing by open/filled markers; one-step contrasts use labeled ID/OOD facets. Every requested round and block is retained, with no smoothing. QA uses the compiled manuscript pages.\n''')
    readme = OUT/'README.md'
    readme.write_text(readme.read_text() +
        '\nThe combined main-text endpoint table is also regenerated as '
        '`h100_endpoint_table.tex`, using separate five-block summaries for '
        'Llama and Qwen. Its audit contrasts are changes in **error** '
        '(negative favors auditing), not accuracy; the Qwen appendix audit-change '
        'table uses the opposite sign. Qwen base accuracy/error use four decimal '
        'places to avoid ambiguous rounding of 999/2000 and 1001/2000. '
        'The main-text Qwen accuracy figure has the same two-panel layout and '
        '0.7-textwidth footprint as the Llama figure. The separate target-error '
        'plot remains in the appendix.\n')
    provenance['exports'] = {p.name:digest(p) for p in OUT.iterdir()
                             if p.is_file() and p.name != 'provenance.json'}
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='validated',source_files=len(READ),
        qwen_endpoints={s:{p:{a:multi['summary'][p][s]['pass1'][a]['4']['mean'] for a in ARMS} for p in ['none','audit']} for s in ['eval_id','eval_ood']},
        one_step_accuracy_contrasts={m:{s:one[m]['summary'][s]['pass1']['S-R']['1'] for s in ['eval_id','eval_ood']} for _,m in MODELS}),indent=2))


if __name__ == '__main__':main()
