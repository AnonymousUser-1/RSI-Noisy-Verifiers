#!/usr/bin/env python3
"""Check every numeric H100 table cell against validated_results.json (stdlib)."""
import json
import math
from pathlib import Path
import re
import statistics as st
from table_typography import canonical_rows

ROOT = Path(__file__).resolve().parent
D = json.loads((ROOT/'figures/h100_runs/validated_results.json').read_text())
CHECKS = 0
BLOCKS = [f'b{i:02}' for i in range(5)]


def rows(name):
    text=(ROOT/name).read_text()
    body=text.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    return [[x.strip() for x in line[:-2].split('&')]
            for line in body.splitlines() if '&' in line and line.endswith('\\\\')]


def nums(cell):
    # TeX's -- is a range dash, not a minus sign on its right endpoint.
    return [float(v) for v in re.findall(r'[-+]?(?:\d+\.\d+|\d+)',cell.replace('--',' '))]


def check(cell, expected, decimals):
    global CHECKS
    actual=nums(cell)
    assert len(actual)==len(expected),(cell,expected)
    for v,w in zip(actual,expected):
        assert abs(v-w)<=.5*10**(-decimals)+1e-10,(cell,expected)
        CHECKS+=1


def main():
    for b,r in zip(BLOCKS,rows('h100_matching_table.tex')):
        a=D['matching']['per_block'][b]['audit']
        assert r[0]==b
        check(r[1],[a['N_plus'],a['N_minus']],0)
        check(r[2],[16384-a['N_plus']-a['N_minus']],0)
        check(r[3],[a['final_TPR'],a['final_FPR']],6)
        check(r[4],[a['yield']],6)
        check(r[5],[a['R_target_hits'],a['S_target_hits']],0)
        assert r[6]=='pass'
    rr=rows('h100_endpoint_table.tex')
    assert len(rr)==11
    check(rr[0][4],[.752],3);check(rr[0][5],[.791],3)
    for r,(po,arm) in zip(rr[1:5],[(po,a) for po in ['none','audit'] for a in ['R','S']]):
        tr=[D['training'][po][b+'_'+arm] for b in BLOCKS]
        steps=[sum(q['training']['steps'] for q in t.values()) for t in tr]
        check(r[2],[tr[0]['4']['cumulative_audit_queries']],0)
        check(r[3],[64] if po=='none' else [st.mean(steps),min(steps),max(steps)],1)
        for col,sp in [(4,'eval_id'),(5,'eval_ood')]:
            check(r[col],[1-D['summary'][sp][po]['pass1'][arm]['4']['mean']],3)
        if po=='audit':
            for col,sp in [(6,'eval_id'),(7,'eval_ood')]:
                v=D['audit_minus_none'][sp][arm]['error']
                check(r[col],[v['mean'],v['low'],v['high']],3)
    extra=json.loads((ROOT/'figures/graph_extensions/validated_results.json').read_text())
    qwen=extra['multiround']
    rr=rows('qwen17_policy_endpoints_table.tex')
    assert len(rr)==5
    for col,split in [(4,'eval_id'),(5,'eval_ood')]:
        check(rr[0][col],[1-qwen['evaluations']['none'][split]['base']['pass1']],4)
    assert rr[0][:4]==['Base','---','0','0']
    for r,(po,arm) in zip(rr[1:],[(po,a) for po in ['none','audit'] for a in ['R','S']]):
        assert r[:2]==['Unaudited' if po=='none' else 'Adaptive, B=16',arm]
        tr=[qwen['training'][po][b+'_'+arm] for b in BLOCKS]
        steps=[sum(q[str(t)]['training']['steps'] for t in range(1,5)) for q in tr]
        check(r[2],[tr[0]['4']['cumulative_audit_queries']],0)
        check(r[3],[64] if po=='none' else [st.mean(steps),min(steps),max(steps)],1)
        for col,split in [(4,'eval_id'),(5,'eval_ood')]:
            check(r[col],[1-qwen['summary'][po][split]['pass1'][arm]['4']['mean']],3)
        if po=='audit':
            for col,split in [(6,'eval_id'),(7,'eval_ood')]:
                differences=[qwen['evaluations']['none'][split][b+'_'+arm+'_4']['pass1']-
                             qwen['evaluations']['audit'][split][b+'_'+arm+'_4']['pass1']
                             for b in BLOCKS]
                mean=st.mean(differences)
                half=2.7764451051977987*st.stdev(differences)/math.sqrt(5)
                check(r[col],[mean,mean-half,mean+half],3)
        else: assert r[6:]==['---','---']
    rr=rows('h100_rounds_table.tex')
    assert len(rr)==18
    for r in rr:
        sp='eval_id' if r[0].startswith('ID') else 'eval_ood'
        po='audit' if r[1]=='B=16' else 'none';t=str(int(r[2]))
        for col,met,arm in [(3,'pass1','R'),(4,'pass1','S'),(5,'pass1','S-R'),
                            (6,'target_error_rate','R'),(7,'target_error_rate','S'),(8,'target_error_rate','S-R')]:
            if t=='0' and arm=='S-R': assert r[col]=='---';continue
            q=D['summary'][sp][po][met][arm][t]
            check(r[col],[q['mean']] if t=='0' else [q['mean'],q['half_width']],3)
    for r in rows('h100_blocks_table.tex'):
        po='none' if r[0]=='None' else 'audit';b=r[1]
        for col,sp,met in [(2,'eval_id','pass1'),(3,'eval_ood','pass1'),
                           (4,'eval_id','target_error_rate'),(5,'eval_ood','target_error_rate')]:
            check(r[col],[D['evaluations'][sp][po][b+'_'+a+'_4'][met] for a in ['R','S']],3)
    rr=rows('h100_dynamics_table.tex')
    assert len(rr)==16
    for r in rr:
        po='none' if r[0]=='None' else 'audit';t=r[1];arm=r[2]
        vs=[D['training'][po][b+'_'+arm][t] for b in BLOCKS]
        for col,k in [(3,'h_T_delta_theta'),(4,'delta_theta_norm'),(5,'mean_loss')]:
            check(r[col],[st.mean(v['training'][k] for v in vs)],4)
        check(r[6],[st.mean(v['training']['steps'] for v in vs)],1)
        check(r[7],[st.mean(v['composition']['positive_weight_examples'] for v in vs)],1)
        check(r[8],[st.mean(v['composition']['effective_sample_size'] for v in vs)],2)
    for r in rows('h100_composition_table.tex')[1:]:
        tr=D['training']['audit'][r[0]+'_'+r[1]];q=tr['4'];c=q['composition']
        check(r[2],[c['K'],c['positive_weight_examples']],0)
        check(r[3],[c['C'],c['E']],0)
        check(r[4],[c['effective_sample_size']],2)
        check(r[5],[sum(v['training']['steps'] for v in tr.values())],0)
        check(r[6],[q['cumulative_audit_queries']],0)
    for r,q in zip(rows('h100_feasibility_table.tex'),D['feasibility']):
        assert r[1]==q['block']
        check(r[2],[100*q['correct']],2)
        for col,k in [(3,'truncated'),(4,'target_errors'),(5,'J_exact'),(6,'J_actual'),(8,'K')]:
            check(r[col],[q[k]],0)
        assert r[7]==('off' if q['tolerance'] is None else 'exact')
    q4=json.loads((ROOT/'figures/qwen4b_multiround/detailed_results.json').read_text())
    models=['Llama-3.2-3B','Qwen3-1.7B','Qwen3-4B']
    expected=[]
    keys={f'{b}_{a}_{t}' for b in BLOCKS for a in ['R','S'] for t in range(1,5)}
    for model in models:
        for po in ['none','audit']:
            training=({'Llama-3.2-3B':D,'Qwen3-1.7B':qwen,'Qwen3-4B':q4}[model])['training'][po]
            assert set(training)=={b+'_'+a for b in BLOCKS for a in ['R','S']}
            assert all(all(str(t) in group for t in range(1,5)) for group in training.values())
            for split in ['eval_id','eval_ood']:
                ev=D['evaluations'][split][po] if model=='Llama-3.2-3B' else (qwen if model=='Qwen3-1.7B' else q4)['evaluations'][po][split]
                assert keys<=set(ev)
                assert any(v.get('round')==0 for v in ev.values()) or 'base' in ev
            expected.append([model,'Four rounds','None' if po=='none' else 'B=16','5','40','41'])
    for model in models:
        q=extra['one_step'][model]
        for split in ['eval_id','eval_ood']:
            assert set(q['evaluations'][split])=={'base'}|{f'{b}_{a}_1' for b in BLOCKS for a in ['R','S']}
        expected.append([model,'One step','None','5','10','11'])
    assert rows('h100_progress_table.tex')==expected
    pending=canonical_rows('h100_pending_table.tex',rows('h100_pending_table.tex'))
    planned=[('Qwen3-1.7B','Graph'),('Qwen3-4B','Graph'),
             ('Llama-3.2-3B','Graph'),('Llama-3.2-3B','Arithmetic'),
             ('Llama-3.2-1B','Arithmetic')]
    assert len(pending)==len(planned)
    for r,(model,task) in zip(pending,planned):
        assert r[:3]==[model,task,'5'],r
        assert len(r)==8,r
        if task=='Arithmetic':
            assert all(v==r'\TBD' for v in r[3:]),r
            continue
        q=extra['one_step'][model];s=q['summary']
        for col,split in [(3,'eval_id'),(4,'eval_ood')]:
            check(r[col],[1-s[split]['pass1'][a]['1']['mean'] for a in ['R','S']],4)
        v=s['eval_id']['pass1']['S-R']['1']
        check(r[5],[v['mean'],v['half_width']],4)
        check(r[6],[q['diagnostics']['projection'][a]['mean'] for a in ['R','S']],5)
        check(r[7],[q['diagnostics']['norm'][a]['mean'] for a in ['R','S']],4)
    rates=json.loads((ROOT/'figures/h100_runs/tpr_rates/validated_tpr.json').read_text())
    records=rates['records']
    assert len(records)==len({(q['policy'],q['block'],q['arm'],q['round']) for q in records})==80
    rate_rows=rows('h100_tpr_table.tex')
    assert len(rate_rows)==16
    expected_keys=[(t,po,a) for t in range(1,5) for po in ['none','audit'] for a in ['R','S']]
    for r,(t,po,arm) in zip(rate_rows,expected_keys):
        assert r[:3]==[str(t),'None' if po=='none' else 'B=16',arm]
        group=[q for q in records if (q['round'],q['policy'],q['arm'])==(t,po,arm)]
        assert sorted(q['block'] for q in group)==BLOCKS
        for q in group:
            assert q['C_pre']==48 and q['N_plus']>0
            assert abs(q['TPR_pre']-48/q['N_plus'])<1e-13
            if po=='audit':
                c=D['training']['audit'][q['block']+'_'+arm][str(t)]['composition']
                assert (q['N_plus'],q['C_retained'],q['C_positive'])==(c['N_plus'],c['C'],c['positive_weight_correct'])
        check(r[3],[st.mean(q['N_plus'] for q in group),min(q['N_plus'] for q in group),max(q['N_plus'] for q in group)],1)
        check(r[4],[100*st.mean(48/q['N_plus'] for q in group)],4)
        if po=='audit':
            check(r[5],[100*st.mean(q['C_retained']/q['N_plus'] for q in group)],4)
            check(r[6],[100*st.mean(q['C_positive']/q['N_plus'] for q in group)],4)
        else:
            assert r[5:]==['---','---']
    print(json.dumps({'status':'passed','numeric_cells_checked':CHECKS,
                     'scope':'H100 display tables vs validated numerical/count artifacts; raw-response rejudging not possible.'}))


if __name__=='__main__':main()
