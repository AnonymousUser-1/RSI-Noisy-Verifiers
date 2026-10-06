#!/usr/bin/env python3
"""Reconstruct detailed Qwen3-4B tables and verify their displayed cells.

Uses saved checkpoint/completion/audit records and hash-bound paired input
pools. Pool labels are independently rejudged; exact-length eligibility is
reported from the matching certificates, not retokenized here.
"""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import statistics as st
import sys
from table_typography import numeric_math

PAPER=Path(__file__).resolve().parent
BLOCKS=[f'b{i:02d}' for i in range(5)]
TCRIT=2.7764451051977987

def stats(v):
    m=st.mean(v);half=TCRIT*st.stdev(v)/math.sqrt(5)
    return dict(mean=m,half_width=half,low=m-half,high=m+half)

def interval(v,d=4):return '$'+f'{v["mean"]:.{d}f}'+r'\pm'+f'{v["half_width"]:.{d}f}'+'$'
def pair(v,d=4):return '/'.join(f'{x:.{d}f}' for x in v)

def rows(name):
    body=(PAPER/name).read_text().split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
    return [[x.strip() for x in line[:-2].split('&')] for line in body.splitlines() if '&' in line and line.endswith('\\\\')]

def main():
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--code',type=Path,required=True)
    ap.add_argument('--check',action='store_true');args=ap.parse_args();code=args.code.resolve()
    sys.path.insert(0,str(code))
    from rsi.tasks import judge
    from rsi.common import digest
    sources={}
    def read(p,jsonl=False):
        raw=p.read_bytes();sources[str(p.relative_to(code))]=hashlib.sha256(raw).hexdigest()
        return [json.loads(x) for x in raw.decode().splitlines() if x.strip()] if jsonl else json.loads(raw)
    roots={'none':code/'experiments/outputs/2026-10-05-qwen3-4b-multiround-unaudited/graph_unaudited_qwen3-4b',
           'audit':code/'experiments/outputs/2026-10-05-qwen3-4b-graph-audit-b16'}
    q4=json.loads((PAPER/'figures/qwen4b_multiround/validated_results.json').read_text())
    matching={po:read(root/'out/matching/matched_subsets.json') for po,root in roots.items()}
    assert matching['none']==matching['audit']==q4['unaudited_matching']
    manifests={po:read(root/'out/experiment.json') for po,root in roots.items()}
    training={};evaluation={};progress=[]
    for po,root in roots.items():
        assert manifests[po]['status']=='complete' and len(manifests[po]['completed'])==40
        finished=[read(root/f'out/{b}/{a}/finished.json') for b in BLOCKS for a in ['R','S']]
        assert len(finished)==10
        progress.append(['Graph','Qwen3-4B','None' if po=='none' else 'B=16','40/40','10/10','41/41','41/41','complete'])
        evaluation[po]={}
        for split,n in [('eval_id',2000),('eval_ood',1000)]:
            ev={'base':read(root/f'out/evaluation_greedy_{split}/round_000.json')}
            for b in BLOCKS:
                for a in ['R','S']:
                    for t in range(1,5):
                        row=read(root/f'out/evaluation_greedy_{split}/{b}_{a}_round_{t:03d}.json')
                        assert row['questions']==n and row['block']==b and row['arm']==a and row['round']==t
                        assert abs(row['pass1']-(1-sum(row['error_counts'].values())/n))<1e-12
                        assert abs(row['target_error_rate']-row['error_counts'].get('nonshortest',0)/n)<1e-12
                        ev[f'{b}_{a}_{t}']=row
            evaluation[po][split]=ev
        training[po]={}
        for b in BLOCKS:
            for a in ['R','S']:
                training[po][b+'_'+a]={}
                for t in range(1,5):
                    folder=root/f'out/{b}/{a}/round_{t:03d}'
                    completion=read(folder/'complete.json');selection=read(folder/'selection.json')
                    pre=matching[po]['per_block'][b]['audit'] if t==1 else selection['audit']
                    assert (completion['block_id'],completion['branch'],completion['round'])==(b,a,t)
                    assert pre['N_plus']>0 and completion['cumulative_audit_queries']==(0 if po=='none' else 16*t)
                    if po=='audit':
                        audit=read(folder/'audit.json');comp=audit['after']
                        assert comp['C']==48 and comp['N_plus']==pre['N_plus']
                        assert completion['training']['steps']==math.ceil(comp['positive_weight_examples']/4)
                        assert completion['examples']==comp['K']
                    else:
                        assert completion['examples']==64 and completion['training']['steps']==16
                        comp=dict(K=64,C=48,E=16,positive_weight_examples=64,positive_weight_correct=48,effective_sample_size=64)
                    training[po][b+'_'+a][str(t)]=dict(completion=completion,pre=pre,composition=comp)
    tasks={q['id']:q for q in read(code/'RSI-Qwen3-4B-Graph-paired-inputs-20261004/graph/train_001.jsonl',True)}
    feasible=[]
    for b in BLOCKS:
        pool=code/f'RSI-Qwen3-4B-Graph-paired-inputs-20261004/pools/{b}.jsonl'
        pool_rows=read(pool,True)
        assert sources[str(pool.relative_to(code))]==manifests['audit']['identity']['pools'][b]
        assert len(pool_rows)==len({x['id'] for x in pool_rows})==16384
        labels=[];eligible_correct=0;cut=0
        for x in pool_rows:
            assert x['id']==digest([x['task_id'],x['sample']])
            result=judge(tasks[x['task_id']],x['response']);labels.append(result['error'])
            cut+=x['truncated'];eligible_correct+=result['correct'] and not x['truncated']
        counts=Counter(labels);cert=matching['none']['per_block'][b]['certificate']
        assert eligible_correct==cert['N_plus'] and cut==cert['truncated_excluded']
        assert cert['token_tolerance']==0 and all(cert[k] for k in ['condition_correct_tasks','condition_error_tasks','condition_one_task_each'])
        j=cert['tasks_eligible_J_E'];assert j>=16
        feasible.append(dict(block=b,correct=counts.get('correct',0)/16384,truncated=cut,target_errors=counts.get('nonshortest',0),J_exact=j,J_actual=j,K=64,label_counts=dict(counts)))
    tables={}
    def table(name,caption,label,headers,rr):
        rr = [[numeric_math(c) for c in row] for row in rr]
        tables[name]=dict(headers=headers,rows=rr)
        if args.check:
            assert rows(name)==rr,name
            return
        text=[r'\begin{table}[H]',r'\caption{'+caption+'}',r'\label{'+label+'}',
              r'\centering\small\setlength{\tabcolsep}{3pt}',r'\begin{tabular}{@{}'+'l'*len(headers)+r'@{}}',r'\toprule',
              ' & '.join(headers)+r'\\',r'\midrule']
        text+=[' & '.join(r)+r'\\' for r in rr];text += [r'\bottomrule',r'\end{tabular}',r'\end{table}','']
        (PAPER/name).write_text('\n'.join(text))
    matchrows=[]
    for b,q in matching['none']['per_block'].items():
        a=q['audit'];c=q['certificate']
        matchrows.append([b,f'{a["N_plus"]}/{a["N_minus"]}',str(c['truncated_excluded']),pair([a['final_TPR'],a['final_FPR']],6),f'{a["yield"]:.6f}',f'{a["R_target_hits"]}/{a["S_target_hits"]}','pass'])
    table('qwen4b_matching_table.tex',r'Qwen3-4B round-one matching, shared across both four-round policies and the one-step reuse. $N^+/N^-$ exclude truncated answers. Both arms have identical prompt and correct-response IDs, 48/16 correct/error quotas, and exact per-prompt supervised response lengths. Later checkpoint-dependent pools are not jointly matched.','tab:qwen4b-matching',
          ['Block',r'$N^+/N^-$','Cut','TPR/FPR','Yield',r'Target hits $R/S$','Controls'],matchrows)
    def summary(po,split,metric,a,t):
        ev=evaluation[po][split]
        return stats([ev[f'{b}_{a}_{t}'][metric] if a!='S-R' else ev[f'{b}_S_{t}'][metric]-ev[f'{b}_R_{t}'][metric] for b in BLOCKS])
    trajectories=[]
    for split in ['eval_id','eval_ood']:
        for po in ['none','audit']:
            base=evaluation[po][split]['base'];display='ID' if split=='eval_id' else 'OOD'
            trajectories.append([display,'Base, '+('none' if po=='none' else 'B=16'),'0',f'{base["pass1"]:.3f}',f'{base["pass1"]:.3f}','---',f'{base["target_error_rate"]:.3f}',f'{base["target_error_rate"]:.3f}','---'])
            for t in range(1,5):
                trajectories.append([display,'None' if po=='none' else 'B=16',str(t)]+[interval(summary(po,split,m,a,t),3) for m in ['pass1','target_error_rate'] for a in ['R','S','S-R']])
    table('qwen4b_trajectory_table.tex',r'Qwen3-4B four-round graph trajectories: means $\pm$ pointwise descriptive 95\% block-$t$ half-widths over five blocks. Target incidence counts nonshortest answers among all evaluated questions.','tab:qwen4b-rounds',
          ['Split','Policy','Round',r'Pass@1 $R$',r'Pass@1 $S$',r'Pass@1 $S-R$',r'Target $R$',r'Target $S$',r'Target $S-R$'],trajectories)
    blockrows=[]
    for po in ['none','audit']:
        for b in BLOCKS:
            blockrows.append(['None' if po=='none' else 'B=16',b]+[pair([evaluation[po][s][f'{b}_{a}_4'][m] for a in ['R','S']]) for s,m in [('eval_id','pass1'),('eval_ood','pass1'),('eval_id','target_error_rate'),('eval_ood','target_error_rate')]])
    table('qwen4b_blocks_table.tex',r'Qwen3-4B round-four graph endpoints by paired block. Slash-separated entries are $R/S$ values, not ratios. All five blocks are retained, including unfavorable audit changes.','tab:qwen4b-blocks',
          ['Policy','Block','ID Pass@1','OOD Pass@1','ID target','OOD target'],blockrows)
    contrasts=[]
    for a in ['R','S']:
        steps=[sum(q['completion']['training']['steps'] for q in training['audit'][b+'_'+a].values()) for b in BLOCKS]
        contrasts.append([a,'64',f'{st.mean(steps):.1f} [{min(steps)}--{max(steps)}]']+[interval(stats([evaluation['audit'][s][f'{b}_{a}_4']['pass1']-evaluation['none'][s][f'{b}_{a}_4']['pass1'] for b in BLOCKS])) for s in ['eval_id','eval_ood']])
    table('qwen4b_audit_contrast_table.tex',r'Qwen3-4B same-arm audited-minus-unaudited accuracy changes at round four (positive favors auditing), with paired 95\% block-$t$ half-widths. Unaudited arms each take 64 steps; auditing also changes retained data, weights, and optimizer effort.','tab:qwen4b-audit-change',
          ['Arm','Queries','Audited steps: mean [range]','ID accuracy change','OOD accuracy change'],contrasts)
    readouts=[];diagnostics=[]
    for t in range(1,5):
        for po in ['none','audit']:
            for a in ['R','S']:
                group=[training[po][b+'_'+a][str(t)] for b in BLOCKS]
                avg=lambda field:st.mean(q['composition'][field] for q in group)
                mean_training=lambda field:st.mean(q['completion']['training'][field] for q in group)
                readouts.append([str(t),'None' if po=='none' else 'B=16',a,f'{mean_training("h_T_delta_theta"):.4f}',f'{mean_training("steps"):.1f}',f'{avg("K"):.1f}',f'{avg("positive_weight_examples"):.1f}',f'{avg("effective_sample_size"):.2f}',f'{100*st.mean(48/q["pre"]["N_plus"] for q in group):.4f}',f'{100*st.mean(q["composition"]["positive_weight_correct"]/q["pre"]["N_plus"] for q in group):.4f}'])
                diagnostics.append(['Graph','Qwen3-4B','None' if po=='none' else 'B=16',str(t),a,'5',f'{mean_training("h_T_delta_theta"):.4f}',f'{mean_training("delta_theta_norm"):.4f}',f'{mean_training("mean_loss"):.4f}'])
    table('qwen4b_audit_readouts_table.tex',r'Qwen3-4B graph update and audit readouts, means over five blocks. $K\prime$ counts retained rows, $n^+$ positive-weight rows, and ESS is verified from audited weights. TPRs are percentages on each round-specific original nontruncated pool: pre-audit $48/N^+$ and positive-weight correct count divided by $N^+$. All 48 correct rows survive deletion, so retained post-audit TPR equals pre-audit TPR. The projection uses the fixed initial reference gradient, not measured task risk.','tab:qwen4b-audit',
          ['Round','Policy','Arm',r'$h^T\Delta\theta_t$','Steps',r'$K\prime$',r'$n^+$','ESS',r'Pre TPR (\%)',r'Positive TPR (\%)'],readouts)
    table('qwen4b_training_table.tex',r'Qwen3-4B four-round training diagnostics, means over five blocks per arm and policy. $\Delta\theta_t$ is the parameter change within round $t$; $h$ is fixed at initialization. Norms are not cumulative displacement norms, and loss is the policy-specific weighted training objective. Neither projection nor training loss is held-out task accuracy.','tab:qwen4b-training',
          ['Round','Audit','Arm',r'$h^T\Delta\theta_t$',r'$\|\Delta\theta_t\|_2$','Loss'],
          [[r[i] for i in [3,2,4,6,7,8]] for r in diagnostics])
    # Llama endpoint and audit-change views support the parallel model subsection.
    llama=json.loads((PAPER/'figures/h100_runs/validated_results.json').read_text());lr=[]
    for split in ['eval_id','eval_ood']:
        for po in ['none','audit']:
            su=llama['summary'][split][po];base=su['pass1']['R']['0']['mean']
            lr.append(['ID' if split=='eval_id' else 'OOD','None' if po=='none' else 'B=16',f'{base:.4f}',pair([su['pass1'][a]['4']['mean'] for a in ['R','S']],3),interval(su['pass1']['S-R']['4'],3),pair([su['target_error_rate'][a]['4']['mean'] for a in ['R','S']],3),interval(su['target_error_rate']['S-R']['4'],3)])
    table('llama_four_round_endpoint_table.tex',r'Llama-3.2-3B graph endpoints after four rounds. Target rates count nonshortest answers among all evaluated questions. Paired $S-R$ differences are within policy, with descriptive 95\% block-$t$ half-widths over five blocks. The base is one shared checkpoint.','tab:llama-endpoints',
          ['Split','Policy','Base',r'Pass@1 $R/S$',r'$S-R$ Pass@1',r'Target $R/S$',r'$S-R$ target'],lr)
    lr=[]
    for a in ['R','S']:
        steps=[sum(q['training']['steps'] for q in llama['training']['audit'][b+'_'+a].values()) for b in BLOCKS]
        values=[]
        for split in ['eval_id','eval_ood']:
            q=dict(llama['audit_minus_none'][split][a]['error']);q['mean']=-q['mean'];values.append(interval(q,3))
        lr.append([a,'64',f'{st.mean(steps):.1f} [{min(steps)}--{max(steps)}]']+values)
    table('llama_four_round_audit_contrast_table.tex',r'Llama-3.2-3B same-arm audited-minus-unaudited accuracy changes at round four (positive favors auditing), with paired 95\% block-$t$ half-widths. All unaudited arms take 64 steps. Auditing spends 16 correctness queries per arm-round and changes weights, positive-weight volume, and actual optimizer effort.','tab:llama-audit-change',
          ['Arm','Queries','Audited steps: mean [range]','ID accuracy change','OOD accuracy change'],lr)
    feasibilityrows=[['Graph / Qwen3-4B',q['block'],f'{100*q["correct"]:.2f}',str(q['truncated']),str(q['target_errors']),str(q['J_exact']),str(q['J_actual']),'exact','64'] for q in feasible]
    if args.check:
        for name,expected,total in [('h100_feasibility_table.tex',feasibilityrows,25)]:
            actual=rows(name);assert len(actual)==total and actual[-len(expected):]==expected,name
        assert [r for r in rows('h100_progress_table.tex') if r[:2]==['Qwen3-4B','Four rounds']]==[
            ['Qwen3-4B','Four rounds',p,'5','40','41'] for p in ['None','B=16']]
        print(json.dumps(dict(status='passed',detailed_tables_checked=len(tables),display_rows_checked=sum(len(q['rows']) for q in tables.values())+len(feasibilityrows)+2,round_one_pool_answers_rejudged=81920,source_hashes_checked=len(sources)),indent=2))
    else:
        report=dict(training=training,evaluations=evaluation,feasibility=feasible,progress_rows=progress,diagnostic_rows=diagnostics,feasibility_rows=feasibilityrows,tables=tables,source_files=sources,
                    limits=['Exact-length eligibility is from saved certificates, not independent retokenization.','Unaudited held-out validation is count-based; independent pool rejudging is a separate check.','Separate baselines and variable optimizer effort limit causal audit attribution.'])
        (PAPER/'figures/qwen4b_multiround/detailed_results.json').write_text(json.dumps(report,indent=2,sort_keys=True)+'\n')
        for name,rr in [('coverage',progress),('feasibility',feasibilityrows),('training',diagnostics)]:
            print(name+'\n'+'\n'.join(' & '.join(r)+r'\\' for r in rr))

if __name__=='__main__':main()
