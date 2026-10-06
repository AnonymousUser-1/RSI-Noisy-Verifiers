#!/usr/bin/env python3
"""Validate and summarize the compact 2026-10-03 H100 snapshot (no training).

python summarize_h100_results.py --snapshot ../experiments/outputs/2026-10-03-h100 --out figures/h100_runs
Requires reportlab for vector figures; numerical aggregation uses the stdlib.
This cannot rejudge omitted raw responses or provide prompt-bootstrap intervals.
"""
import argparse
import csv
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import shutil
import signal
import statistics as st
import sys

from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

BLOCKS = [f"b{i:02d}" for i in range(5)]
ARMS = ("R", "S")
NAMES = {"none": "graph_unaudited_llama3.2-3b", "audit": "graph_audited_llama3.2-3b"}
READ = {}
ROOT = None
TCRIT = None


def read(path):
    def stalled(signum, frame):
        raise TimeoutError('Cloud file still unavailable: '+str(path))
    raw = path.read_bytes()
    READ[str(path.relative_to(ROOT))] = hashlib.sha256(raw).hexdigest()
    return json.loads(raw)


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    n = len(values)
    mean = st.mean(values)
    sd = st.stdev(values) if n > 1 else None
    # Intervals only for the completed, fixed five-block sample.
    half = TCRIT * sd / math.sqrt(n) if n == 5 else None
    return dict(mean=mean, sd=sd, n=n, half_width=half,
                low=mean-half if half is not None else None,
                high=mean+half if half is not None else None,
                minimum=min(values), maximum=max(values))


def close(x, y):
    assert math.isclose(x, y, abs_tol=1e-12, rel_tol=1e-10), (x, y)


def dump_csv(path, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def chart(path, summaries, metrics):
    """Figure-1 theme: teal R, plum S; filled/open markers encode auditing.

    The five measured rounds are discrete checkpoints, not an interpolated
    continuous-time fit. All point locations and uncertainty intervals are
    unchanged by this presentation-only theme.
    """
    directories = [Path("/usr/share/fonts/truetype/dejavu"),
                   Path("/Applications/LibreOffice.app/Contents/Resources/fonts/truetype"),
                   Path(sys.prefix).parent / "native/libreoffice-headless/libreoffice/"
                   "LibreOfficeDev.app/Contents/Resources/fonts/truetype"]
    mpl = importlib.util.find_spec("matplotlib")
    if mpl is not None and mpl.origin:
        directories.insert(0, Path(mpl.origin).parent / "mpl-data/fonts/ttf")
    pairs = [(d / "DejaVuSans.ttf", d / "DejaVuSans-Bold.ttf") for d in directories]
    pairs.append((Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
                  Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")))
    for regular, bold in pairs:
        if regular.is_file() and bold.is_file():
            break
    else:
        raise RuntimeError("An embeddable DejaVu Sans or Arial font is required")
    pdfmetrics.registerFont(TTFont("H100Sans", str(regular)))
    pdfmetrics.registerFont(TTFont("H100Bold", str(bold)))
    c = canvas.Canvas(str(path), pagesize=(486, 184), invariant=1)
    c.setTitle("Llama-3.2-3B graph: four-round greedy evaluation")
    c.setAuthor("Anonymous research manuscript")
    # Exact categorical/neutral tokens of Figure 1's population-dynamics panels.
    colors = {"R": "#127F83", "S": "#805677"}
    ink = "#38434A"
    dashes = {"R": (), "S": (5, 2.5)}
    # One shared legend avoids repeating four labels in both small multiples.
    c.setFont("H100Sans", 8.5)
    for j, (arm, policy) in enumerate((('R', 'none'), ('S', 'none'),
                                      ('R', 'audit'), ('S', 'audit'))):
        x, y = 34 + 116*j, 174
        c.setStrokeColor(colors[arm])
        c.setLineWidth(1.6)
        c.setDash(dashes[arm])
        c.line(x, y, x+16, y)
        c.setDash(())
        c.setFillColor(colors[arm] if policy == "none" else "#ffffff")
        c.setLineWidth(1.15)
        c.circle(x+8, y, 2.3, stroke=1, fill=1)
        c.setFillColor(ink)
        c.drawString(x+21, y-2.6, f"{arm}, {'B=16' if policy=='audit' else 'unaudited'}")
    for i, (split, metric, ylim, ticks, title, ylabel) in enumerate(metrics):
        x0, y0, width, height = 43+248*i, 35, 184, 113
        def pt(t, v):
            return x0+width*t/4, y0+height*(v-ylim[0])/(ylim[1]-ylim[0])
        c.setFont("H100Bold", 10)
        c.setFillColor(ink)
        c.drawString(x0-30, y0+height+7, title[:3])
        c.setFont("H100Sans", 9.5)
        c.drawCentredString(x0+width/2, 157, title[4:])
        c.setFont("H100Sans", 8.5)
        for v in ticks:
            _, y = pt(0, v)
            c.setStrokeColor(ink)
            c.setLineWidth(.65)
            c.line(x0-3, y, x0, y)
            c.setFillColor(ink)
            c.drawRightString(x0-5, y-2.6, f"{v:g}")
        c.setStrokeColor(ink)
        c.setLineWidth(.65)
        c.line(x0, y0, x0+width, y0)
        c.line(x0, y0, x0, y0+height)
        for t in range(5):
            x, _ = pt(t, ylim[0])
            c.line(x, y0, x, y0-3)
            c.drawCentredString(x, y0-13, str(t))
        c.setFont("H100Sans", 9.5)
        c.drawCentredString(x0+width/2, 6, "Self-training round")
        c.saveState()
        c.translate(x0-29, y0+height/2)
        c.rotate(90)
        c.drawCentredString(0, 0, ylabel)
        c.restoreState()
        for policy in ("none", "audit"):
            for arm in ARMS:
                points = [summaries[split][policy][metric][arm][str(t)] for t in range(5)]
                c.saveState()
                c.setStrokeColor(colors[arm])
                c.setLineWidth(1.6)
                c.setDash(dashes[arm])
                p = c.beginPath()
                for t, q in enumerate(points):
                    x, y = pt(t, q["mean"])
                    (p.moveTo if t == 0 else p.lineTo)(x, y)
                c.drawPath(p)
                c.setDash(())
                c.setLineWidth(.65)
                for t, q in enumerate(points):
                    x, y = pt(t, q["mean"])
                    if t:
                        _, lo = pt(t, q["low"])
                        _, hi = pt(t, q["high"])
                        # Unshifted observations; caps differentiate overlapping intervals.
                        c.line(x, lo, x, hi)
                        c.line(x-2.5, lo, x+2.5, lo)
                        c.line(x-2.5, hi, x+2.5, hi)
                    c.setFillColor(colors[arm] if policy == "none" else "#ffffff")
                    c.setLineWidth(1.15)
                    c.circle(x, y, 2.3, stroke=1, fill=1)
                    c.setLineWidth(.65)
                c.restoreState()
    c.showPage()
    c.save()
    return {"family": "DejaVu Sans" if regular.name == "DejaVuSans.ttf" else "Arial",
            "regular_sha256": digest(regular), "bold_sha256": digest(bold)}


def main():
    global ROOT, TCRIT
    a = argparse.ArgumentParser(description=__doc__)
    a.add_argument("--snapshot", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    args = a.parse_args()
    ROOT = args.snapshot.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    # Recover and cross-check scipy's df=4 critical value from the saved plot exports.
    source = read(ROOT/"figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_eval_id_summary.json")
    intervals = []
    for metric in ('pass1', 'target_error_rate'):
        for s in source[metric].values():
            for arm in ARMS:
                intervals.extend(q for t, q in s['arms'][arm].items() if t != '0')
    crits = [q['half_width']*math.sqrt(q['n'])/q['sd'] for q in intervals if q['sd']]
    TCRIT = st.mean(crits)
    assert all(q['n']==5 for q in intervals)
    assert all(abs(v-TCRIT)<1e-11 for v in crits)
    assert 2.7764<TCRIT<2.7765
    data, protocols, training, matches, progress, feasibility = {}, {}, {}, {}, [], []
    for policy, name in NAMES.items():
        print('Checking completed run:', name, flush=True)
        e = ROOT/name
        exp = read(e/'out/experiment.json')
        assert exp['status']=='complete' and exp['exit_code']==0 and not exp['DEMO_ONLY']
        assert exp['rounds']>=4 and exp['blocks']==BLOCKS and not exp['stopped']
        assert exp['K']==64 and exp['matching_settings']=={'error_fraction':.25,'token_tolerance':0}
        matches[policy] = read(e/'out/matching/matched_subsets.json')
        training[policy] = {}
        for b in BLOCKS:
            print('  training block', b, flush=True)
            m=matches[policy]['per_block'][b]['audit']
            for key in ('K_equal','final_TPR_equal','final_FPR_equal','same_correct_ids','same_task_ids',
                        'task_token_counts_within_tolerance','weights_all_one'):
                assert m[key]
            assert m['C_R']==m['C_S']==48 and m['E_R']==m['E_S']==16
            assert m['max_token_mismatch']==0 and m['S_target_hits']==16
            close(m['final_TPR'],48/m['N_plus'])
            close(m['final_FPR'],16/m['N_minus'])
            close(m['yield'],64/(m['N_plus']+m['N_minus']))
            for arm in ARMS:
                if b == 'b00':
                    run=read(e/f'out/{b}/{arm}/run.json')
                    assert run['bindings']['shared_adapter']['parameter_hash']==exp['identity']['shared_adapter']
                rounds={}
                for t in range(1,5):
                    q=read(e/f'out/{b}/{arm}/round_{t:03d}/complete.json')
                    assert q['round']==t and q['block_id']==b and q['branch']==arm
                    assert q['training']['steps']==q['expected_optimizer_steps']
                    assert q['audit_queries']==(16 if policy=='audit' else 0)
                    if policy=='audit':
                        audit=read(e/f'out/{b}/{arm}/round_{t:03d}/audit.json')
                        assert len(audit['labels'])==16 and audit['before']['K']==64
                        assert audit['after']['K']==q['examples']
                        assert q['training']['steps']==math.ceil(audit['after']['positive_weight_examples']/4)
                        q['composition']=audit['after']
                    else:
                        assert q['examples']==64 and q['training']['steps']==16
                        q['composition']={'K':64,'C':48,'E':16,'positive_weight_examples':64,'effective_sample_size':64.0}
                    rounds[str(t)]=q
                assert (e/f'out/{b}/{arm}/finished.json').is_file()
                assert rounds['4']['cumulative_audit_queries']==(64 if policy=='audit' else 0)
                training[policy][b+'_'+arm]=rounds
        for split, n in [('eval_id',2000),('eval_ood',1000)]:
            print('  evaluation', split, flush=True)
            directory=e/('out/evaluation_greedy_'+split)
            pr=read(directory/'protocol.json')
            assert pr['questions']==n and pr['answers_per_question']==1 and pr['task']=='graph'
            assert pr['decoding']['temperature']==0 and pr['base']['revision']=='0cb88a4f764b7a12671c53f0838cd831a0843b95'
            protocols.setdefault(split,{})[policy]=pr
            rows={}
            for t in range(5):
                for b in BLOCKS:
                    for arm in ARMS:
                        q=read(directory/('round_000.json' if t==0 else f'{b}_{arm}_round_{t:03d}.json'))
                        assert q['round']==t and q['questions']==n
                        close(1-q['pass1'],sum(q['error_counts'].values())/n)
                        close(q['target_error_rate'],q['error_counts'].get('nonshortest',0)/n)
                        assert sum(q['questions_by_difficulty'].values())==n
                        close(q['pass1'],sum(q['pass1_by_difficulty'][d]*k for d,k in q['questions_by_difficulty'].items())/n)
                        rows[f'{b}_{arm}_{t}']=q
            data.setdefault(split,{})[policy]=rows
    assert matches['none']==matches['audit']
    exps=[read(ROOT/name/'out/experiment.json') for name in NAMES.values()]
    for key in ('dataset_hash','shared_adapter','pools','source_hash'):
        assert exps[0]['identity'][key]==exps[1]['identity'][key], key
    for split, per in protocols.items():
        for key in ('split','question_ids','decoding','base','questions','dataset_hash'):
            assert per['none'][key]==per['audit'][key], (split,key)
        assert data[split]['none']['b00_R_0']['pass1']==data[split]['audit']['b00_R_0']['pass1']
    summary, contrasts, flat = {}, {}, []
    for split, per in data.items():
        summary[split]={}
        contrasts[split]={}
        saved=read(ROOT/f'figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_{split}_summary.json')
        for policy, rows in per.items():
            summary[split][policy]={}
            for metric in ('pass1','target_error_rate','format_error_rate','mean_completion_tokens'):
                summary[split][policy][metric]={}
                for series in ('R','S','S-R'):
                    per_round={}
                    for t in range(5):
                        vals=[rows[f'{b}_{series}_{t}'][metric] if series!='S-R' else
                              rows[f'{b}_S_{t}'][metric]-rows[f'{b}_R_{t}'][metric] for b in BLOCKS]
                        q=stats(vals)
                        if t==0:
                            # One common base checkpoint, not five independent base fits.
                            q.update(n=1,sd=None,half_width=None,low=None,high=None)
                        per_round[str(t)]=q
                        flat.append(dict(split=split,policy=policy,metric=metric,series=series,round=t,**q))
                        if metric in ('pass1','target_error_rate') and t:
                            label='audited B=16' if policy=='audit' else 'unaudited'
                            s=saved[metric][label]
                            old=s['s_minus_r'][str(t)] if series=='S-R' else s['arms'][series][str(t)]
                            for key in ('mean','sd','half_width'): close(q[key],old[key])
                    summary[split][policy][metric][series]=per_round
        for arm in ARMS:
            contrasts[split][arm]={}
            for metric in ('error','target_error_rate'):
                vals=[]
                for b in BLOCKS:
                    na=per['none'][f'{b}_{arm}_4']; au=per['audit'][f'{b}_{arm}_4']
                    vals.append(na['pass1']-au['pass1'] if metric=='error' else au[metric]-na[metric])
                contrasts[split][arm][metric]=stats(vals)
    for e in sorted(ROOT.iterdir()):
        if not (e/'out/experiment.json').exists(): continue
        exp=read(e/'out/experiment.json')
        complete=list((e/'out').glob('b*/*/round_*/complete.json'))
        progress.append(dict(experiment=e.name,completed_arm_rounds=len(complete),
                             finished_arms=len(list((e/'out').glob('b*/*/finished.json'))),
                             evaluated_ID=len(list((e/'out/evaluation_greedy_eval_id').glob('*round_*.json'))),
                             evaluated_OOD=len(list((e/'out/evaluation_greedy_eval_ood').glob('*round_*.json'))),
                             status=exp['status']))
    partial=[]
    for q in progress:
        if q['experiment'] in NAMES.values(): continue
        e=ROOT/q['experiment']
        for t in range(1,5):
            for arm in ARMS:
                rows=[read(p) for p in sorted((e/'out').glob(f'b*/{arm}/round_{t:03d}/complete.json'))]
                if rows:
                    partial.append(dict(experiment=e.name,round=t,arm=arm,n=len(rows),
                      h_projection=st.mean(r['training']['h_T_delta_theta'] for r in rows),
                      displacement_norm=st.mean(r['training']['delta_theta_norm'] for r in rows),
                      loss=st.mean(r['training']['mean_loss'] for r in rows)))
    for name in ('graph_unaudited','graph_unaudited_llama3.2-3b','arithmetic_unaudited_llama3.2-3b','arithmetic_unaudited_llama3.2-1b'):
        r=read(ROOT/f'reports/round1_options_{name}.json')
        actual=next(o for o in r['options'] if o['is_config'])
        strict=next(o for o in r['options'] if o['error_fraction']==.25 and o['token_tolerance']==0)
        for b,p in r['pools'].items():
            assert sum(p['labels'].values())==p['answers']==16384
            close(p['correct'],p['labels']['correct']/p['answers'])
            feasibility.append(dict(experiment=name,block=b,**p,
                J_actual=actual['per_block'][b]['prompts_eligible_J_E'],
                J_exact=strict['per_block'][b]['prompts_eligible_J_E'],
                tolerance=r['config_matching']['token_tolerance'],K=actual['K']))
    font=chart(out/'graph_accuracy.pdf',summary,[
      ('eval_id','pass1',(0,.75),(0,.25,.5,.75),'(a) ID: 2,000 questions','Greedy Pass@1'),
      ('eval_ood','pass1',(0,.75),(0,.25,.5,.75),'(b) OOD: 1,000 questions','Greedy Pass@1')])
    chart(out/'graph_target_errors.pdf',summary,[
      ('eval_id','target_error_rate',(0,.3),(0,.1,.2,.3),'(a) ID target errors','Fraction of all answers'),
      ('eval_ood','target_error_rate',(0,.3),(0,.1,.2,.3),'(b) OOD target errors','Fraction of all answers')])
    copied=[]
    for split in ('eval_id','eval_ood'):
        for kind in ('pass1','by_difficulty'):
            p=ROOT/f'figures/graph_unaudited_llama3.2-3b_vs_graph_audited_llama3.2-3b_rounds0-4_{split}_{kind}.png'
            READ[str(p.relative_to(ROOT))]=digest(p)
            dest=out/f'original_{split}_{kind}.png'
            shutil.copyfile(p,dest)
            copied.append(dest.name)
    dump_csv(out/'round_summary.csv',flat)
    dump_csv(out/'snapshot_progress.csv',progress)
    dump_csv(out/'partial_training.csv',partial)
    result=dict(snapshot='2026-10-03 16:06 UTC',completed_model='meta-llama/Llama-3.2-3B-Instruct',
                protocols=protocols,summary=summary,evaluations=data,audit_minus_none=contrasts,training=training,
                matching=matches['none'],progress=progress,feasibility=feasibility,partial_training=partial)
    (out/'validated_results.json').write_text(json.dumps(result,indent=2,sort_keys=True)+'\n')
    manifest=dict(snapshot_root=str(ROOT),snapshot='2026-10-03 16:06 UTC',
        tcritical_df4=TCRIT,source_files=READ,script_sha256=digest(Path(__file__)),font=font,
        checks='Completion, matching identities and rates, paired pool/adapter/data bindings, split/decoding equality, error-count identities, difficulty aggregation, exported means/SD/CIs, steps/queries/composition.',
        missing=['Raw candidate/evaluation responses','training rows and weight vectors','adapter weights','reference h.pt','end-to-end cost ledger'],
        uncertainty='Pointwise descriptive 95% Student-t intervals over five paired blocks; fixed evaluation prompts; no multiplicity adjustment, no prompt bootstrap, no confirmatory claim.',
        chart_contract=dict(question='How do R/S accuracy and target errors evolve, with and without budgeted auditing?',
          palette={'R':'#127F83','S':'#805677','axes':'#38434A'},
          noncolor={'R':'solid','S':'dashed','none':'filled circles','audit':'open circles'},
          footprint_points=[486,184],manuscript_width='0.7 textwidth',
          style_reference='Figure 1 population-dynamics panels',
          renderer='ReportLab vector PDF',shared_base_n=1),
        copied_original_figures=copied,
        exports={p.name:digest(p) for p in out.iterdir() if p.is_file() and p.name!='provenance.json'})
    (out/'provenance.json').write_text(json.dumps(manifest,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(status='validated',source_files=len(READ),critical=TCRIT,
      endpoint={s:{p:{a:summary[s][p]['pass1'][a]['4']['mean'] for a in ARMS} for p in NAMES} for s in summary},
      audit_error_changes=contrasts,progress=progress),indent=2))


if __name__=='__main__':
    main()
