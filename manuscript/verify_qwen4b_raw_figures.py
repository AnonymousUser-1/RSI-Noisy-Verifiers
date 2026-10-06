#!/usr/bin/env python3
"""Check native and joint Qwen3-4B figures against checkpoint statistics."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import struct

PAPER=Path(__file__).resolve().parent
OUT=PAPER/'figures/qwen4b_four_round_raw'
BLOCKS=[f'b{i:02d}' for i in range(5)]
TCRIT=2.7764451051977987

def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()

def check(code):
    pr=json.loads((OUT/'provenance.json').read_text())
    source=PAPER/'figures/qwen4b_multiround/validated_results.json'
    assert sha(source)==pr['validated_results_sha256']
    assert sha(code/pr['plot_script_relative_to_code'])==pr['plot_script_sha256']
    records=json.loads(source.read_text())
    for rel,expected in records['source_files'].items():
        assert sha(code/rel)==expected,rel
    pngs=originals=joints=summaries=entries=csv_rows=0
    for name,entry in pr['files'].items():
        p=OUT/name;assert sha(p)==entry['sha256'],name
        if entry['kind']=='original_export':
            assert sha(code/entry['source_relative_to_code'])==entry['sha256']
        if name.endswith('.png'):
            header=p.read_bytes()[:24]
            assert header[:8]==b'\x89PNG\r\n\x1a\n'
            assert list(struct.unpack('>II',header[16:24]))==entry['size_pixels']
            pngs+=1;originals+=entry['kind']=='original_export';joints+=entry['kind']=='regenerated_joint_view'
            continue
        if name.endswith('.csv'):continue
        raw=json.loads(p.read_text());split=entry['split'];summaries+=1
        n=2000 if split=='eval_id' else 1000
        counts=[667,667,666] if split=='eval_id' else [334,333,333]
        for label,policy in entry['policies'].items():
            ev=records[('unaudited' if policy=='none' else 'audited')+'_evaluations'][split]
            for row in ev.values():
                assert [row['questions_by_difficulty'][str(i)] for i in range(3)]==counts
            for metric in ['pass1','target_error_rate','difficulty_0','difficulty_1','difficulty_2']:
                if metric.startswith('difficulty_'):
                    d=metric[-1];exported=raw['by_difficulty'][label][d]
                    value=lambda r:r['pass1_by_difficulty'][d]
                else:
                    exported=raw[metric][label]
                    value=(lambda r:1-sum(r['error_counts'].values())/n) if metric=='pass1' else (lambda r:r['error_counts'].get('nonshortest',0)/n)
                for t in range(5):
                    values={a:[value(ev['base'] if t==0 else ev[f'{b}_{a}_{t}']) for b in BLOCKS] for a in ['R','S']}
                    values['S-R']=[s-r for s,r in zip(values['S'],values['R'])]
                    for series,vals in values.items():
                        q=(exported['s_minus_r'] if series=='S-R' else exported['arms'][series])[str(t)]
                        assert q['n']==5 and q['blocks']==BLOCKS
                        m=sum(vals)/5;sd=math.sqrt(sum((v-m)**2 for v in vals)/4)
                        for field,expected in [('mean',m),('sd',sd),('half_width',TCRIT*sd/math.sqrt(5))]:
                            assert abs(q[field]-expected)<1e-12,(name,metric,t,series,field)
                        entries+=1
        csvfile=p.with_suffix('.csv')
        rows=list(csv.DictReader(csvfile.open()))
        assert len(rows)==75*len(entry['policies'])
        for r in rows:
            group=raw[r['metric']][r['run']] if r['difficulty']=='all' else raw['by_difficulty'][r['run']][r['difficulty']]
            q=group['s_minus_r'][r['round']] if r['series']=='S-R' else group['arms'][r['series']][r['round']]
            for col,field in [('mean','mean'),('sd','sd'),('n','n'),('ci95_half_width','half_width')]:
                assert math.isclose(float(r[col]),q[field],abs_tol=1e-12)
            csv_rows+=1
    assert (pngs,originals,joints,summaries,entries,csv_rows)==(12,8,4,6,600,600)
    tex=(PAPER/'qwen4b_four_round_raw_figures.tex').read_text()
    assert tex.count(r'\includegraphics')==12 and tex.count(r'\begin{figure}')==6
    assert all('qwen4b_four_round_raw/'+name in tex for name in pr['files'] if name.endswith('.png'))
    return dict(status='passed',unchanged_original_pngs=originals,regenerated_joint_pngs=joints,
        figure_groups=6,plotting_summaries=summaries,summary_entries_checked=entries,
        numeric_fields_checked=3*entries,csv_rows_checked=csv_rows,
        scope='Eight original image hashes, four joint-image hashes and all plotted aggregate/stratum statistics vs checkpoint records')

if __name__=='__main__':
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument('--code',type=Path,required=True)
    print(json.dumps(check(ap.parse_args().code.resolve()),indent=2))
