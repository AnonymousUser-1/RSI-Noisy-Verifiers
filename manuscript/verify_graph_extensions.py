#!/usr/bin/env python3
"""Check new rendered tables and independently recount paired accuracy metrics."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import contextlib
import io
import os
import sys
from table_typography import canonical_rows

PAPER=Path(__file__).resolve().parent
OUT=PAPER/'figures/graph_extensions'
BLOCKS=[f'b{i:02d}' for i in range(5)]
TCRIT=2.7764451051977987


def paired(values, expected):
    # Explicit sum-of-squares path, independent of the aggregation helper.
    mean=sum(values)/5
    variance=sum((x-mean)**2 for x in values)/4
    half=TCRIT*math.sqrt(variance/5)
    for k,v in [('mean',mean),('half_width',half),('low',mean-half),('high',mean+half)]:
        assert abs(v-expected[k])<1e-12,(k,v,expected[k])


def verify():
    data=json.loads((OUT/'validated_results.json').read_text())
    contract=json.loads((OUT/'table_contract.json').read_text())
    cells=0
    for filename,q in contract.items():
        text=(PAPER/filename).read_text()
        body=text.split(r'\midrule',1)[1].split(r'\bottomrule',1)[0]
        rows=[[c.strip() for c in line[:-2].split('&')]
              for line in body.splitlines() if '&' in line and line.endswith('\\\\')]
        assert canonical_rows(filename,rows)==canonical_rows(filename,q['rows']),filename
        cells+=sum(len(re.findall(r'[-+]?(?:\d+\.\d+|\d+)',c)) for r in rows for c in r)
    # Reconstruct accuracy from error counts, not the exported pass1 field.
    multi=data['multiround']
    for p in ['none','audit']:
        for split,n in [('eval_id',2000),('eval_ood',1000)]:
            for t in range(1,5):
                vals=[]
                targets=[]
                for b in BLOCKS:
                    r=multi['evaluations'][p][split][f'{b}_R_{t}']
                    s=multi['evaluations'][p][split][f'{b}_S_{t}']
                    vals.append((sum(r['error_counts'].values())-sum(s['error_counts'].values()))/n)
                    targets.append((s['error_counts'].get('nonshortest',0)-r['error_counts'].get('nonshortest',0))/n)
                paired(vals,multi['summary'][p][split]['pass1']['S-R'][str(t)])
                paired(targets,multi['summary'][p][split]['target_error_rate']['S-R'][str(t)])
    for model,q in data['one_step'].items():
        for split,n in [('eval_id',2000),('eval_ood',1000)]:
            vals=[]
            for b in BLOCKS:
                r=q['evaluations'][split][f'{b}_R_1'];s=q['evaluations'][split][f'{b}_S_1']
                vals.append((sum(r['error_counts'].values())-sum(s['error_counts'].values()))/n)
            paired(vals,q['summary'][split]['pass1']['S-R']['1'])
        vals=[q['records'][b+'_S']['diagnostic']['main_diagnostic']['h_T_delta_theta']-
              q['records'][b+'_R']['diagnostic']['main_diagnostic']['h_T_delta_theta'] for b in BLOCKS]
        paired(vals,q['diagnostics']['projection']['S-R'])
    return dict(status='passed',tables=len(contract),numeric_tokens_checked=cells,
                independent_contrasts=41,scope='Rendered tables, count-based paired contrasts, fixed-h projections; not omitted raw answers or tensors.')


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code',type=Path)
    ap.add_argument('--record-notebook',action='store_true')
    args=ap.parse_args()
    report=verify()
    if args.code:
        provenance=json.loads((OUT/'provenance.json').read_text())
        for rel,sha in provenance['source_files'].items():
            assert hashlib.sha256((args.code/rel).read_bytes()).hexdigest()==sha,rel
        report['source_hashes_verified']=len(provenance['source_files'])
    if args.record_notebook:
        path=PAPER/'graph_validation.ipynb'
        nb=json.loads(path.read_text())
        assert nb['nbformat']==4 and nb['nbformat_minor']==5
        assert len({c['id'] for c in nb['cells']})==len(nb['cells'])
        namespace={'__name__':'__main__'}
        original=os.getcwd()
        os.chdir(PAPER)
        try:
            count=0
            for c in nb['cells']:
                assert c['cell_type'] in ['code','markdown']
                if c['cell_type']!='code':continue
                count+=1
                output=io.StringIO()
                with contextlib.redirect_stdout(output):
                    exec(compile(''.join(c['source']),str(path)+'#'+c['id'],'exec'),namespace)
                c['execution_count']=count
                c['outputs']=[{'output_type':'stream','name':'stdout','text':output.getvalue().splitlines(keepends=True)}]
        finally:os.chdir(original)
        nb['metadata']['language_info']['version']=sys.version.split()[0]
        nb['metadata']['validation']={'status':'passed','method':'Sequential standard-Python execution, not a Jupyter kernel','code_cells':count}
        path.write_text(json.dumps(nb,indent=1)+'\n')
        report['notebook_code_cells_executed']=count
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
