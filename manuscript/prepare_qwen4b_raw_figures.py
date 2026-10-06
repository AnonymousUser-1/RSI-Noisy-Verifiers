#!/usr/bin/env python3
"""Preserve original Qwen3-4B exports and complete their six-group layout.

Eight PNGs are copied unchanged; four joint-view PNGs use plot_multiround.py.
Requires matplotlib/scipy only for regeneration, not the verification script.
"""
import argparse
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import shutil
import struct
import sys
import tempfile

PAPER = Path(__file__).resolve().parent
OUT = PAPER/'figures/qwen4b_four_round_raw'

def sha(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--code',type=Path,required=True)
    ap.add_argument('--plot-deps',type=Path)
    args=ap.parse_args();code=args.code.resolve()
    if args.plot_deps:sys.path.insert(0,str(args.plot_deps.resolve()))
    os.environ.setdefault('MPLCONFIGDIR',tempfile.mkdtemp(prefix='rsi-qwen4b-mpl-'))
    script=code/'plot_multiround.py'
    spec=importlib.util.spec_from_file_location('original_plot',script)
    plot=importlib.util.module_from_spec(spec);spec.loader.exec_module(plot)
    OUT.mkdir(parents=True,exist_ok=True)
    roots={
        'none':code/'experiments/outputs/2026-10-05-qwen3-4b-multiround-unaudited',
        'audit':code/'experiments/outputs/2026-10-05-qwen3-4b-graph-audit-b16'}
    runs={'none':roots['none']/'graph_unaudited_qwen3-4b/out','audit':roots['audit']/'out'}
    labels={'unaudited':'none','audited B=16':'audit'}
    files={}
    for split in ['eval_id','eval_ood']:
        for policy,prefix in [('none','unaudited'),('audit','audited')]:
            for suffix in ['pass1.png','by_difficulty.png','summary.json','summary.csv']:
                source=next((roots[policy]/'figures').glob('*'+split+'_'+suffix))
                name=prefix+'_'+split+'_'+suffix
                shutil.copy2(source,OUT/name)
                summary=json.loads(next((roots[policy]/'figures').glob('*'+split+'_summary.json')).read_text())
                files[name]=dict(kind='original_export',source_relative_to_code=str(source.relative_to(code)),
                    sha256=sha(source),split=split,policies={label:policy for label in summary['pass1']})
        loaded={label:plot.load(runs[policy],split) for label,policy in labels.items()}
        assert len({json.dumps({k:v[0][k] for k in ['split','question_ids','decoding','base']},sort_keys=True)
                    for v in loaded.values()})==1
        prefix=OUT/('comparison_'+split)
        plot.write_tables(plot.plot(loaded,prefix,split),prefix)
        for suffix in ['pass1.png','by_difficulty.png','summary.json','summary.csv']:
            name='comparison_'+split+'_'+suffix
            files[name]=dict(kind='regenerated_joint_view',sha256=sha(OUT/name),split=split,policies=labels)
    for name,entry in files.items():
        if name.endswith('.png'):
            entry['size_pixels']=list(struct.unpack('>II',(OUT/name).read_bytes()[16:24]))
    import matplotlib,scipy
    provenance=dict(files=files,plot_script_relative_to_code='plot_multiround.py',
        plot_script_sha256=sha(script),validated_results_sha256=sha(PAPER/'figures/qwen4b_multiround/validated_results.json'),
        block_count=5,rounds=[0,1,2,3,4],questions={'eval_id':2000,'eval_ood':1000},
        generated_with={'matplotlib':matplotlib.__version__,'scipy':scipy.__version__},
        chart_contract=dict(surface='LaTeX appendix PNG panels',family='Discrete checkpoint lines with pointwise intervals',
            question='Display both policies, arm contrasts, error composition and strata for the same four rounds',
            native_exports='Eight unchanged PNGs',generated_views='Four joint PNGs, same plot script and saved checkpoints',
            palette='Original blue R/red S; joint policies distinguished by line style and markers',
            footprint='Two full-width split panels per figure; six groups',
            limitations='Separate policy baselines; descriptive intervals and original panel-specific axes retained'))
    (OUT/'provenance.json').write_text(json.dumps(provenance,indent=2,sort_keys=True)+'\n')
    print(json.dumps(dict(original_pngs=8,joint_pngs_generated=4,groups=6),indent=2))

if __name__=='__main__':main()
