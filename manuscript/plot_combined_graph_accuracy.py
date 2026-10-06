#!/usr/bin/env python3
"""Render a combined vector chart from validated summaries (requires reportlab).

Run from any directory. Means and intervals are never recalculated.
"""
import hashlib
import json
from pathlib import Path
import importlib.util
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parent
SOURCES = [('Llama-3.2-3B', 'figures/h100_runs/validated_results.json', .75),
           ('Qwen3-4B', 'figures/qwen4b_multiround/validated_results.json', 1.)]
COLORS = {'R': '#127F83', 'S': '#805677'}
INK = '#38434A'

def main():
    dirs = [Path('/usr/share/fonts/truetype/dejavu'),
            Path('/Applications/LibreOffice.app/Contents/Resources/fonts/truetype')]
    mpl = importlib.util.find_spec('matplotlib')
    if mpl and mpl.origin:
        dirs.insert(0, Path(mpl.origin).parent / 'mpl-data/fonts/ttf')
    fonts = [(d/'DejaVuSans.ttf', d/'DejaVuSans-Bold.ttf') for d in dirs]
    fonts.append((Path('/System/Library/Fonts/Supplemental/Arial.ttf'),
                  Path('/System/Library/Fonts/Supplemental/Arial Bold.ttf')))
    regular, bold = next((a,b) for a,b in fonts if a.is_file() and b.is_file())
    for name, path in [('GraphSans',regular), ('GraphBold',bold)]:
        pdfmetrics.registerFont(TTFont(name,str(path)))
    out = ROOT/'figures/combined_graph_accuracy'
    out.mkdir(parents=True, exist_ok=True)
    c = canvas.Canvas(str(out/'graph_accuracy.pdf'), pagesize=(486,284), invariant=1)
    c.setTitle('Llama-3.2-3B and Qwen3-4B: four-round graph accuracy')
    c.setAuthor('Anonymous research manuscript')
    def mark(x,y,arm,policy):
        c.setFillColor(COLORS[arm] if policy=='none' else '#ffffff')
        c.setLineWidth(1.15)
        c.circle(x,y,2.3,stroke=1,fill=1)
    for j,(arm,policy) in enumerate([('R','none'),('S','none'),('R','audit'),('S','audit')]):
        x,y = 34+116*j,274
        c.setStrokeColor(COLORS[arm]); c.setLineWidth(1.6)
        c.setDash(() if arm=='R' else (5,2.5)); c.line(x,y,x+16,y); c.setDash(())
        mark(x+8,y,arm,policy)
        c.setFont('GraphSans',8.5); c.setFillColor(INK)
        c.drawString(x+21,y-2.6,f'{arm}, {"unaudited" if policy=="none" else "B=16"}')
    points, hashes = [], {}
    for row,(model,filename,upper) in enumerate(SOURCES):
        raw = (ROOT/filename).read_bytes()
        hashes[filename] = hashlib.sha256(raw).hexdigest()
        summary = json.loads(raw)['summary']
        for col,split in enumerate(['eval_id','eval_ood']):
            x0,y0,w,h = 43+248*col,158-123*row,184,82
            def pt(t,v):
                return x0+w*t/4,y0+h*v/upper
            c.setFillColor(INK); c.setFont('GraphBold',10)
            c.drawCentredString(x0+w/2,y0+h+9,
                f'({"abcd"[2*row+col]}) {model}: {"ID" if col==0 else "OOD"}')
            c.setFont('GraphSans',8.5); c.setStrokeColor(INK); c.setLineWidth(.65)
            for v in [0,.25,.5,.75]+([1] if upper==1 else []):
                _,y = pt(0,v)
                c.line(x0-3,y,x0,y); c.drawRightString(x0-5,y-2.6,f'{v:g}')
            c.line(x0,y0,x0+w,y0); c.line(x0,y0,x0,y0+h)
            for t in range(5):
                x,_ = pt(t,0)
                c.line(x,y0,x,y0-3); c.drawCentredString(x,y0-13,str(t))
            c.saveState(); c.translate(x0-29,y0+h/2); c.rotate(90)
            c.setFont('GraphSans',9.5); c.drawCentredString(0,0,'Greedy Pass@1'); c.restoreState()
            for policy in ['none','audit']:
                for arm in ['R','S']:
                    series=[summary[split][policy]['pass1'][arm][str(t)] for t in range(5)]
                    c.setStrokeColor(COLORS[arm]); c.setLineWidth(1.6)
                    c.setDash(() if arm=='R' else (5,2.5))
                    p=c.beginPath()
                    for t,q in enumerate(series):
                        x,y=pt(t,q['mean'])
                        (p.moveTo if t==0 else p.lineTo)(x,y)
                    c.drawPath(p); c.setDash(())
                    for t,q in enumerate(series):
                        assert 0<=q['mean']<=upper
                        x,y=pt(t,q['mean'])
                        if t:
                            assert q['n']==5
                            assert abs(q['low']-(q['mean']-q['half_width']))<1e-12
                            assert abs(q['high']-(q['mean']+q['half_width']))<1e-12
                            assert 0<=q['low']<=q['high']<=upper
                            _,lo=pt(t,q['low']); _,hi=pt(t,q['high'])
                            c.setLineWidth(.65); c.line(x,lo,x,hi)
                            c.line(x-2.5,lo,x+2.5,lo); c.line(x-2.5,hi,x+2.5,hi)
                        mark(x,y,arm,policy)
                        points.append(dict(model=model,split=split,policy=policy,arm=arm,round=t,
                            mean=q['mean'],low=q.get('low') if t else None,high=q.get('high') if t else None))
    c.setFillColor(INK); c.setFont('GraphSans',9.5)
    c.drawCentredString(259,4,'Self-training round')
    c.showPage(); c.save()
    assert len(points)==80
    (out/'provenance.json').write_text(json.dumps(dict(sources_sha256=hashes,
        plotted_points=points,description='Original means and intervals; shared legend and model/split panel labels.'),indent=2)+'\n')
    print('Rendered 80 unchanged means and 64 original confidence intervals.')

if __name__=='__main__':
    main()
