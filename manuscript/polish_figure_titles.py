"""Change one-step titles only; remove unused PDF font resources."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pypdfium2 as pdfium
from pypdf import PdfReader, PdfWriter
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

ROOT = Path(__file__).resolve().parent

def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()

def pixels(data):
    doc = pdfium.PdfDocument(data)
    page = doc[0]
    bitmap = page.render(scale=2)
    array = np.array(bitmap.to_pil().convert('RGB'))
    bitmap.close(); page.close(); doc.close()
    return array

def strip_unused_fonts(page):
    stream = page.get_contents()
    fonts = page['/Resources']['/Font'].get_object()
    current, used = None, set()
    for args, op in stream.operations:
        if op == b'Tf': current = str(args[0])
        if op in (b'Tj', b'TJ', b"'", b'"'): used.add(current)
    unused = set(fonts) - used
    for alias in unused: del fonts[alias]
    stream.operations = [(a, o) for a, o in stream.operations if not (o == b'Tf' and str(a[0]) in unused)]
    page.replace_contents(stream)
    return sorted(unused)

def title_only(source, title):
    reader = PdfReader(source)
    writer = PdfWriter()
    page = writer.add_page(reader.pages[0])
    stream = page.get_contents()
    ops = stream.operations
    start = max(i for i, (_, op) in enumerate(ops) if op == b'BT')
    end = next(i for i in range(start + 1, len(ops)) if ops[i][1] == b'ET')
    original = reader.pages[0].extract_text().splitlines()[-1]
    assert original.startswith('graph_unaudited'), original
    assert ops[start-1][1] == b'cm' and len(ops[start-1][0]) == 6
    baseline = float(ops[start-1][0][5])
    assert baseline > float(page.mediabox.height)-25
    assert ops[start+1][1] == b'Tf' and float(ops[start+1][0][1]) == 10
    # Remove only the title's text object; retain all scientific drawing operators.
    stream.operations = ops[:start] + ops[end+1:]
    page.replace_contents(stream)
    overlay = io.BytesIO()
    canvas = Canvas(overlay, pagesize=(float(page.mediabox.width), float(page.mediabox.height)), invariant=1)
    canvas.setFont('ManuscriptTitle', 10)
    canvas.drawCentredString(float(page.mediabox.width)/2, baseline, title)
    canvas.save()
    page.merge_page(PdfReader(overlay).pages[0])
    strip_unused_fonts(page)
    writer.add_metadata({'/Title':title,'/Author':'Anonymous research manuscript'})
    output = io.BytesIO(); writer.write(output); output = output.getvalue()
    before, after = pixels(source.read_bytes()), pixels(output)
    assert before.shape == after.shape
    cut = round((float(page.mediabox.height) - baseline + 4)*2)
    ys, xs = np.where(np.any(before != after, axis=2))
    assert np.array_equal(before[cut:], after[cut:]), ('Plot changed outside title strip', int(ys.min()), int(ys.max()), int(xs.min()), int(xs.max()), len(ys), cut)
    assert 'graph_unaudited' not in PdfReader(io.BytesIO(output)).pages[0].extract_text()
    return output, dict(original_title=original, title=title, unchanged_below_y_pixels=cut, scale=2)

def active_pdfs():
    seen, assets = set(), set()
    def walk(name):
        path = ROOT / name
        if path in seen: return
        seen.add(path)
        text = path.read_text()
        for name in re.findall(r'\\input\{([^}]+)\}', text):
            walk(name if name.endswith('.tex') else name+'.tex')
        assets.update(name for name in re.findall(r'\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}', text) if name.endswith('.pdf'))
    walk('main.tex')
    return sorted(assets)

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path, required=True)
    parser.add_argument('--font', type=Path, required=True)
    args = parser.parse_args()
    pdfmetrics.registerFont(TTFont('ManuscriptTitle', str(args.font)))
    out = ROOT/'figures/one_step_selected_raw'
    provenance = json.loads((out/'provenance.json').read_text())
    report = {'title_revisions':{}, 'unused_font_cleanup':{}}
    for name, entry in provenance['files'].items():
        if not name.endswith('.pdf'): continue
        source = args.code/entry['source_relative_to_code']
        original_hash = entry.get('source_sha256', entry['sha256'])
        assert sha(source) == original_hash
        title = entry['model']+' one-step: '+('projection on h' if entry['metric']=='projection' else 'ID error change against null')
        output, details = title_only(source, title)
        (out/name).write_bytes(output)
        entry.update(source_sha256=original_hash, sha256=sha(out/name), presentation_revision=details)
        report['title_revisions'][name] = details
    provenance['image_transformations'] = 'Model-specific readable titles; all pixels below the title strip are unchanged at 144 dpi. JSON summaries remain byte-identical to the source.'
    (out/'provenance.json').write_text(json.dumps(provenance, indent=2, sort_keys=True)+'\n')
    for name in active_pdfs():
        path = ROOT/name
        original = path.read_bytes()
        writer = PdfWriter(); page = writer.add_page(PdfReader(io.BytesIO(original)).pages[0])
        removed = strip_unused_fonts(page)
        if removed:
            output = io.BytesIO(); writer.write(output); output = output.getvalue()
            assert np.array_equal(pixels(original), pixels(output)), ('Rendering changed', name)
            path.write_bytes(output)
            report['unused_font_cleanup'][name] = dict(removed=removed, before_sha256=hashlib.sha256(original).hexdigest(), after_sha256=sha(path), pixel_equality=True)
    (ROOT/'figure_title_revision_checks.json').write_text(json.dumps(report, indent=2, sort_keys=True)+'\n')
    print('Updated six model-specific titles; all plot pixels below each title are unchanged.')
    print('Removed unused font resources from', len(report['unused_font_cleanup']), 'other PDFs; rendered pixels are unchanged.')

if __name__ == '__main__': main()
