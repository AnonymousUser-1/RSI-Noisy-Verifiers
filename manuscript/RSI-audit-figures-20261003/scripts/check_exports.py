"""Verify the actual PDF/SVG exports and their source-grid geometry."""
from pathlib import Path
import argparse
import hashlib
import json
import re
import xml.etree.ElementTree as ET
import pymupdf

ROOT = Path(__file__).resolve().parents[1]


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('figure_dir', type=Path)
    cli.add_argument('--out', type=Path, required=True)
    args = cli.parse_args()
    if args.out.exists():
        cli.error('Output exists; choose a new evidence file.')
    results = []
    expected = {'audit_map': ('QuadMesh', 12221), 'audit_detail': ('QuadMesh', 4941),
                'audit_surface': ('Poly3DCollection', 12000)}
    for name, (prefix, count) in expected.items():
        svg = args.figure_dir / (name + '.svg'); pdf = args.figure_dir / (name + '.pdf')
        root = ET.fromstring(svg.read_bytes()); ns = {'s': 'http://www.w3.org/2000/svg'}
        for el in root.iter():
            tag = el.tag.split('}')[-1]
            if tag in ('script', 'foreignObject', 'image'):
                raise ValueError(f'{name}: unexpected active or raster content {tag}')
            for key, value in el.attrib.items():
                if key.split('}')[-1].startswith('on'):
                    raise ValueError('Event handler in SVG')
                if key.split('}')[-1] in ('href', 'src') and not value.startswith('#'):
                    raise ValueError('External SVG reference')
                if re.search(r'url\(\s*[\'"]?(?!#)[A-Za-z]', value):
                    raise ValueError('External SVG URL')
        groups = [g for g in root.findall('.//s:g', ns) if g.get('id', '').startswith(prefix)]
        counts = [len(g.findall('s:path', ns)) for g in groups]
        if count not in counts:
            raise ValueError(f'{name}: missing full vector mesh ({count}); found {counts}')
        doc = pymupdf.open(pdf)
        if len(doc) != 1 or doc[0].get_images(full=True):
            raise ValueError('PDF must be a single vector-only page')
        page = doc[0]
        size = [page.rect.width * 25.4 / 72, page.rect.height * 25.4 / 72]
        brief = json.loads((ROOT / f'.plotting/briefs/{name}.json').read_text())
        if max(abs(x-y) for x,y in zip(size, brief['figsize_mm'])) > .01:
            raise ValueError('PDF physical size mismatch')
        fonts = []
        for font in page.get_fonts(full=True):
            if not doc.extract_font(font[0])[3]:
                raise ValueError('Unembedded font')
            fonts.append(font[3])
        outside = []
        for block in page.get_text('dict')['blocks']:
            for line in block.get('lines', []):
                for span in line['spans']:
                    if not page.rect.contains(pymupdf.Rect(span['bbox'])):
                        outside.append(span['text'])
        if outside:
            raise ValueError(f'{name}: PDF text outside page: {outside}')
        results.append({'figure': name, 'status': 'passed', 'dimensions_mm': size,
                        'expected_data_polygons': count, 'actual_vector_mesh_counts': counts,
                        'embedded_fonts': fonts, 'raster_images': 0, 'text_outside_page': [],
                        'sha256': {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in (pdf, svg)}})
        args.out.parent.mkdir(parents=True, exist_ok=True)
        page.get_pixmap(matrix=pymupdf.Matrix(1.5, 1.5), alpha=False).save(args.out.parent / f'{name}_pdf_preview.png')
        doc.close()
    with args.out.open('x', encoding='utf-8') as f:
        json.dump({'status': 'passed', 'figures': results}, f, indent=2)
        f.write('\n')
    print(json.dumps({'status': 'passed', 'figures': [r['figure'] for r in results],
                      'data_polygons': [r['expected_data_polygons'] for r in results]}, indent=2))


if __name__ == '__main__':
    main()
