#!/usr/bin/env python3
"""Compose verified vector panels into Figure 1, without numerical recomputation.

Requires pypdf and reportlab. From manuscript/:
    python scripts/compose_population_overview.py --overwrite
Original source PDFs and numerical CSVs are left untouched.
"""
from __future__ import annotations
import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path
import sys
from pypdf import PdfReader, PdfWriter, Transformation
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen import canvas


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fonts():
    directories = [Path('/usr/share/fonts/truetype/dejavu'),
                   Path('/Applications/LibreOffice.app/Contents/Resources/fonts/truetype'),
                   Path(sys.prefix).parent / 'native/libreoffice-headless/libreoffice/'
                   'LibreOfficeDev.app/Contents/Resources/fonts/truetype']
    mpl = importlib.util.find_spec('matplotlib')
    if mpl is not None and mpl.origin:
        directories.insert(0, Path(mpl.origin).parent / 'mpl-data/fonts/ttf')
    for directory in directories:
        regular, bold = directory / 'DejaVuSans.ttf', directory / 'DejaVuSans-Bold.ttf'
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont('OverviewSans', str(regular)))
            pdfmetrics.registerFont(TTFont('OverviewSansBold', str(bold)))
            return 'OverviewSans', 'OverviewSansBold', {
                'regular_sha256': digest(regular), 'bold_sha256': digest(bold)}
    return 'Helvetica', 'Helvetica-Bold', {'fallback': 'standard PDF Helvetica'}


def one_page(path):
    reader = PdfReader(path)
    if len(reader.pages) != 1:
        raise ValueError('Expected single-page panel source: ' + str(path))
    return reader.pages[0]


def compose(root, overwrite=False):
    flow = root / 'figures/population_flow/population_flow.pdf'
    audit = root / 'RSI-audit-figures-20261003/figures/final/audit_map.pdf'
    # Panel-tag coordinates below are tied to this delivered source version.
    if digest(audit) != '6ec35b074d51e593aa1eec2b6cfe184f210640b1cbeb820ab356535ac8ea1ef5':
        raise ValueError('Audit-map source changed; recheck panel-tag coordinates before composing')
    validation = json.loads((flow.parent / 'validation.json').read_text())
    if validation['status'] != 'passed':
        raise ValueError('Population source has not passed numerical validation')
    for name, expected in validation['exports'].items():
        if name.endswith(('.csv', '.pdf')) and digest(flow.parent / name) != expected:
            raise ValueError('Population source hash mismatch: ' + name)
    out = root / 'figures/population_overview'
    pdf_path, record_path = out / 'population_overview.pdf', out / 'composition.json'
    if not overwrite and (pdf_path.exists() or record_path.exists()):
        raise FileExistsError('Use --overwrite to replace named composition exports')
    out.mkdir(parents=True, exist_ok=True)
    regular, bold, font_record = fonts()
    audit_page, flow_page = one_page(audit), one_page(flow)
    # Cover only old panel tags, leaving axes, ticks, and plot data untouched.
    masks = [(3.2269816211, 203.9205505689, '(c)'),
             (314.8976770462, 203.9205505689, '(d)')]
    label_buffer = io.BytesIO()
    overlay = canvas.Canvas(label_buffer, pagesize=tuple(audit_page.mediabox[2:]), invariant=1)
    for x, y, label in masks:
        overlay.setFillColor('white')
        overlay.rect(x - 1, y - 1.5, 21, 13, stroke=0, fill=1)
        overlay.setFillColor('#000000')
        overlay.setFont(bold, 10)
        overlay.drawString(x, y, label)
    overlay.save()
    audit_page.merge_page(PdfReader(label_buffer).pages[0])
    width, audit_height = float(audit_page.mediabox.width), float(audit_page.mediabox.height)
    scale = width / float(flow_page.mediabox.width)
    flow_height, gap, heading_space = float(flow_page.mediabox.height) * scale, 24., 18.
    height = audit_height + gap + flow_height + heading_space
    writer = PdfWriter()
    page = writer.add_blank_page(width=width, height=height)
    page.merge_page(audit_page)
    page.merge_transformed_page(flow_page, Transformation().scale(scale).translate(0, audit_height + gap))
    heading_buffer = io.BytesIO()
    headings = canvas.Canvas(heading_buffer, pagesize=(width, height), invariant=1)
    headings.setFont(regular, 9)
    headings.setFillColor('#38434A')
    headings.drawCentredString(width / 2, height - 10, 'Unaudited population dynamics')
    headings.drawCentredString(width / 2, audit_height + 13, 'Targeted auditing: initial-update effect')
    headings.save()
    page.merge_page(PdfReader(heading_buffer).pages[0])
    writer.add_metadata({'/Title': 'Population learning dynamics and targeted auditing',
                         '/Subject': 'Distinct population flow and one-step regimes'})
    with pdf_path.open('wb') as stream:
        writer.write(stream)
    record = {'source_hashes': {str(flow.relative_to(root)): digest(flow),
                               str(audit.relative_to(root)): digest(audit)},
              'output_sha256': digest(pdf_path), 'footprint_points': [width, height],
              'manuscript_width': '0.7 textwidth', 'font_hashes': font_record,
              'composer_sha256': digest(Path(__file__)),
              'panel_map': {'a': 'unaudited accuracy trajectories', 'b': 'unaudited gradients',
                            'c': 'audited initial-effect map', 'd': 'fixed-L initial-effect slices'},
              'data_modified': False, 'original_audit_package_modified': False,
              'scope': 'No audited trajectories or new empirical measurements',
              'label_overlays_points': masks}
    with record_path.open('w', encoding='utf-8') as stream:
        json.dump(record, stream, indent=2)
        stream.write('\n')
    print(json.dumps(record, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manuscript-dir', type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    compose(args.manuscript_dir.resolve(), args.overwrite)
