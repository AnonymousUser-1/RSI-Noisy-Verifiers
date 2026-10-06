#!/usr/bin/env python3
"""Verify original Llama four-round and selected one-step plotting exports."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

from pypdf import PdfReader

PAPER = Path(__file__).resolve().parent
BLOCKS = [f'b{i:02d}' for i in range(5)]
TCRIT = 2.7764451051977987


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    mean = sum(values) / 5
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / 4)
    return dict(mean=mean, sd=sd, half_width=TCRIT * sd / math.sqrt(5))


def close(a, b, context):
    assert abs(a - b) < 1e-12, (context, a, b)


def check(code=None):
    pngs = pdfs = summaries = entries = one_fields = 0
    for folder, data_file in [('llama_four_round_raw', 'h100_runs'),
                              ('one_step_selected_raw', 'graph_extensions')]:
        out = PAPER / 'figures' / folder
        provenance = json.loads((out / 'provenance.json').read_text())
        data_path = PAPER / 'figures' / data_file / 'validated_results.json'
        assert digest(data_path) == provenance['validated_results_sha256']
        data = json.loads(data_path.read_text())
        if code is not None:
            assert digest(code / provenance['plot_script_relative_to_code']) == provenance['plot_script_sha256']
        for name, source in provenance['files'].items():
            path = out / name
            assert digest(path) == source['sha256'], name
            if code is not None:
                assert digest(code / source['source_relative_to_code']) == source.get('source_sha256', source['sha256']), name
            if name.endswith('.png'):
                header = path.read_bytes()[:24]
                assert header[:8] == b'\x89PNG\r\n\x1a\n'
                assert list(struct.unpack('>II', header[16:24])) == source['size_pixels']
                pngs += 1
                continue
            if name.endswith('.pdf'):
                reader = PdfReader(path)
                assert len(reader.pages) == 1, name
                if 'presentation_revision' in source:
                    assert source['presentation_revision']['title'] in reader.pages[0].extract_text()
                    assert 'graph_unaudited' not in reader.pages[0].extract_text()
                pdfs += 1
                continue
            raw = json.loads(path.read_text())
            summaries += 1
            if folder == 'llama_four_round_raw':
                split = source['split']
                n = 2000 if split == 'eval_id' else 1000
                for label, policy in source['policies'].items():
                    ev = data['evaluations'][split][policy]
                    for metric in ['pass1', 'target_error_rate', 'difficulty_0', 'difficulty_1', 'difficulty_2']:
                        if metric.startswith('difficulty_'):
                            d = metric[-1]
                            exported = raw['by_difficulty'][label][d]
                            value = lambda r: r['pass1_by_difficulty'][d]
                        else:
                            exported = raw[metric][label]
                            if metric == 'pass1':
                                value = lambda r: 1 - sum(r['error_counts'].values()) / n
                            else:
                                value = lambda r: r['error_counts'].get('nonshortest', 0) / n
                        assert set(exported['s_minus_r']) == {str(t) for t in range(5)}
                        for arm in ['R', 'S']:
                            assert set(exported['arms'][arm]) == {str(t) for t in range(5)}
                        for t in range(5):
                            vals = {a: [value(ev[f'{b}_{a}_{t}']) for b in BLOCKS] for a in ['R', 'S']}
                            vals['S-R'] = [s - r for s, r in zip(vals['S'], vals['R'])]
                            for series in ['R', 'S', 'S-R']:
                                supplied = (exported['s_minus_r'] if series == 'S-R'
                                            else exported['arms'][series])[str(t)]
                                assert supplied['n'] == 5 and supplied['blocks'] == BLOCKS
                                for field, value_ in stats(vals[series]).items():
                                    close(supplied[field], value_, (name, metric, t, series, field))
                                entries += 1
            else:
                q = data['one_step'][source['model']]
                assert set(raw['per_block']) == set(raw['paired']['per_block']) == set(BLOCKS)
                vals = {'R': [], 'S': []}
                for b in BLOCKS:
                    assert set(raw['per_block'][b]) == {'R', 'S'}
                    for a in ['R', 'S']:
                        if source['metric'] == 'error_eval_id':
                            ev = q['evaluations']['eval_id']
                            v = (sum(ev[f'{b}_{a}_1']['error_counts'].values())
                                 - sum(ev['base']['error_counts'].values())) / 2000
                        else:
                            assert source['metric'] == 'projection'
                            v = q['records'][f'{b}_{a}']['diagnostic']['main_diagnostic']['h_T_delta_theta']
                        close(raw['per_block'][b][a], v, (name, b, a))
                        vals[a].append(v)
                        one_fields += 1
                differences = [s - r for s, r in zip(vals['S'], vals['R'])]
                for b, diff in zip(BLOCKS, differences):
                    close(raw['paired']['per_block'][b], diff, (name, b, 'S-R'))
                    one_fields += 1
                supplied = raw['paired']['s_minus_r']
                assert supplied['n'] == 5
                for field, v in stats(differences).items():
                    close(supplied[field], v, (name, field))
                    one_fields += 1
    assert (pngs, pdfs, summaries, entries, one_fields) == (12, 6, 12, 600, 108)
    for filename, image_count, figure_count in [('llama_four_round_raw_figures.tex', 12, 6),
                                               ('one_step_selected_raw_figures.tex', 6, 3)]:
        tex = (PAPER / filename).read_text()
        assert tex.count('\\includegraphics') == image_count
        assert tex.count('\\begin{figure}') == figure_count
    return dict(status='passed', original_llama_pngs=pngs, selected_one_step_pdfs=pdfs,
                summaries=summaries, llama_summary_entries=entries,
                llama_statistic_fields=entries * 3, one_step_numeric_fields=one_fields,
                source_data_byte_equality=code is not None,
                one_step_pdf_titles='Readable model names; source and delivered hashes checked separately',
                scope='Image/PDF hashes and saved count/scalar records; no raw-answer or tensor reconstruction.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path)
    print(json.dumps(check(parser.parse_args().code), indent=2))
