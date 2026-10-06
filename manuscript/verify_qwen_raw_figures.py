#!/usr/bin/env python3
"""Check unchanged Qwen PNG exports and their four-round plotting summaries."""
import argparse
import hashlib
import json
import math
from pathlib import Path
import struct

PAPER = Path(__file__).resolve().parent
OUT = PAPER / 'figures/qwen_four_round_raw'
BLOCKS = [f'b{i:02d}' for i in range(5)]
TCRIT = 2.7764451051977987


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stats(values):
    mean = sum(values) / len(values)
    sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (len(values) - 1))
    return {'mean': mean, 'sd': sd,
            'half_width': TCRIT * sd / math.sqrt(len(values))}


def check(code=None):
    provenance = json.loads((OUT / 'provenance.json').read_text())
    if code is not None:
        assert digest(code / provenance['plot_script_relative_to_code']) == provenance['plot_script_sha256']
    records = json.loads((PAPER / 'figures/graph_extensions/validated_results.json').read_text())['multiround']
    assert digest(PAPER / 'figures/graph_extensions/validated_results.json') == provenance['validated_results_sha256']
    pngs = summaries = entries = 0
    for name, source in provenance['files'].items():
        path = OUT / name
        assert digest(path) == source['sha256'], name
        if code is not None:
            assert digest(code / source['source_relative_to_code']) == source['sha256'], name
        if name.endswith('.png'):
            header = path.read_bytes()[:24]
            assert header[:8] == b'\x89PNG\r\n\x1a\n'
            assert list(struct.unpack('>II', header[16:24])) == source['size_pixels'], name
            pngs += 1
            continue
        raw = json.loads(path.read_text())
        split = source['split']
        n = 2000 if split == 'eval_id' else 1000
        summaries += 1
        for label, policy in source['policies'].items():
            ev = records['evaluations'][policy][split]
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
                assert set(exported['s_minus_r']) == {str(t) for t in range(5)}, (name, metric)
                for arm in ['R', 'S']:
                    assert set(exported['arms'][arm]) == {str(t) for t in range(5)}, (name, metric, arm)
                for t in range(5):
                    values = {a: [value(ev['base'] if t == 0 else ev[f'{b}_{a}_{t}'])
                                  for b in BLOCKS] for a in ['R', 'S']}
                    values['S-R'] = [s - r for s, r in zip(values['S'], values['R'])]
                    for series in ['R', 'S', 'S-R']:
                        expected = stats(values[series])
                        supplied = (exported['s_minus_r'] if series == 'S-R'
                                    else exported['arms'][series])[str(t)]
                        assert supplied['n'] == 5 and supplied['blocks'] == BLOCKS
                        for field, number in expected.items():
                            assert abs(supplied[field] - number) < 1e-12, (name, metric, t, series, field)
                        entries += 1
    assert (pngs, summaries, entries) == (12, 6, 600), (pngs, summaries, entries)
    tex = (PAPER / 'qwen_four_round_raw_figures.tex').read_text()
    assert tex.count('\\includegraphics') == 12 and tex.count('\\begin{figure}') == 6
    assert all('qwen_four_round_raw/' + name in tex for name in provenance['files'] if name.endswith('.png'))
    return {'status': 'passed', 'unchanged_pngs': pngs, 'plotting_summaries': summaries,
            'summary_entries_checked': entries, 'numeric_fields_checked': 3 * entries,
            'source_byte_equality': code is not None,
            'scope': 'Image hashes and exported four-round statistics vs saved count/scalar records; no raw-answer rejudging.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code', type=Path)
    print(json.dumps(check(parser.parse_args().code), indent=2))
