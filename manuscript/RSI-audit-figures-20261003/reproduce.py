"""Regenerate the audit scan and all three figures into a new directory."""
from pathlib import Path
import argparse
import json
import subprocess
import sys

ROOT = Path(__file__).resolve().parent


def main():
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--out-dir', type=Path, required=True)
    args = cli.parse_args(); out = args.out_dir.resolve()
    if out.exists():
        cli.error('Choose a new output directory.')
    out.mkdir(parents=True)
    def run(*items):
        subprocess.run([sys.executable, '-B', *map(str, items)], cwd=ROOT, check=True)
    run(ROOT / 'scripts/generate_audit_scan.py', '--out-dir', out / 'data')
    original = json.loads((ROOT / 'data/scan-v1/scan_validation.json').read_text())
    current = json.loads((out / 'data/scan_validation.json').read_text())
    if original['data_hashes'] != current['data_hashes']:
        raise ValueError('Regenerated CSV data differ from delivery data')
    run(ROOT / 'scripts/plot_audit_figures.py', '--data-dir', out / 'data', '--out-dir', out / 'figures')
    run(ROOT / 'scripts/check_exports.py', out / 'figures', '--out', out / 'vector_qc.json')
    print(json.dumps({'status': 'passed', 'out_dir': str(out), 'data_byte_identical': True}, indent=2))


if __name__ == '__main__':
    main()
