#!/usr/bin/env python3
"""Freeze, run and analyze independent evaluation of B's one-step outputs."""
import argparse
from rsi.paired_evaluation import freeze, run, analyze


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('freeze', help='Freeze resolved configuration before formal training')
    p.add_argument('--config', required=True)
    p.add_argument('--data', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--draws', type=int, required=True)
    p.add_argument('--seed', type=int, required=True)
    p.add_argument('--null-repeats', type=int, required=True)
    p.add_argument('--bootstrap', type=int, default=2000)
    p.add_argument('--bootstrap-seed', type=int, default=2027)
    p = sub.add_parser('run')
    p.add_argument('--protocol', required=True)
    p.add_argument('--handoff', required=True)
    p.add_argument('--out', required=True)
    p = sub.add_parser('analyze')
    p.add_argument('--evaluation', required=True)
    p.add_argument('--out', required=True)
    p.add_argument('--include-demo', action='store_true')
    a = parser.parse_args()
    if a.command == 'freeze':
        freeze(a.config,a.data,a.out,a.draws,a.seed,a.null_repeats,a.bootstrap,a.bootstrap_seed)
    elif a.command == 'run':
        run(a.protocol,a.handoff,a.out)
    else:
        analyze(a.evaluation,a.out,a.include_demo)


if __name__ == '__main__':
    main()
