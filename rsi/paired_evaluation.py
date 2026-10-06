"""Independent evaluation of matched one-step experiments (no training)."""
from collections import Counter, defaultdict
from pathlib import Path
import copy
import csv
import re
import time

import numpy as np
from scipy.stats import t

from .common import (digest, environment, file_hash, read_json, read_jsonl,
                     seed_for, source_hash, verify_dataset, write_json, write_jsonl)
from .matching import quotas
from .tasks import TASKS, judge, stratum


def require(condition, message):
    if not condition:
        raise ValueError(message)


def tree_hash(path):
    path = Path(path)
    require(path.is_dir(), 'Missing adapter directory: ' + str(path))
    files = sorted(p for p in path.rglob('*') if p.is_file())
    require(bool(files), 'Empty adapter directory: ' + str(path))
    return digest({str(p.relative_to(path)): file_hash(p) for p in files})


def freeze(config_path, data, output, draws, seed, null_repeats, bootstrap, bootstrap_seed):
    """Freeze before confirmatory training; config must be explicitly resolved."""
    require(not Path(output).exists(), 'Protocol already exists; do not overwrite a freeze')
    cfg = read_json(config_path)
    for key in ('backend', 'model', 'revision', 'device', 'dtype', 'generation'):
        require(key in cfg, 'Resolved config missing ' + key)
    for key in ('batch_size', 'max_new_tokens', 'max_sequence_length', 'temperature', 'top_p', 'top_k'):
        require(key in cfg['generation'], 'Generation config missing ' + key)
    require(cfg['backend'] in ('hf', 'mock'), 'Unknown backend')
    if cfg['backend'] == 'hf':
        require(bool(re.fullmatch('[0-9a-fA-F]{40}', cfg['revision'])), 'Pin the base revision first')
    require(draws > 0 and null_repeats > 0 and bootstrap > 0, 'Counts must be positive')
    root = Path(data).resolve()
    dataset = verify_dataset(root)
    require(dataset['task'] in TASKS, 'Unsupported task')
    seen = set()
    for name in dataset['files']:
        rows = read_jsonl(root / name)
        ids = [r['id'] for r in rows]
        require(len(ids) == len(set(ids)) and not seen.intersection(ids), 'Overlapping dataset instances')
        seen.update(ids)
    # eval_id always; eval_ood when the dataset has one (an imported GSM8K dataset has none).
    splits = [s for s in ('eval_id', 'eval_ood') if s + '.jsonl' in dataset['files']]
    require('eval_id' in splits, 'Missing eval_id')
    for split in splits:
        require(bool(read_jsonl(root / (split + '.jsonl'))), 'Empty ' + split)
    protocol = dict(version=1, config=cfg, data=str(root), dataset_hash=digest(dataset),
                    splits=splits, draws=draws, seed=seed,
                    null_repeats=null_repeats, bootstrap=bootstrap, bootstrap_seed=bootstrap_seed,
                    environment=environment(), source_hash=source_hash(), DEMO_ONLY=cfg['backend'] == 'mock')
    write_json(output, protocol)
    return protocol


def audit_matching(pool, subsets, tasks, token_count, error_fraction=0.25, token_tolerance=0):
    """Rejudge original pool; never trust B's audit booleans or completion_tokens.  C:E and the
    per-prompt token tolerance are the run's (config `matching`; 3:1 and 0 by default)."""
    ids = [c['id'] for c in pool]
    require(len(ids) == len(set(ids)), 'Duplicate candidate IDs')
    by_id = {c['id']: c for c in pool}
    truths = {c['id']: judge(tasks[c['task_id']], c['response']) for c in pool}
    n_pos = sum(x['correct'] for x in truths.values())
    n_neg = len(pool) - n_pos
    selected, metrics = {}, {}
    for arm in ('R', 'S'):
        chosen = subsets[arm]
        require(len(chosen) == len(set(chosen)) and bool(chosen), 'Empty or duplicate subset')
        require(set(chosen) <= set(by_id), 'Unknown candidate ID')
        rows = [by_id[c] for c in chosen]
        require(len({r['task_id'] for r in rows}) == len(rows), 'More than one response per prompt')
        require(all(r.get('weight', 1) == 1 for r in rows), 'Nonuniform weights')
        correct = {c for c in chosen if truths[c]['correct']}
        errors = set(chosen) - correct
        try:
            c_quota, e_quota = quotas(len(rows), error_fraction)
        except ValueError:
            c_quota, e_quota = None, None
        require(len(correct) == c_quota and len(errors) == e_quota,
                'Expected C:E = %s:%s at error_fraction %r' % (c_quota, e_quota, error_fraction))
        lengths = {r['task_id']: token_count(r['response']) for r in rows}
        selected[arm] = (correct, lengths)
        metrics[arm] = dict(K=len(rows), C=len(correct), E=len(errors), N_plus=n_pos, N_minus=n_neg,
                            final_TPR=len(correct)/n_pos, final_FPR=len(errors)/n_neg,
                            precision=len(correct)/len(rows), yield_rate=len(rows)/len(pool),
                            error_counts=dict(Counter(truths[c]['error'] for c in errors)))
    require(selected['R'][0] == selected['S'][0], 'Correct IDs differ')
    r_len, s_len = selected['R'][1], selected['S'][1]
    require(set(r_len) == set(s_len) and all(
        token_tolerance is None or abs(r_len[t] - s_len[t]) <= token_tolerance for t in r_len),
        'Prompt sets or supervised token counts differ (token tolerance %s)' % token_tolerance)
    require(metrics['R']['K'] == metrics['S']['K'], 'K differs')
    return dict(passed=True, denominator='common_original_pool', arms=metrics,
                stages_available=['original_pool', 'final_matched_subset'],
                note='Joint matching has no legacy provisional stage; none is fabricated.')


def run(protocol_path, handoff_path, output):
    """Explicit C-side handoff bridges B's one_step.json without guessing defaults."""
    from .backends import backend
    protocol, handoff = read_json(protocol_path), read_json(handoff_path)
    require(protocol['source_hash'] == source_hash(), 'Code changed since freeze; resolve before evaluation')
    require(digest(verify_dataset(protocol['data'])) == protocol['dataset_hash'], 'Dataset changed')
    root = Path(handoff_path).resolve().parent
    resolve = lambda p: (root / p).resolve()
    blocks = handoff['blocks']
    require(bool(blocks), 'No blocks')
    names = [b['id'] for b in blocks]
    require(len(set(names)) == len(names), 'Duplicate blocks')
    require(all(re.fullmatch('[A-Za-z0-9_-]+', n) for n in names), 'Unsafe block ID')
    require(len({b['seed'] for b in blocks}) == len(blocks), 'Duplicate training seeds')
    cfg = protocol['config']
    # Validate every block before decoding any final-test prompt.
    provenance, audits = {}, {}
    token_model = backend(cfg)
    try:
        if cfg['backend'] == 'mock':
            token_count = lambda s: len(s.split()) + 1
        else:
            require(token_model.tokenizer.eos_token_id is not None, 'Tokenizer has no EOS')
            token_count = lambda s: len(token_model.tokenizer.encode(s, add_special_tokens=False)) + 1
        tasks = {}
        for path in sorted(Path(protocol['data']).glob('train_*.jsonl')):
            tasks.update({r['id']: r for r in read_jsonl(path)})
        for block in blocks:
            require(block['base_model'] == cfg['model'] and block['base_revision'] == cfg['revision'], 'Base mismatch')
            paths = {a: resolve(block['adapters'][a]) for a in ('initial', 'R', 'S')}
            hashes = {a: tree_hash(p) for a, p in paths.items()}
            require(hashes == block['adapter_hashes'], 'Adapter hash mismatch')
            shared = read_json(paths['initial'] / 'shared_adapter.json')
            require(shared['base_revision'] == cfg['revision'] and shared['base_model'] == cfg['model'], 'Shared base mismatch')
            for arm in ('R', 'S'):
                require(paths[arm] != paths['initial'], 'Updated adapter aliases initial')
                require(block['training'][arm]['initial_parameter_hash'] == shared['parameter_hash'], 'Initial parameter mismatch')
                require(block['training'][arm]['completed'] is True and block['training'][arm]['optimizer_steps'] == 1,
                        'Training must complete exactly one update')
            artifacts = {}
            for kind in ('pool', 'matched', 'one_step'):
                p = resolve(block[kind])
                require(file_hash(p) == block[kind + '_hash'], kind + ' hash mismatch')
                artifacts[kind] = str(p)
            record = read_json(artifacts['one_step'])
            require(record['block'] == block['id'] and record['seed'] == block['seed'], 'Wrong one-step block')
            for arm in ('R', 'S'):
                if not protocol['DEMO_ONLY']:
                    require(record['arms'][arm].get('trained') is True and record['arms'][arm].get('steps') == 1,
                            'One-step arm not completed')
            matched = read_json(artifacts['matched'])
            subset = matched['per_block'][block['id']]
            settings = cfg.get('matching') or {}
            audit = audit_matching(read_jsonl(artifacts['pool']), subset, tasks, token_count,
                                   error_fraction=settings.get('error_fraction', 0.25),
                                   token_tolerance=settings.get('token_tolerance', 0))
            require(audit['arms']['R']['K'] in (16, 32, 64), 'K outside frozen ladder')
            require(matched['K'] == audit['arms']['R']['K'], 'Declared K mismatch')
            audits[block['id']] = audit
            provenance[block['id']] = dict(adapters={a: str(p) for a,p in paths.items()}, adapter_hashes=hashes,
                                            artifacts=artifacts)
    finally:
        token_model.close()
    require(len({a['arms']['R']['K'] for a in audits.values()}) == 1, 'Blocks use different K')
    out = Path(output)
    require(not out.exists() or not any(out.iterdir()), 'Use a fresh output directory (no partial-result reuse)')
    manifest = dict(protocol=protocol, protocol_hash=digest(protocol), handoff=handoff,
                    handoff_hash=digest(handoff), provenance=provenance, DEMO_ONLY=protocol['DEMO_ONLY'])
    write_json(out / 'evaluation_manifest.json', manifest)
    write_json(out / 'matching_audit.json', audits)
    outputs = {}
    for block in blocks:
        for arm in ['initial', 'R', 'S'] + ['null_repeat_%d' % i for i in range(1, protocol['null_repeats']+1)]:
            adapter_arm = 'initial' if arm.startswith('null_repeat') else arm
            model = backend(copy.deepcopy(cfg), adapter=provenance[block['id']]['adapters'][adapter_arm])
            start = time.perf_counter()
            try:
                for split in protocol['splits']:
                    tasks = read_jsonl(Path(protocol['data']) / (split + '.jsonl'))
                    by_id = {r['id']: r for r in tasks}
                    repeat = int(arm.rsplit('_',1)[1]) if arm.startswith('null_repeat') else 0
                    seed = seed_for(protocol['seed'], block['id'], split, repeat, 'paired_eval')
                    rows = model.generate(tasks, protocol['draws'], seed)
                    expected = {(t['id'], d) for t in tasks for d in range(protocol['draws'])}
                    require(len(rows) == len(expected) and {(r['task_id'], r['sample']) for r in rows} == expected,
                            'Missing or duplicate generated answers')
                    scored = [dict(r, block=block['id'], arm=arm, split=split, generation_seed=seed,
                                   stratum=stratum(by_id[r['task_id']]),
                                   reached_length_limit=r.get('truncated', r['completion_tokens'] >= cfg['generation']['max_new_tokens']),
                                   **judge(by_id[r['task_id']], r['response'])) for r in rows]
                    path = out / block['id'] / (arm + '_' + split + '.jsonl')
                    write_jsonl(path, scored)
                    outputs[str(path.relative_to(out))] = file_hash(path)
            finally:
                model.close()
            write_json(out / block['id'] / (arm + '_timing.json'), {'seconds': time.perf_counter()-start})
    write_json(out / 'complete.json', {'outputs': outputs, 'manifest_hash': digest(manifest)})
    return manifest


def prompt_errors(rows):
    grouped = defaultdict(dict)
    for row in rows:
        require(row['sample'] not in grouped[row['task_id']], 'Duplicate answer draw')
        grouped[row['task_id']][row['sample']] = float(not row['correct'])
    return {k: float(np.mean(list(v.values()))) for k,v in grouped.items()}


def paired_interval(left, right, repetitions, seed):
    require(set(left) == set(right) and bool(left), 'Unpaired or empty prompts')
    delta = np.array([left[k]-right[k] for k in sorted(left)])
    rng = np.random.default_rng(seed)
    boot = [float(delta[rng.integers(len(delta), size=len(delta))].mean()) for _ in range(repetitions)]
    low, high = np.quantile(boot, [.025, .975])
    return float(delta.mean()), float(low), float(high)


def save_csv(path, rows):
    if rows:
        with Path(path).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=sorted(set().union(*(r.keys() for r in rows))))
            writer.writeheader()
            writer.writerows(rows)


def analyze(evaluation, output, include_demo=False):
    root, out = Path(evaluation), Path(output)
    manifest = read_json(root / 'evaluation_manifest.json')
    require(include_demo or not manifest['DEMO_ONLY'], 'Demo results excluded; use --include-demo explicitly')
    complete = read_json(root / 'complete.json')
    require(complete['manifest_hash'] == digest(manifest), 'Evaluation manifest changed')
    for path, expected in complete['outputs'].items():
        require(file_hash(root / path) == expected, 'Evaluation answers changed')
    protocol = manifest['protocol']
    metrics, contrasts, summaries = [], [], []
    for block in manifest['handoff']['blocks']:
        for split in protocol['splits']:
            arms = ['initial', 'R', 'S'] + ['null_repeat_%d' % i for i in range(1, protocol['null_repeats']+1)]
            errors = {}
            for arm in arms:
                rows = read_jsonl(root / block['id'] / (arm + '_' + split + '.jsonl'))
                errors[arm] = prompt_errors(rows)
                risk = float(np.mean(list(errors[arm].values())))
                metrics.append(dict(block=block['id'], seed=block['seed'], split=split, arm=arm,
                                    prompts=len(errors[arm]), answers=len(rows), error_rate=risk, accuracy=1-risk,
                                    error_composition=str(dict(Counter(r['error'] for r in rows)))))
            pairs = [('R','initial'), ('S','initial'), ('S','R')] + [(a,'initial') for a in arms if a.startswith('null_repeat')]
            for left,right in pairs:
                mean,lo,hi = paired_interval(errors[left], errors[right], protocol['bootstrap'],
                                             seed_for(protocol['bootstrap_seed'], block['id'], split, left, right))
                contrasts.append(dict(block=block['id'], split=split, contrast=left+'-'+right,
                                      difference=mean, ci_low=lo, ci_high=hi,
                                      interval_unit='paired_prompt_bootstrap_fixed_models'))
    for split in protocol['splits']:
        for contrast in sorted({r['contrast'] for r in contrasts}):
            values = [r['difference'] for r in contrasts if r['split']==split and r['contrast']==contrast]
            mean = float(np.mean(values))
            half = float(t.ppf(.975,len(values)-1)*np.std(values,ddof=1)/np.sqrt(len(values))) if len(values)>1 else None
            summaries.append(dict(split=split, contrast=contrast, blocks=len(values), mean=mean,
                                  ci_low=mean-half if half is not None else None,
                                  ci_high=mean+half if half is not None else None,
                                  interval_unit='paired_training_block_t', small_sample_warning=len(values)<5))
    out.mkdir(parents=True, exist_ok=True)
    save_csv(out / 'metrics_per_block.csv', metrics)
    save_csv(out / 'paired_contrasts.csv', contrasts)
    save_csv(out / 'block_summary.csv', summaries)
    write_json(out / 'analysis_manifest.json', {'evaluation_manifest_hash':digest(manifest),
               'environment':environment(), 'DEMO_ONLY':manifest['DEMO_ONLY'],
               'interpretation':'Positive S-R means S has higher error, not necessarily regression from initial. '
               'Prompt intervals condition on fixed models; block t intervals have small-sample assumptions.'})
    return summaries
