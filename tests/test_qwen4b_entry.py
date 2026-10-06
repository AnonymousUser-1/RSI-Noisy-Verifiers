"""Check real shell dispatch and separation of fresh runs from original-input replay."""
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import unittest

from experiments import import_pools

REPO = Path(__file__).resolve().parents[1]
NAME = 'graph_unaudited_qwen3-4b'
BASH = (r'C:\Program Files\Git\bin\bash.exe' if os.name == 'nt' else shutil.which('bash'))


@unittest.skipUnless(BASH and Path(BASH).is_file(), 'requires Git Bash on Windows or POSIX bash')
class FreshAndReplayTests(unittest.TestCase):
    def run_shell(self, command, root, python=None):
        return subprocess.run([BASH, '-c', command], cwd=REPO,
                              env=dict(os.environ, RSI_ROOT=str(root),
                                       PY=python or Path(sys.executable).as_posix(),
                                       PYTHONIOENCODING='utf-8'),
                              capture_output=True, text=True, encoding='utf-8', timeout=30)

    def recorder(self, root):
        """Replace GPU entry points only; exercise the actual shell stage and guard code."""
        script = root / 'record_calls.py'
        script.write_text('''import json, pathlib, sys
args = sys.argv[1:]
if args[0] == '-c':
    exec(args[1])
    raise SystemExit
with pathlib.Path(__file__).with_name('calls.jsonl').open('a', encoding='utf-8') as f:
    f.write(json.dumps(args) + '\\n')
''', encoding='utf-8')
        wrapper = root / 'python-recorder'
        wrapper.write_text('#!/bin/bash\nexec "%s" "%s" "$@"\n' %
                           (Path(sys.executable).as_posix(), script.as_posix()), encoding='utf-8')
        wrapper.chmod(0o755)
        return wrapper.as_posix()

    def test_fresh_run_dispatches_sampling_adapter_reference_and_training_without_archive(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            result = self.run_shell('bash experiments/run.sh %s pools adapter train' % NAME,
                                    root, self.recorder(root))
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)
            calls = [json.loads(line) for line in (root / 'calls.jsonl').read_text().splitlines()]
            for entry, count in [('sample_candidates.py', 5), ('make_shared_adapter.py', 1),
                                 ('compute_reference_gradient.py', 1), ('run_iterative_experiment.py', 1)]:
                self.assertEqual(sum(entry in args for args in calls), count, (entry, calls))
            self.assertFalse((root / 'experiments' / NAME / 'pools' / 'IMPORTED.json').exists())

    def test_original_input_receipt_retains_multiround_replay_guard(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            receipt = root / 'experiments' / NAME / 'pools' / 'IMPORTED.json'
            receipt.parent.mkdir(parents=True)
            receipt.write_text('{}', encoding='utf-8')
            result = self.run_shell('bash experiments/run.sh %s train' % NAME,
                                    root, self.recorder(root))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('A multi-round run is not started', result.stderr)
            self.assertFalse((root / 'calls.jsonl').exists())

    def test_fresh_one_step_names_missing_generated_inputs_without_requiring_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            result = self.run_shell('bash experiments/one_step.sh %s reference' % NAME, Path(tmp))
            self.assertNotEqual(result.returncode, 0)
            self.assertIn('Missing:', result.stderr)
            self.assertNotIn('are imported, not drawn', result.stderr)


class ImportModeTests(unittest.TestCase):
    def test_import_cannot_convert_a_generated_run_to_replay(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            output = root / 'experiments' / NAME / 'out'
            output.mkdir(parents=True)
            sentinel = output / 'keep.json'
            sentinel.write_text('{"keep": true}', encoding='utf-8')
            with self.assertRaisesRegex(SystemExit, 'fresh RSI_ROOT'):
                import_pools.plan(NAME, root / 'absent-export', root)
            self.assertEqual(sentinel.read_text(), '{"keep": true}')
            self.assertFalse((root / 'data').exists())


if __name__ == '__main__':
    unittest.main()
