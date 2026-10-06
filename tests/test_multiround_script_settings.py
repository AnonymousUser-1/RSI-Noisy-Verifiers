"""scripts/multiround: 04 keeps a round-1 pool only while POOL_CONFIG has the generation settings it was
drawn with, and config.sh stops on a setting that moved into the config files (EVAL_*, LATER_*)."""
import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from rsi.common import load_config

REPO = Path(__file__).resolve().parents[1]
RETIRED = ("EVAL_SPLITS", "EVAL_BATCH", "LATER_PROMPTS", "LATER_SAMPLES")
CLEARED = RETIRED + ("WORK", "DATA", "SEEDS", "MODEL_TAG", "TASK", "EVAL_CONFIG", "RSI_BASE_PIN")


def bash(script, work, **env):
    environment = {k: v for k, v in os.environ.items() if k not in CLEARED}
    environment.update(WORK=str(work), RSI_ROOT=str(work), DATA=str(Path(work) / "data"), SEEDS="0",
                       PY=sys.executable, **env)
    return subprocess.run(["bash", "-c", script], cwd=REPO, env=environment, capture_output=True, text=True,
                          timeout=300)


class Round1PoolSettingsTests(unittest.TestCase):
    def write_meta(self, work, **changes):
        generation = dict(load_config(REPO / "configs" / "matched_pool.json")["generation"], **changes)
        meta = Path(work) / "pools" / "b00.jsonl.meta.json"
        meta.parent.mkdir(parents=True)
        meta.write_text(json.dumps({"generation": generation}))

    def test_a_pool_drawn_with_the_config_s_settings_is_kept(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_meta(work)
            done = bash("bash scripts/multiround/04_round1_pools.sh", work)
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertIn("skip: pool b00 exists", done.stdout)

    def test_a_pool_drawn_with_other_settings_stops_04(self):
        with tempfile.TemporaryDirectory() as work:
            self.write_meta(work, temperature=0.7)
            done = bash("bash scripts/multiround/04_round1_pools.sh", work)
            self.assertEqual(done.returncode, 1, done.stdout)
            self.assertIn("was drawn with generation", done.stderr)
            self.assertNotIn("skip:", done.stdout)


class RetiredSettingsTests(unittest.TestCase):
    def test_a_retired_variable_stops_every_script(self):
        for name in RETIRED:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as work:
                done = bash("source scripts/multiround/config.sh && echo sourced", work, **{name: "64"})
                self.assertEqual(done.returncode, 1, done.stdout)
                self.assertIn("%s is no longer read" % name, done.stderr)
                self.assertIn("EVAL_CONFIG" if name.startswith("EVAL") else "ITER_CONFIG", done.stderr)

    def test_eval_config_can_point_at_another_file(self):
        with tempfile.TemporaryDirectory() as work:
            done = bash('source scripts/multiround/config.sh && echo "$EVAL_CONFIG"', work,
                        EVAL_CONFIG="/somewhere/evaluation_h100.json")
            self.assertEqual(done.returncode, 0, done.stderr)
            self.assertEqual(done.stdout.strip().splitlines()[-1], "/somewhere/evaluation_h100.json")
            default = bash('source scripts/multiround/config.sh && echo "$EVAL_CONFIG"', work)
            self.assertEqual(default.stdout.strip().splitlines()[-1], "configs/matched_evaluation.json")


if __name__ == "__main__":
    unittest.main()
