"""generation.repetition_stop: a row whose last `span` generated tokens repeat with a period of at most
`max_period` tokens is stopped, cut there and marked `truncated`; a finished answer is never touched."""
import json
import tempfile
import types
import unittest
from pathlib import Path

import evaluate_multiround
from rsi import backends
from rsi.common import load_config
from rsi.experiment import decoding_record

try:
    import torch
except ImportError:  # pragma: no cover
    torch = None


def step_through(criterion, width, sequences):
    """Feed the criterion one generated token at a time, up to the shortest sequence; the first length
    at which each row stopped."""
    stopped = {}
    for n in range(1, min(len(s) for s in sequences) + 1):
        ids = torch.tensor([[7] * width + s[:n] for s in sequences])
        done = criterion(ids, None)
        for row in done.nonzero().flatten().tolist():
            stopped.setdefault(row, n)
    return stopped


@unittest.skipIf(torch is None, "torch is not installed")
class CriterionTests(unittest.TestCase):
    def test_a_periodic_tail_stops_its_row_and_nothing_else(self):
        loop = [1, 2] + [4, 5, 6] * 30                 # period 3 after a two-token prefix
        counting = list(range(10, 100))                 # never repeats
        long_cycle = list(range(100, 140)) * 3          # period 40 > max_period
        criterion = backends.repetition_stop(torch, 3, span=24, max_period=8)
        stopped = step_through(criterion, 3, [loop, counting, long_cycle])
        self.assertEqual(stopped, {0: 26})              # 2 + 24: the first 24-token tail inside the loop
        self.assertEqual(criterion.lengths.tolist(), [26, -1, -1])


@unittest.skipIf(torch is None, "torch is not installed")
class GenerateTests(unittest.TestCase):
    EOS = 9   # also the padding id, as for Llama: padding after a stopped row looks like a stop id

    def backend(self, stop_rule, max_new_tokens=300):
        eos, scripts = self.EOS, [[5, 6, self.EOS], [1, 2] + [3, 4] * 400, list(range(10, 10 + 2 * max_new_tokens))]

        class Tokenizer:
            pad_token_id, eos_token_id = eos, eos

            def decode(self, ids, skip_special_tokens=True):
                return " ".join(str(i) for i in ids if i != eos)

        class Model:
            generation_config = types.SimpleNamespace(eos_token_id=eos)
            config = types.SimpleNamespace(use_cache=False)
            passed = {}

            def eval(self):
                pass

            def generate(self, input_ids, attention_mask, max_new_tokens, pad_token_id, **kwargs):
                """transformers' loop in miniature: a finished row is padded, and stopping_criteria end rows."""
                Model.passed = kwargs
                ids, finished = input_ids, [False] * input_ids.shape[0]
                for step in range(max_new_tokens):
                    column = [pad_token_id if finished[r] else scripts[r][step] for r in range(len(finished))]
                    finished = [f or token == eos for f, token in zip(finished, column)]
                    ids = torch.cat([ids, torch.tensor(column)[:, None]], dim=1)
                    if kwargs.get("stopping_criteria") is not None:
                        done = kwargs["stopping_criteria"](ids, None)
                        finished = [f or bool(d) for f, d in zip(finished, done)]
                    if all(finished):
                        break
                return ids

        model = backends.HFBackend.__new__(backends.HFBackend)
        model.torch, model.device, model.tokenizer, model.model = torch, "cpu", Tokenizer(), Model()
        model.config = {"generation": {"batch_size": 3, "max_new_tokens": max_new_tokens, "max_sequence_length": 1024,
                                       "temperature": 1.0, "top_p": 1.0, "top_k": None, "repetition_stop": stop_rule}}
        model.prompt_ids = lambda prompt: [1, 2]
        return model

    def test_a_loop_is_cut_where_it_was_stopped_and_marked(self):
        rows = self.backend({"span": 16, "max_period": 4}).generate([{"id": "a", "prompt": "p"}], 3, seed=0)
        finished, looping, capped = rows
        self.assertEqual((finished["truncated"], finished["completion_tokens"], finished["response"]), (False, 3, "5 6"))
        self.assertEqual((looping["truncated"], looping["completion_tokens"]), (True, 18))   # 2 + 16
        self.assertEqual(looping["response"], " ".join(str(t) for t in ([1, 2] + [3, 4] * 8)))
        self.assertEqual((capped["truncated"], capped["completion_tokens"]), (True, 300))

    def test_off_the_loop_runs_to_the_cap(self):
        model = self.backend(None)
        rows = model.generate([{"id": "a", "prompt": "p"}], 3, seed=0)
        self.assertEqual([(r["truncated"], r["completion_tokens"]) for r in rows], [(False, 3), (True, 300), (True, 300)])
        self.assertNotIn("stopping_criteria", model.model.passed)


class ConfigTests(unittest.TestCase):
    def load(self, rule):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "config.json"
            path.write_text(json.dumps({"backend": "mock", "generation": {"repetition_stop": rule}}))
            return load_config(path)

    def test_null_or_a_span_and_a_period(self):
        self.assertIsNone(self.load(None)["generation"]["repetition_stop"])
        self.assertEqual(self.load({"span": 128, "max_period": 32})["generation"]["repetition_stop"],
                         {"span": 128, "max_period": 32})
        for bad in ({"span": 10, "max_period": 8}, {"span": 128}, {"span": 128, "max_period": 0},
                    {"span": True, "max_period": 2}, {"span": 128, "max_period": 32, "extra": 1}, "yes", 5):
            with self.subTest(rule=bad), self.assertRaises(ValueError):
                self.load(bad)


class RecordTests(unittest.TestCase):
    def test_the_evaluation_protocol_names_the_rule_only_when_a_run_sets_it(self):
        base = {"temperature": 0.0, "top_p": 1.0, "top_k": None, "max_new_tokens": 2048, "max_sequence_length": 4096}
        self.assertNotIn("repetition_stop", evaluate_multiround.decoding_protocol(dict(base, repetition_stop=None)))
        self.assertNotIn("repetition_stop", evaluate_multiround.decoding_protocol(base))   # a run from before it
        rule = {"span": 128, "max_period": 32}
        self.assertEqual(evaluate_multiround.decoding_protocol(dict(base, repetition_stop=rule))["repetition_stop"], rule)

    def test_the_pool_s_decoding_record_holds_the_rule(self):
        rule = {"span": 128, "max_period": 32}
        config = {"backend": "hf", "generation": {"temperature": 1.3, "top_p": 1.0, "top_k": None, "max_new_tokens": 2048,
                                                  "batch_size": 16, "repetition_stop": rule}}
        hf = types.SimpleNamespace(
            model=types.SimpleNamespace(generation_config=types.SimpleNamespace(to_dict=lambda: {}, eos_token_id=1)),
            tokenizer=types.SimpleNamespace(pad_token_id=0, eos_token_id=1, chat_template="stub"))
        record = decoding_record(hf, config)
        self.assertEqual(record["stopping"]["repetition_stop"], rule)
        self.assertNotIn("stopping_criteria", record["overrides"])


if __name__ == "__main__":
    unittest.main()
