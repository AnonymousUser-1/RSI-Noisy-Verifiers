"""Llama-3.2 support: day-independent chat prompts, model/task selection in suites, smoke arguments."""
import datetime
import importlib.util
import io
import os
import sys
import tempfile
import unittest
from contextlib import redirect_stderr, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import prepare_suite
from rsi.backends import CHAT_TEMPLATE_KWARGS, HFBackend, chat_prompt_ids
from rsi.common import DEFAULTS, digest, load_config, read_json

LLAMA_1B = "meta-llama/Llama-3.2-1B-Instruct"
LLAMA_1B_REVISION = "9213176726f574b556790deb65791e0c5aa438b6"
HAS_TOKENIZERS = all(importlib.util.find_spec(m) for m in ("transformers", "tokenizers"))

# The date logic of the Llama 3.x chat template (meta-llama/Llama-3.2-*-Instruct
# tokenizer_config.json), reduced to plain user turns.
LLAMA_DATE_TEMPLATE = (
    "{{- bos_token }}"
    "{%- if not date_string is defined %}"
    "{%- if strftime_now is defined %}{%- set date_string = strftime_now('%d %b %Y') %}"
    "{%- else %}{%- set date_string = '26 Jul 2024' %}{%- endif %}"
    "{%- endif %}"
    "{{- '<|start_header_id|>system<|end_header_id|>\\n\\n' }}"
    "{{- 'Cutting Knowledge Date: December 2023\\n' }}"
    "{{- 'Today Date: ' + date_string + '\\n\\n<|eot_id|>' }}"
    "{%- for message in messages %}"
    "{{- '<|start_header_id|>' + message['role'] + '<|end_header_id|>\\n\\n' + message['content'] | trim + '<|eot_id|>' }}"
    "{%- endfor %}"
    "{%- if add_generation_prompt %}{{- '<|start_header_id|>assistant<|end_header_id|>\\n\\n' }}{%- endif %}")


class ChatPromptTests(unittest.TestCase):
    def test_every_render_gets_the_fixed_keywords(self):
        tokenizer = mock.Mock(chat_template="template")
        tokenizer.apply_chat_template.return_value = [1, 2]
        self.assertEqual(chat_prompt_ids(tokenizer, "question"), [1, 2])
        tokenizer.apply_chat_template.assert_called_once_with(
            [{"role": "user", "content": "question"}], tokenize=True,
            add_generation_prompt=True, enable_thinking=False, date_string="26 Jul 2024")

    def test_the_backend_encodes_prompts_through_the_shared_render(self):
        tokenizer = mock.Mock(chat_template="template")
        tokenizer.apply_chat_template.return_value = [5]
        model = HFBackend.__new__(HFBackend)
        model.tokenizer = tokenizer
        self.assertEqual(model.prompt_ids("question"), [5])
        kwargs = tokenizer.apply_chat_template.call_args.kwargs
        self.assertEqual({key: kwargs[key] for key in CHAT_TEMPLATE_KWARGS}, CHAT_TEMPLATE_KWARGS)
        model.tokenizer = mock.Mock(chat_template=None)
        with self.assertRaises(ValueError):
            model.prompt_ids("question")


class DecodingRecordTests(unittest.TestCase):
    def test_a_pool_records_how_its_prompts_were_rendered(self):
        from rsi.experiment import decoding_record
        tokenizer = SimpleNamespace(pad_token_id=128009, eos_token_id=128009, chat_template="{{ messages }}")
        loaded = SimpleNamespace(generation_config=SimpleNamespace(to_dict=lambda: {}, eos_token_id=[128001, 128009]))
        record = decoding_record(SimpleNamespace(model=loaded, tokenizer=tokenizer), dict(DEFAULTS, backend="hf"))
        self.assertEqual(record["prompt"]["template_kwargs"], CHAT_TEMPLATE_KWARGS)
        self.assertEqual(record["prompt"]["chat_template_digest"], digest("{{ messages }}"))


@unittest.skipUnless(importlib.util.find_spec("torch"), "torch not installed")
class BackendTokenizerSetupTests(unittest.TestCase):
    def test_llama_tokenizer_pads_with_eos_and_decodes_without_cleanup(self):
        """Llama has no pad token and tidies spaces on decode; neither may reach the recorded response."""
        import torch  # noqa: F401 -- loaded before patch.dict, which drops modules first imported inside it
        tokenizer = SimpleNamespace(pad_token_id=None, pad_token=None, eos_token="<|eot_id|>",
                                    clean_up_tokenization_spaces=True)
        model = mock.Mock()
        model.to.return_value = model
        stub = SimpleNamespace(AutoTokenizer=SimpleNamespace(from_pretrained=mock.Mock(return_value=tokenizer)),
                               AutoModelForCausalLM=SimpleNamespace(from_pretrained=mock.Mock(return_value=model)))
        config = dict(DEFAULTS, model=LLAMA_1B, revision=LLAMA_1B_REVISION, device="cpu")
        with mock.patch.dict(sys.modules, {"transformers": stub}):
            HFBackend(config)
        stub.AutoTokenizer.from_pretrained.assert_called_once_with(LLAMA_1B, revision=LLAMA_1B_REVISION)
        self.assertEqual(tokenizer.pad_token, "<|eot_id|>")
        self.assertIs(tokenizer.clean_up_tokenization_spaces, False)


@unittest.skipUnless(HAS_TOKENIZERS, "transformers/tokenizers not installed")
class LlamaTemplateDateTests(unittest.TestCase):
    """The real Jinja render, with the clock moved: the fixed date is what keeps prompts stable."""

    def tokenizer(self):
        from tokenizers import Tokenizer
        from tokenizers.models import WordLevel
        from transformers import PreTrainedTokenizerFast
        tokenizer = PreTrainedTokenizerFast(tokenizer_object=Tokenizer(WordLevel({"<unk>": 0}, unk_token="<unk>")),
                                            bos_token="<|begin_of_text|>")
        tokenizer.chat_template = LLAMA_DATE_TEMPLATE
        return tokenizer

    def render(self, day, **kwargs):
        class Clock(datetime.datetime):
            @classmethod
            def now(cls, tz=None):
                return cls(day.year, day.month, day.day)

        with mock.patch("transformers.utils.chat_template_utils.datetime", Clock):
            return self.tokenizer().apply_chat_template(
                [{"role": "user", "content": "Evaluate this expression exactly.\n(1 + 2)"}],
                tokenize=False, add_generation_prompt=True, **kwargs)

    def test_without_a_date_the_template_writes_the_day_it_runs(self):
        first, later = self.render(datetime.date(2026, 10, 1)), self.render(datetime.date(2031, 3, 5))
        self.assertIn("Today Date: 01 Oct 2026", first)
        self.assertIn("Today Date: 05 Mar 2031", later)

    def test_the_fixed_keywords_make_the_prompt_independent_of_the_day(self):
        first = self.render(datetime.date(2026, 10, 1), **CHAT_TEMPLATE_KWARGS)
        later = self.render(datetime.date(2031, 3, 5), **CHAT_TEMPLATE_KWARGS)
        self.assertEqual(first, later)
        self.assertIn("Today Date: 26 Jul 2024\n", first)
        self.assertTrue(first.endswith("<|start_header_id|>assistant<|end_header_id|>\n\n"))


class PrepareSuiteModelTests(unittest.TestCase):
    def prepare(self, root, study="pilot", **kwargs):
        with redirect_stdout(io.StringIO()):
            return prepare_suite.prepare(study, root / "suite", root / "data", root / "runs", root / "calibration",
                                         **kwargs)

    def configs(self, root):
        return {path.stem: load_config(path) for path in sorted((root / "suite").glob("*.json"))
                if path.name != "jobs.json"}

    def test_default_pilot_is_unchanged(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            jobs = self.prepare(root)
            self.assertEqual([j["id"] for j in jobs],
                             ["pilot_graph_%s_seed%d" % (kind, seed) for kind in ("exact", "iid", "persistent")
                              for seed in (0, 1)])
            for cfg in self.configs(root).values():
                self.assertEqual((cfg["model"], cfg["revision"]), (DEFAULTS["model"], DEFAULTS["revision"]))

    def test_llama_pilot_on_both_tasks_carries_the_commit(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            # A commit id is used as given: pinning it needs no Hub access.
            jobs = self.prepare(root, pin=True, model=LLAMA_1B, revision=LLAMA_1B_REVISION,
                                tasks=["graph", "arithmetic"])
            self.assertEqual(len(jobs), 12)
            self.assertEqual({Path(j["argv"][j["argv"].index("--data") + 1]).name for j in jobs},
                             {"graph", "arithmetic"})
            self.assertEqual(sum("--frozen-null" in j["argv"] for j in jobs), 4)
            configs = self.configs(root)
            self.assertEqual(len(configs), 6)
            for cfg in configs.values():
                self.assertEqual((cfg["model"], cfg["revision"], cfg["rounds"]), (LLAMA_1B, LLAMA_1B_REVISION, 2))
            self.assertEqual(read_json(root / "suite" / "jobs.json")["pinned_revisions"],
                             {LLAMA_1B: LLAMA_1B_REVISION})

    def test_a_study_naming_its_own_model_does_not_inherit_the_revision(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            self.prepare(root, study="replication", model=LLAMA_1B, revision=LLAMA_1B_REVISION)
            for cfg in self.configs(root).values():
                self.assertEqual((cfg["model"], cfg["revision"]), ("Qwen/Qwen3-4B", DEFAULTS["revision"]))

    def test_inconsistent_requests_are_refused_before_writing(self):
        with tempfile.TemporaryDirectory() as d:
            root = Path(d)
            for study, kwargs in (("pilot", {"revision": LLAMA_1B_REVISION}),
                                  ("pilot", {"model": LLAMA_1B, "revision": LLAMA_1B_REVISION.upper()}),
                                  ("learned", {"model": LLAMA_1B, "revision": LLAMA_1B_REVISION}),
                                  ("all", {"model": LLAMA_1B, "revision": LLAMA_1B_REVISION}),
                                  ("controlled", {"tasks": ["graph"]}),
                                  ("pilot", {"tasks": ["graph", "graph"]}),
                                  ("pilot", {"tasks": ["svamp"]})):
                with self.assertRaises(ValueError, msg=(study, kwargs)):
                    self.prepare(root, study=study, **kwargs)
            self.assertFalse((root / "suite").exists())


REPO = Path(__file__).resolve().parents[1]
LLAMA_PINS = {"1b": ("meta-llama/Llama-3.2-1B-Instruct", "9213176726f574b556790deb65791e0c5aa438b6"),
              "3b": ("meta-llama/Llama-3.2-3B-Instruct", "0cb88a4f764b7a12671c53f0838cd831a0843b95")}


class MatchedLinePinTests(unittest.TestCase):
    """The matched line runs on Llama only on its own committed pin, named with RSI_BASE_PIN."""

    def test_pins_and_configs_agree(self):
        # tests/test_matched_pool_config.py checks every config is complete and on its own pin.
        from rsi.base_pin import read_base_pin
        readme = (REPO / "README.md").read_text(encoding="utf-8")
        for size, (model, revision) in LLAMA_PINS.items():
            pin = read_base_pin(REPO / "configs" / "pins" / ("base_pin_llama3.2-%s.json" % size))
            self.assertEqual((pin["model"], pin["revision"]), (model, revision))
            self.assertIn("| `%s` | `%s` |" % (model, revision), readme)
            for kind in ("pool", "iterative"):
                config = load_config(REPO / "configs" / ("matched_%s_llama3.2-%s.json" % (kind, size)))
                self.assertEqual((config["model"], config["revision"]), (model, revision))

    def entry_refusal(self, entry, config, base_pin, **extra):
        """Run an hf entry's main() up to its input checks; return the SystemExit message."""
        import argparse
        args = argparse.Namespace(config=str(REPO / "configs" / config), data=str(self.data),
                                  shared_pool=str(self.data), out=str(self.data / "out"), seed=0,
                                  study_id="pin-test", phase="main", backend="hf",
                                  shared_adapter=str(self.data / "no-adapter"),
                                  reference_gradient=str(self.data / "no-h"), **extra)
        env = {k: v for k, v in os.environ.items() if k != "RSI_BASE_PIN"}
        if base_pin:
            env["RSI_BASE_PIN"] = base_pin
        with mock.patch.dict(os.environ, env, clear=True), self.assertRaises(SystemExit) as stop, \
                redirect_stdout(io.StringIO()), redirect_stderr(io.StringIO()):
            entry.main(args)
        return str(stop.exception.code)

    def test_both_entries_refuse_llama_on_the_frozen_pin_and_take_its_own(self):
        import run_iterative_experiment
        import run_matched_experiment
        from generate_data import generate
        with tempfile.TemporaryDirectory() as d:
            self.data = Path(d) / "data"
            with redirect_stdout(io.StringIO()):
                generate(self.data, "graph", 7, 4, 8, 4, 4, 4, 4)
            pin = str(REPO / "configs" / "pins" / "base_pin_llama3.2-1b.json")
            for entry, config, extra in ((run_matched_experiment, "matched_pool_llama3.2-1b.json", {}),
                                         (run_iterative_experiment, "matched_iterative_llama3.2-1b.json",
                                          {"later_prompts": 4, "later_samples": 2, "resume": False})):
                with self.subTest(entry=entry.__name__):
                    self.assertIn("base pin mismatch", self.entry_refusal(entry, config, None, **extra))
                    past_pin = self.entry_refusal(entry, config, pin, **extra)
                    self.assertNotIn("base pin", past_pin)
                    self.assertIn("no-adapter", past_pin)  # stopped at the next input, the shared adapter
            self.assertFalse((self.data / "out").exists())


class SmokeArgumentTests(unittest.TestCase):
    def run_main(self, *argv):
        import evaluation_gpu_smoke
        with mock.patch.object(sys, "argv", ["evaluation_gpu_smoke.py", *argv]), redirect_stderr(io.StringIO()):
            evaluation_gpu_smoke.main()

    def test_a_model_needs_its_own_commit_id(self):
        with tempfile.TemporaryDirectory() as d:
            out = str(Path(d) / "smoke")
            with self.assertRaises(SystemExit):
                self.run_main(out, "--model", LLAMA_1B)
            with self.assertRaises(SystemExit):
                self.run_main(out, "--revision", LLAMA_1B_REVISION)
            with self.assertRaises(ValueError):
                self.run_main(out, "--model", LLAMA_1B, "--revision", "main")
            self.assertFalse(Path(out).exists())


if __name__ == "__main__":
    unittest.main()
