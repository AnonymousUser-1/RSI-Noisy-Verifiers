#!/usr/bin/env python3
"""The Values tables of configs/README.md, generated from the matched configs.

    python scripts/multiround/config_table.py           # print the tables
    python scripts/multiround/config_table.py --write   # replace the README's Values section

The config files are the source; tests/test_matched_pool_config.py fails while the README's
tables differ from what this prints.
"""
import argparse
import json
from pathlib import Path

CONFIGS = Path(__file__).resolve().parents[2] / "configs"
README = CONFIGS / "README.md"
HEADING = "## Values\n"
INTRO = ("The config files are the source; `scripts/multiround/config_table.py --write` regenerates these\n"
         "tables, and `tests/test_matched_pool_config.py` fails while they differ from the files.\n")


def cell(value):
    if value is None:
        return "off"
    if isinstance(value, bool):
        return "on" if value else "off"
    if isinstance(value, list):
        return ", ".join(value)
    return str(value)


def tables(configs=CONFIGS):
    paths = sorted(p for p in configs.glob("matched_*.json") if p.name != "matched_evaluation.json")
    rows = [(p.name, json.loads(p.read_text())) for p in paths]
    sampling = ["| Config | Model | Rounds | Prompts per pool | Answers per prompt | Batch | Max new tokens | Max sequence | Temperature | Top-p | Top-k | Repetition stop (span / period) |",
                "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    training = ["| Config | Learning rate | LoRA r / alpha / dropout | Target modules | Epochs | Micro-batch | Examples per step | Gradient checkpointing | Audit policy / budget / weighting | C:E (error fraction) | Token tolerance |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
    for name, c in rows:
        g, t, a = c["generation"], c["training"], c["audit"]
        stop = g.get("repetition_stop")
        sampling.append("| `%s` | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s | %s |" % (
            name, c["model"], c["rounds"], "all" if g["prompts_per_pool"] is None else g["prompts_per_pool"],
            g["candidates"], g["batch_size"], g["max_new_tokens"], g["max_sequence_length"],
            g["temperature"], g["top_p"], cell(g["top_k"]),
            "off" if not stop else "%s / %s" % (stop["span"], stop["max_period"])))
        m = c["matching"]
        training.append("| `%s` | %s | %s / %s / %s | %s | %s | %s | %s | %s | %s / %s / %s | %s | %s |" % (
            name, t["learning_rate"], t["lora_rank"], t["lora_alpha"], t["lora_dropout"],
            cell(t["target_modules"]), *(cell(t.get(k, "--")) for k in (
                "epochs", "batch_size", "effective_batch_size", "gradient_checkpointing")),
            a["policy"], a["budget"], cell(a["weighting"]),
            "%g:%g (%s)" % ((1 - m["error_fraction"]) / m["error_fraction"], 1, m["error_fraction"]),
            "no limit" if m["token_tolerance"] is None else m["token_tolerance"]))
    evaluation = json.loads((configs / "matched_evaluation.json").read_text())
    return ("Sampling (pools):\n\n" + "\n".join(sampling) +
            "\n\nTraining, auditing and the R/S selection (`--`: not read by that config's stages):\n\n" + "\n".join(training) +
            "\n\nEvaluation (`matched_evaluation.json`, every model): splits %s, batch size %d.\n"
            % (", ".join(evaluation["splits"]), evaluation["batch_size"]))


def section(configs=CONFIGS):
    return HEADING + "\n" + INTRO + "\n" + tables(configs)


def readme_section(readme=README):
    text = readme.read_text()
    return text[text.index(HEADING):]


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--write", action="store_true", help="Replace the Values section of configs/README.md")
    if p.parse_args().write:
        text = README.read_text()
        README.write_text(text[:text.index(HEADING)] + section())
    else:
        print(tables(), end="")
