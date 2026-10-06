#!/usr/bin/env python3
"""Import GSM8K or DeepMind Mathematics into the repository's split layout.

Writes what generate_data.py writes -- train_001..train_NNN, dev, calibration, eval_id, eval_ood
when the dataset has one, gradient_reference (last), and manifest.json with every file's sha256 --
plus the pinned source hashes and the import choices.  Rows are identified by content and are
disjoint across splits.  The raw files must already be local and match the pins (fetch_data.py).

Difficulty is not separated: every row is in one 'mixed' stratum, and levels are pooled.

gsm8k   Train-side splits are drawn at random from the official train set.  eval_id is the official
        test set (all of it unless --eval-id asks for fewer).  GSM8K has no out-of-distribution
        split, so there is no eval_ood.
dmmath  Train-side splits take the chosen modules in turn; each row is drawn at random from that
        module's train-easy, train-medium and train-hard files pooled.  eval_id comes from
        interpolate/ and eval_ood from the extrapolate/ files of the modules that have one, also
        module by module.

Within one import every split is disjoint.  Two imports from the same raw files draw from one
finite pool, so a smaller import's rows can take another role in a larger one (a pilot's dev
question can sit in the full import's train_001).  Import the larger one with --disjoint-from
the smaller: no question the smaller one trains, tunes or calibrates on then enters its
train-side splits.
"""
import argparse
import json
import re
from pathlib import Path

from rsi.common import file_hash, read_jsonl, rng_for, verify_dataset, write_json, write_jsonl
from rsi.external import (DMMATH_LEVELS, DMMATH_PROMPT, EXTRAPOLATE, GSM8K_PROMPT, TABLE2_MODULES, dmmath_task,
                          read_dmmath, read_gsm8k)
from rsi.tasks import ANSWER_RULES, IMPORTED_TASKS

TRAIN_PAIRS_PER_FILE = 10000  # pairs read from each DeepMind Mathematics train file (of 666,666)


class Pool:
    """Rows in one seeded random order, drawn without replacement and never twice across splits."""

    def __init__(self, name, rows, rng):
        self.name, self.rows, self.position = name, list(rows), 0
        rng.shuffle(self.rows)

    def draw(self, seen):
        while self.position < len(self.rows):
            row = self.rows[self.position]
            self.position += 1
            if row["id"] not in seen:
                seen.add(row["id"])
                return row
        raise ValueError("%s has no distinct row left (%d rows); ask for fewer" % (self.name, len(self.rows)))


def train_side(rounds, per_round, dev, calibration):
    return [("train_%03d" % r, per_round) for r in range(1, rounds + 1)] + [("dev", dev), ("calibration", calibration)]


def draw_splits(plan, exclude=()):
    """`plan` lists (split, count, choose); choose(i) is the pool of the split's i-th row.

    Splits draw in plan order: the official test splits first, so a question a training file
    shares with them stays out of training, and gradient_reference last, as in generate_data.py.
    """
    seen = set(exclude)
    return [(name, [dict(choose(i).draw(seen), split=name) for i in range(count)]) for name, count, choose in plan]


def in_turn(pools, names):
    return lambda i: pools[names[i % len(names)]]


def gsm8k_splits(raw, seed, rounds, per_round, dev, calibration, eval_id, gradient_reference, answer_rule,
                 exclude=(), sources=None):
    rows, provenance = read_gsm8k(raw, answer_rule, sources)
    train = Pool("GSM8K train", rows["train"], rng_for(seed, "gsm8k", "train"))
    test = Pool("GSM8K test", rows["test"], rng_for(seed, "gsm8k", "test"))
    plan = [("eval_id", len(test.rows) if eval_id is None else eval_id, lambda i: test)]
    plan += [(name, count, lambda i: train) for name, count in train_side(rounds, per_round, dev, calibration)]
    plan.append(("gradient_reference", gradient_reference, lambda i: train))
    choices = {"answer_rule": answer_rule, "prompt": GSM8K_PROMPT, "difficulty": "mixed",
               "pools": {"train": len(train.rows), "test": len(test.rows)},
               "layout": "train-side splits drawn at random from the official train set; eval_id from the "
                         "official test set; GSM8K has no eval_ood"}
    return draw_splits(plan, exclude), provenance, choices


def dmmath_splits(raw, seed, rounds, per_round, dev, calibration, eval_id, eval_ood, gradient_reference,
                  answer_rule, modules, train_pairs=TRAIN_PAIRS_PER_FILE, exclude=(), sources=None):
    modules = list(modules)
    if len(set(modules)) != len(modules):
        raise ValueError("--modules names a module twice")
    ood_modules = [m for m in modules if m in EXTRAPOLATE]
    if eval_ood and not ood_modules:
        raise ValueError("no chosen module has an extrapolate/ file; modules that have one: "
                         + ", ".join(sorted(EXTRAPOLATE)))
    eval_id = 2000 if eval_id is None else eval_id
    eval_ood = (1000 if ood_modules else 0) if eval_ood is None else eval_ood
    # A fixed prefix of every train file, whatever the counts asked for, so the pools do not depend on
    # the counts: a count changes only the splits drawn after it, and gradient_reference, drawn last,
    # changes nothing else.
    pairs, provenance = read_dmmath(raw, modules, train_pairs, sources)

    def pool(kind, module, levels):
        rows = [dmmath_task(q, a, module, level, answer_rule,
                            {"file": "%s/%s.txt" % (level, EXTRAPOLATE[module] if level == "extrapolate" else module),
                             "pair": i})
                for level in levels for q, a, i in pairs[(level, module)]]
        return Pool("DeepMind Mathematics %s %s" % (kind, module), rows, rng_for(seed, "dmmath", kind, module))

    train = {m: pool("train", m, DMMATH_LEVELS) for m in modules}
    plan = [("eval_id", eval_id, in_turn({m: pool("interpolate", m, ("interpolate",)) for m in modules}, modules))]
    if eval_ood:
        plan.append(("eval_ood", eval_ood,
                     in_turn({m: pool("extrapolate", m, ("extrapolate",)) for m in ood_modules}, ood_modules)))
    plan += [(name, count, in_turn(train, modules)) for name, count in train_side(rounds, per_round, dev, calibration)]
    plan.append(("gradient_reference", gradient_reference, in_turn(train, modules)))
    choices = {"answer_rule": answer_rule, "prompt": DMMATH_PROMPT, "difficulty": "mixed", "modules": modules,
               "eval_ood_files": {m: EXTRAPOLATE[m] for m in ood_modules} if eval_ood else {},
               "train_pairs_read_per_file": train_pairs,
               "layout": "modules in turn; train-side rows from each module's train-easy/medium/hard files pooled, "
                         "eval_id from interpolate/, eval_ood from extrapolate/"}
    return draw_splits(plan, exclude), provenance, choices


TRAIN_SIDE_FILE = re.compile(r"(train_\d{3}|dev|calibration|gradient_reference)\.jsonl")


def train_side_ids(datasets, task):
    """The ids of every train-side row (train_NNN, dev, calibration, gradient_reference) of earlier
    imports, each checked against its manifest, and the record the new manifest keeps of them."""
    ids, record = set(), []
    for root in map(Path, datasets):
        manifest = verify_dataset(root)
        if manifest["task"] != task:
            raise ValueError("%s holds %s rows, not %s" % (root, manifest["task"], task))
        held = set()
        for name in sorted(n for n in manifest["files"] if TRAIN_SIDE_FILE.fullmatch(n)):
            held.update(row["id"] for row in read_jsonl(root / name))
        record.append({"path": str(root.resolve()), "manifest_sha256": file_hash(root / "manifest.json"),
                       "train_side_ids": len(held)})
        ids |= held
    return ids, record


def write_dataset(out, kind, seed, rounds, splits, provenance, choices):
    out.mkdir(parents=True, exist_ok=True)
    files = {}
    for name, rows in splits:
        write_jsonl(out / (name + ".jsonl"), rows)
        files[name + ".jsonl"] = file_hash(out / (name + ".jsonl"))
    manifest = {"version": 1, "task": kind, "seed": seed, "rounds": rounds, "files": files,
                "counts": {name: len(rows) for name, rows in splits},
                "unique_instances": sum(len(rows) for _, rows in splits),
                "source": provenance, "import": choices}
    write_json(out / "manifest.json", manifest)
    return manifest


def main(argv=None, sources=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--task", choices=IMPORTED_TASKS, required=True)
    p.add_argument("--raw", required=True, help="Directory holding the pinned raw files (fetch_data.py)")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=2027)
    p.add_argument("--rounds", type=int, default=2)
    p.add_argument("--per-round", type=int, default=2048)
    p.add_argument("--dev", type=int, default=256)
    p.add_argument("--calibration", type=int, default=256)
    p.add_argument("--eval-id", type=int, help="gsm8k: default the whole official test set; dmmath: default 2000")
    p.add_argument("--eval-ood", type=int, help="dmmath only: default 1000 from extrapolate/")
    p.add_argument("--gradient-reference", type=int, default=64)
    p.add_argument("--answer-rule", choices=ANSWER_RULES, default="strict",
                   help="strict: the whole response is the number (default); first_number / last_number: "
                        "the first / last number anywhere in it")
    p.add_argument("--modules", nargs="+",
                   help="dmmath only: single-number modules; default the Table 2 set: " + " ".join(TABLE2_MODULES))
    p.add_argument("--train-pairs-per-file", type=int, default=TRAIN_PAIRS_PER_FILE,
                   help="dmmath only: question/answer pairs read from the start of each train file "
                        "(default %(default)s; a prefix is a random sample, since the generator draws pairs independently)")
    p.add_argument("--disjoint-from", nargs="+", default=[], metavar="DATASET",
                   help="earlier imports, such as a pilot: no question of their train-side splits (train_NNN, dev, "
                        "calibration, gradient_reference) enters this import")
    a = p.parse_args(argv)
    if min([a.rounds, a.per_round, a.dev, a.calibration, a.gradient_reference]
           + [x for x in (a.eval_id, a.eval_ood) if x is not None]) < 1:
        p.error("All counts must be positive")
    if a.task == "gsm8k" and (a.eval_ood is not None or a.modules or a.train_pairs_per_file != TRAIN_PAIRS_PER_FILE):
        p.error("--eval-ood, --modules and --train-pairs-per-file are DeepMind Mathematics options")
    if a.train_pairs_per_file < 1:
        p.error("--train-pairs-per-file must be positive")
    out = Path(a.out)
    if out.exists() and any(out.iterdir()):
        p.error("Output must be new or empty: %s" % out)
    exclude, disjoint = train_side_ids(a.disjoint_from, a.task)
    if a.task == "gsm8k":
        splits, provenance, choices = gsm8k_splits(a.raw, a.seed, a.rounds, a.per_round, a.dev, a.calibration,
                                                   a.eval_id, a.gradient_reference, a.answer_rule,
                                                   exclude=exclude, sources=sources)
    else:
        splits, provenance, choices = dmmath_splits(a.raw, a.seed, a.rounds, a.per_round, a.dev, a.calibration,
                                                    a.eval_id, a.eval_ood, a.gradient_reference, a.answer_rule,
                                                    a.modules or TABLE2_MODULES, a.train_pairs_per_file,
                                                    exclude=exclude, sources=sources)
    choices = dict(choices, disjoint_from=disjoint)
    manifest = write_dataset(out, a.task, a.seed, a.rounds, splits, provenance, choices)
    print(json.dumps({"out": str(out), "task": a.task, "counts": manifest["counts"]}))
    return manifest


if __name__ == "__main__":
    main()
