#!/usr/bin/env python3
"""Acceptance checks and the per-block tables of a one-step run (run_matched_experiment.py output).

  python experiments/one_step_check.py --run OUT_ONE_STEP --reference MATCHED_SUBSETS.json
                                       [--evaluation EVALUATION.json] [--record CHECKS.json]
                                       [--table TABLE.csv] [--table15 TABLE15.csv]

The checks, each printed as ok or FAIL; exit code is 1 when any check fails:
  1. every block has the arms R, S and null; R and S took exactly one optimizer step, null none, and
     null's parameters are unchanged (|Delta theta| = 0);
  2. the gradient composition G_C + G_E = G holds in every arm, as diagnostics.json records;
  3. no two blocks share half or more of their training prompts (the blocks draw from independent
     pools, so a large overlap means they are not independent replicates);
  4. each block's R and S matched subsets equal those in the reference matching file, the
     matched_subsets.json of the multi-round run whose round-1 pools this run uses.  Without
     --reference this check fails: the run cannot be shown to train on the multi-round round-1 sets.
With --evaluation (the experiment's evaluation.json), also:
  5. every split it names has been evaluated on every checkpoint: the base (round 0, which is also
     the null arm) and R and S of every block, each on all of the split's questions, and each R and S
     result on the adapter the arm holds now.  Figures and tables are drawn only from a complete
     evaluation.

--record writes a JSON file with the result of every check.  --table writes one CSV row per block and
arm: matching statistics, update diagnostics and, per evaluated split, Pass@1, the error rate
(1 - Pass@1) and the error change against null.  --table15 writes the paper's one-step results table
(appendix, the one-step protocol of section 4.4): one row per block and arm with the ID and OOD error,
the ID error change against null, h^T Delta theta and |Delta theta|_2.
"""
import argparse
import csv
import sys
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evaluate_multiround import adapter_hash  # noqa: E402
from rsi.common import read_json, read_jsonl, write_json  # noqa: E402

ARMS = ("R", "S", "null")
UPDATED = ("R", "S")


def blocks(run):
    return sorted(read_json(Path(run) / "matching" / "matched_subsets.json")["per_block"])


def evaluation_name(block, arm):
    """The evaluation file of an arm's checkpoint: null is the base, round 0."""
    return "round_000.json" if arm == "null" else "%s_%s_round_001.json" % (block, arm)


def check(run, reference=None):
    run = Path(run)
    found = []
    names = blocks(run)
    for block in names:
        for arm in ARMS:
            complete = read_json(run / block / arm / "round_001" / "complete.json")
            diagnostics = read_json(run / block / arm / "round_001" / "diagnostics.json")
            steps = complete["training"]["steps"]
            want = 0 if arm == "null" else 1
            found.append((steps == want, "%s/%s optimizer steps %s (want %s)" % (block, arm, steps, want)))
            if arm == "null":
                found.append((diagnostics["delta_theta"]["norm"] == 0,
                              "%s/null |Delta theta| %s (want 0)" % (block, diagnostics["delta_theta"]["norm"])))
            valid = diagnostics["validation"]
            found.append((valid["composition_valid"], "%s/%s gradient composition, relative error %.2e"
                          % (block, arm, valid["relative_error"] or 0.0)))
    prompts = {b: {r["task_id"] for r in read_jsonl(run / b / "R" / "round_001" / "training.jsonl")} for b in names}
    shared = max((len(prompts[a] & prompts[b]) / len(prompts[a]) for a, b in combinations(names, 2)), default=0.0)
    found.append((shared < 0.5, "largest prompt overlap between two blocks: %.0f%% (want well below 50%%)" % (100 * shared)))
    if reference is not None:
        ours = read_json(run / "matching" / "matched_subsets.json")["per_block"]
        ref = read_json(Path(reference))["per_block"]
        for block in names:
            same = block in ref and all(sorted(ours[block][a]) == sorted(ref[block][a]) for a in ("R", "S"))
            found.append((same, "%s R/S subsets equal the reference round-1 matching" % block))
    else:
        found.append((False, "no reference round-1 matching: the R/S subsets cannot be compared with the "
                             "multi-round run's round 1"))
    return found


def check_evaluation(run, splits):
    """Check 5: every checkpoint evaluated on every split, on all its questions and the arm's current adapter."""
    run = Path(run)
    found = []
    names = blocks(run)
    for split in splits:
        directory = run / ("evaluation_greedy_" + split)
        if not (directory / "protocol.json").is_file():
            found.append((False, "%s: not evaluated (no %s)" % (split, directory / "protocol.json")))
            continue
        questions = read_json(directory / "protocol.json")["questions"]
        missing, wrong = [], []
        checkpoints = [("base", None, "round_000.json")] + [(b, a, evaluation_name(b, a)) for b in names for a in UPDATED]
        for block, arm, name in checkpoints:
            path = directory / name
            if not path.is_file():
                missing.append(name[:-len(".json")])
                continue
            result = read_json(path)
            if result.get("questions") != questions:
                wrong.append("%s answered %s of %s questions" % (name, result.get("questions"), questions))
            if arm is not None:
                # complete.json's adapter is relative to the arm directory, as evaluate_multiround.py reads it.
                adapter = run / block / arm / read_json(run / block / arm / "round_001" / "complete.json")["adapter"]
                if result.get("adapter_hash") != adapter_hash(adapter):
                    wrong.append("%s was evaluated on another adapter than %s/%s holds" % (name, block, arm))
        total = len(checkpoints)
        found.append((not missing, "%s: %d of %d checkpoints evaluated%s" % (
            split, total - len(missing), total, "; missing " + ", ".join(missing) if missing else "")))
        for text in wrong:
            found.append((False, "%s: %s" % (split, text)))
    return found


def evaluated(run):
    """{split: directory} of every evaluated split."""
    return {p.name[len("evaluation_greedy_"):]: p for p in sorted(Path(run).glob("evaluation_greedy_*"))
            if (p / "protocol.json").is_file()}


def pass1(directory, block, arm):
    path = directory / evaluation_name(block, arm)
    return read_json(path)["pass1"] if path.is_file() else None


def table(run):
    run = Path(run)
    matched = read_json(run / "matching" / "matched_subsets.json")["per_block"]
    splits = evaluated(run)
    rows = []
    for block in blocks(run):
        cert = matched[block]["certificate"]
        for arm in ARMS:
            complete = read_json(run / block / arm / "round_001" / "complete.json")
            diagnostics = read_json(run / block / arm / "round_001" / "diagnostics.json")
            row = {"block": block, "arm": arm, "K": cert["K"], "C": cert["C"], "E": cert["E"],
                   "N_plus": cert["N_plus"], "N_minus": cert["N_minus"],
                   "TPR": cert["C"] / cert["N_plus"], "FPR": cert["E"] / cert["N_minus"],
                   "yield": cert["K"] / (cert["N_plus"] + cert["N_minus"]), "precision": cert["C"] / cert["K"],
                   "truncated_excluded": cert.get("truncated_excluded"),
                   "steps": complete["training"]["steps"], "mean_loss": complete["training"].get("mean_loss"),
                   "h_T_delta_theta": diagnostics["main_diagnostic"]["h_T_delta_theta"],
                   "delta_theta_norm": diagnostics["delta_theta"]["norm"],
                   "clipping_fired": diagnostics["clipping"]["fired"]}
            for split, directory in splits.items():
                value, base = pass1(directory, block, arm), pass1(directory, block, "null")
                row["pass1_" + split] = value
                row["error_" + split] = None if value is None else 1 - value
                row["delta_error_" + split] = None if value is None or base is None else base - value
            rows.append(row)
    return rows


def table15(run, task, model):
    """The paper's one-step results table: per block and arm, ID/OOD error, the ID error change
    against null, h^T Delta theta and |Delta theta|_2.  Errors are 1 - greedy Pass@1."""
    out = []
    for row in table(run):
        out.append({"task": task, "model": model, "block": row["block"], "arm": row["arm"],
                    "id_error": row.get("error_eval_id"), "ood_error": row.get("error_eval_ood"),
                    "delta_id_error": row.get("delta_error_eval_id"),
                    "h_T_delta_theta": row["h_T_delta_theta"], "delta_theta_l2": row["delta_theta_norm"]})
    return out


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print("wrote", path)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", required=True, help="the one-step run's --out directory")
    p.add_argument("--reference", help="matched_subsets.json of the multi-round run that uses the same pools; "
                   "check 4 fails without it")
    p.add_argument("--evaluation", help="the experiment's evaluation.json: also require a complete evaluation "
                   "of every split it names (check 5)")
    p.add_argument("--record", help="write a JSON record of every check to this file")
    p.add_argument("--table", help="write the per-block table to this CSV")
    p.add_argument("--table15", help="write the paper's one-step results table to this CSV")
    a = p.parse_args(argv)
    results = check(a.run, a.reference)
    if a.evaluation:
        results += check_evaluation(a.run, read_json(a.evaluation)["splits"])
    for ok, text in results:
        print("%-4s %s" % ("ok" if ok else "FAIL", text))
    passed = all(ok for ok, _ in results)
    if a.record:
        Path(a.record).parent.mkdir(parents=True, exist_ok=True)
        write_json(a.record, {"checks": [{"ok": ok, "text": text} for ok, text in results],
                              "evaluation_checked": bool(a.evaluation), "passed": passed})
    if passed and a.table:
        write_csv(a.table, table(a.run))
    if passed and a.table15:
        experiment = read_json(Path(a.run) / "experiment.json")
        model = read_json(Path(a.run) / blocks(a.run)[0] / "R" / "run.json")["config"]["model"]
        write_csv(a.table15, table15(a.run, experiment["task"], model))
    return 0 if passed else 1


if __name__ == "__main__":
    sys.exit(main())
