#!/usr/bin/env python3
"""Validate and tabulate completed graph selection/audit TPRs (stdlib only).

This reads saved summaries, not omitted responses or weights. Run from the
manuscript folder with --snapshot ../experiments/outputs/2026-10-03-h100.
"""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import statistics

BLOCKS = [f"b{i:02d}" for i in range(5)]
NAMES = {"none": "graph_unaudited_llama3.2-3b",
         "audit": "graph_audited_llama3.2-3b"}
SNAPSHOT = "2026-10-03 16:06 UTC"
ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def close(actual, expected, context):
    require(isinstance(actual, (int, float)) and math.isfinite(actual)
            and math.isclose(actual, expected, rel_tol=1e-11, abs_tol=1e-13),
            f"{context}: {actual!r} != {expected!r}")


def load_rounds(snapshot, progress=None):
    snapshot = Path(snapshot).resolve()
    sources, records, matching = {}, [], {}

    def read(path):
        raw = path.read_bytes()
        sources[path.relative_to(snapshot).as_posix()] = hashlib.sha256(raw).hexdigest()
        return json.loads(raw)

    for policy, name in NAMES.items():
        experiment = read(snapshot / name / "out/experiment.json")
        require(experiment["status"] == "complete" and experiment["exit_code"] == 0
                and not experiment["DEMO_ONLY"] and not experiment["stopped"],
                f"{name}: incomplete or demonstration run")
        require(experiment["blocks"] == BLOCKS and experiment["rounds"] >= 4
                and experiment["K"] == 64, f"{name}: unexpected coverage")
        require(experiment["matching_settings"] == {"error_fraction": 0.25, "token_tolerance": 0},
                f"{name}: unexpected matching protocol")
        matches = read(snapshot / name / "out/matching/matched_subsets.json")
        matching[policy] = matches
        for block in BLOCKS:
            initial = matches["per_block"][block]["audit"]
            require(initial["C_R"] == initial["C_S"] == 48
                    and initial["E_R"] == initial["E_S"] == 16,
                    f"{name}/{block}: round-one quotas differ")
            for arm in ("R", "S"):
                for round_number in range(1, 5):
                    directory = snapshot / name / f"out/{block}/{arm}/round_{round_number:03d}"
                    context = f"{name}/{block}/{arm}/{round_number}"
                    selection = read(directory / "selection.json")
                    complete = read(directory / "complete.json")
                    require(selection["round"] == complete["round"] == round_number
                            and selection["arm"] == complete["branch"] == arm
                            and complete["block_id"] == block, f"{context}: record binding")
                    require(selection["K"] == complete["selected_examples"] == 64
                            and len(selection["ids"]) == len(set(selection["ids"])) == 64,
                            f"{context}: selection coverage/duplicates")
                    require(complete["training"]["trained"]
                            and complete["training"]["steps"] == complete["expected_optimizer_steps"],
                            f"{context}: incomplete training")
                    require(complete["audit_queries"] == (16 if policy == "audit" else 0),
                            f"{context}: audit queries")
                    if round_number == 1:
                        before = {"C": initial[f"C_{arm}"], "E": initial[f"E_{arm}"],
                                  "K": 64, "N_plus": initial["N_plus"],
                                  "N_minus": initial["N_minus"], "final_TPR": initial["final_TPR"],
                                  "final_FPR": initial["final_FPR"]}
                        pool_rows = 16384
                        excluded = pool_rows - before["N_plus"] - before["N_minus"]
                    else:
                        before = selection["audit"]
                        certificate = selection["certificate"]
                        require(certificate["feasible"], f"{context}: infeasible selection")
                        for key in ("C", "E", "K", "N_plus", "N_minus"):
                            require(before[key] == certificate[key], f"{context}: certificate {key}")
                        pool_rows = selection["pool"]["rows"]
                        excluded = certificate["truncated_excluded"]
                    require(before["C"] == 48 and before["E"] == 16 and before["K"] == 64,
                            f"{context}: pre-audit quotas")
                    require(before["N_plus"] > 0 and before["N_minus"] > 0
                            and pool_rows == 16384 and excluded >= 0
                            and before["N_plus"] + before["N_minus"] + excluded == pool_rows,
                            f"{context}: nontruncated denominators")
                    close(before["final_TPR"], 48 / before["N_plus"], context + " pre TPR")
                    close(before["final_FPR"], 16 / before["N_minus"], context + " pre FPR")
                    after = None
                    if policy == "audit":
                        audit = read(directory / "audit.json")
                        require(audit["round"] == round_number and audit["enabled"]
                                and audit["policy"] == "adaptive"
                                and audit["queries"] == audit["requested_budget"] == 16
                                and len(audit["labels"]) == 16,
                                f"{context}: audit coverage")
                        require(len({label["candidate_id"] for label in audit["labels"]}) == 16
                                and all(label["candidate_id"] in selection["ids"]
                                        for label in audit["labels"]), f"{context}: audit identity")
                        for key in ("C", "E", "K", "N_plus", "N_minus"):
                            require(audit["before"][key] == before[key], f"{context}: audit input {key}")
                        after = audit["after"]
                        require(after["N_plus"] == before["N_plus"]
                                and after["N_minus"] == before["N_minus"],
                                f"{context}: denominator changed during audit")
                        removed_errors = sum(not label["correct"] for label in audit["labels"])
                        require(after["C"] == before["C"] == 48
                                and after["E"] == before["E"] - removed_errors
                                and after["K"] == after["C"] + after["E"] == complete["examples"],
                                f"{context}: retained composition")
                        require(0 <= after["positive_weight_correct"] <= after["C"]
                                and 0 <= after["positive_weight_errors"] <= after["E"]
                                and after["positive_weight_examples"] == after["positive_weight_correct"]
                                + after["positive_weight_errors"]
                                and complete["training"]["steps"]
                                == math.ceil(after["positive_weight_examples"] / 4),
                                f"{context}: positive-weight composition")
                        close(after["final_TPR"], after["C"] / before["N_plus"], context + " retained TPR")
                        close(after["positive_weight_TPR"],
                              after["positive_weight_correct"] / before["N_plus"], context + " active TPR")
                        close(after["final_FPR"], after["E"] / before["N_minus"], context + " retained FPR")
                    else:
                        require(complete["examples"] == 64 and complete["training"]["steps"] == 16,
                                f"{context}: unaudited training")
                    records.append({
                        "policy": policy, "block": block, "arm": arm, "round": round_number,
                        "N_plus": before["N_plus"], "N_minus": before["N_minus"],
                        "truncated_excluded": excluded, "C_pre": before["C"],
                        "C_retained": after["C"] if after else None,
                        "C_positive": after["positive_weight_correct"] if after else None,
                        "TPR_pre": 48 / before["N_plus"],
                        "TPR_retained": after["C"] / before["N_plus"] if after else None,
                        "TPR_positive": after["positive_weight_correct"] / before["N_plus"] if after else None,
                        "selection_source": (directory / "selection.json").relative_to(snapshot).as_posix(),
                        "audit_source": (directory / "audit.json").relative_to(snapshot).as_posix() if after else None,
                    })
            if progress:
                progress(f"Validated {policy}/{block}: eight completed arm-rounds")
    require(len(records) == len({(r["policy"], r["block"], r["arm"], r["round"])
                                for r in records}) == 80, "duplicate or missing arm-rounds")
    for block in BLOCKS:
        a, b = (matching[p]["per_block"][block] for p in ("none", "audit"))
        require(a["R"] == b["R"] and a["S"] == b["S"], f"{block}: different initial selections")
        for key in ("N_plus", "N_minus", "final_TPR", "final_FPR"):
            require(a["audit"][key] == b["audit"][key], f"{block}: initial {key} differs")
    return records, sources


def summarize(records):
    summary = []
    for round_number in range(1, 5):
        for policy in ("none", "audit"):
            for arm in ("R", "S"):
                group = [r for r in records if (r["round"], r["policy"], r["arm"])
                         == (round_number, policy, arm)]
                require(sorted(r["block"] for r in group) == BLOCKS, "incomplete summary group")
                row = {"round": round_number, "policy": policy, "arm": arm, "n": 5,
                       "N_plus_mean": statistics.fmean(r["N_plus"] for r in group),
                       "N_plus_min": min(r["N_plus"] for r in group),
                       "N_plus_max": max(r["N_plus"] for r in group)}
                for key in ("TPR_pre", "TPR_retained", "TPR_positive"):
                    values = [r[key] for r in group]
                    row[key + "_percent"] = (100 * statistics.fmean(values)
                                                   if all(v is not None for v in values) else None)
                summary.append(row)
    return summary


def render_table(summary):
    lines = [r"\begin{table}[H]",
             r"\caption{Roundwise TPRs for the completed Llama-3.2-3B graph study. "
             r"Rates are percentages, averaged over all five blocks within each condition/arm/round, "
             r"not ratios of pooled counts. $N^+$ is the round-specific nontruncated original-pool "
             r"correct count (mean and range). Audited columns distinguish retained rows from "
             r"positive-weight training rows; both use the same pre-audit denominator. "
             r"Dashes denote no auditing, not missing rounds. These are selection statistics, "
             r"not held-out accuracy or estimates of a population verifier's recall.}",
             r"\label{tab:h100-tpr}",
             r"\centering\small\setlength{\tabcolsep}{5pt}",
             r"\begin{tabular}{@{}rllrccc@{}}", r"\toprule",
             r"Round & Policy & Arm & $N^+$ mean [range] & "
             r"\shortstack{Pre-audit\\TPR (\%)} & "
             r"\shortstack{Post-audit\\retained TPR (\%)} & "
             r"\shortstack{Post-audit\\positive-weight TPR (\%)}\\", r"\midrule"]
    for row in summary:
        policy = "None" if row["policy"] == "none" else "B=16"
        values = ["---" if row[key] is None else f"{row[key]:.4f}"
                  for key in ("TPR_pre_percent", "TPR_retained_percent", "TPR_positive_percent")]
        denominator = f"{row['N_plus_mean']:.1f} [{row['N_plus_min']}--{row['N_plus_max']}]"
        lines.append(f"{row['round']} & {policy} & {row['arm']} & {denominator} & "
                     + " & ".join(values) + r"\\")
    lines.extend([r"\bottomrule", r"\end{tabular}", r"\end{table}"])
    return "\n".join(lines) + "\n"


def write_csv(path, rows):
    with path.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--out", type=Path, default=ROOT / "figures/h100_runs/tpr_rates")
    parser.add_argument("--table", type=Path, default=ROOT / "h100_tpr_table.tex")
    args = parser.parse_args()
    records, sources = load_rounds(args.snapshot, progress=lambda message: print(message, flush=True))
    summary = summarize(records)
    args.out.mkdir(parents=True, exist_ok=True)
    result = {"snapshot": SNAPSHOT, "model": "meta-llama/Llama-3.2-3B-Instruct",
              "completed_arm_rounds": 80, "audited_arm_rounds": 40,
              "aggregation": "unweighted mean of five block rates; display in percent",
              "scope": "saved count/rate consistency; no raw-response rejudging or weight reconstruction",
              "records": records, "summary": summary}
    (args.out / "validated_tpr.json").write_text(json.dumps(result, indent=2) + "\n")
    write_csv(args.out / "block_rates.csv", records)
    write_csv(args.out / "round_summary.csv", summary)
    (args.out / "provenance.json").write_text(json.dumps({
        "snapshot": SNAPSHOT, "source_sha256": dict(sorted(sources.items())),
        "source_files": len(sources), "analysis_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "missing": ["raw candidate responses", "raw training rows and weights"]}, indent=2) + "\n")
    args.table.write_text(render_table(summary))
    print(json.dumps({"status": "passed", "completed_arm_rounds": 80,
                      "audit_reports": 40, "source_files": len(sources), "table_rows": len(summary)}))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
