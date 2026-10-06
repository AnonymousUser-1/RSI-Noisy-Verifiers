#!/usr/bin/env python3
"""Multi-round R/S extension: each arm self-trains for several rounds from its own model.

    python run_iterative_experiment.py --config configs/matched_iterative.json --data DATA \\
        --shared-pool POOLS --out OUT --study-id ID --phase {pilot,main} \\
        --shared-adapter ADAPTER --reference-gradient H_DIR \\
        [--seed 0] [--resume]

Every setting comes from the config file (configs/README.md); on hf it must state
every field this entry reads (rsi.common.STAGE_FIELDS["iterative"]).

An addition to the one-step main comparison (run_matched_experiment.py), not a
replacement.  Team decisions of 2026-10-01: 4 rounds first, continue from the
previous round's model, several optimizer steps per round, and from round 2 each
arm samples and selects its own examples with only the counts and rates matched.

Per seed block (the pools in --shared-pool, named b00, b01, ... with no gaps):
  round 1   the joint matched construction over the block's shared round-1 pool
            (rsi.matching, all blocks at one K, as in run_matched_experiment.py),
            each arm trained from the shared adapter;
  round t>1 each arm samples generation.candidates answers to the first
            generation.prompts_per_pool prompts of train_00t (null: all) from its
            own round t-1 model, with this config's generation settings; answers
            cut at max_new_tokens are never selected; then it selects its own K (rsi.iterative.select_own_subset: R's errors
            uniform over prompts with any error, S's over prompts with a target
            error), trained from its round t-1 adapter.
S's target follows the dataset's task (rsi.matching.targets_for): `nonshortest`
for graph, `ignore_parentheses` for arithmetic, `intermediate` for gsm8k and
`sign` for dmmath.
Both arms of a round share the round's prompts, its sampling seed and its
training seed. Without auditing, every round trains ceil(K / effective_batch_size) * epochs
optimizer steps (configs/matched_iterative.json: 64 / 4 * 1 = 16, one pass over
the K examples) with a fresh AdamW, deterministic kernels, the base's tokenizer
counts as the supervised length, and records |Delta theta| and the projection of
Delta theta on h, the reference gradient at the shared initialisation.
Blocks run in order, all rounds of a block before the next, so a stopped run
leaves whole seeds.  An arm that cannot fill its quota in some round is stopped
there and recorded, never relaxed; the other arm goes on.

Output (OUT must not exist or be empty, unless --resume)
  experiment.json                       status, K, identity, completed, stopped
  matching/                             round-1 ladder and matched subsets
  <block>/{R,S}/run.json                what evaluate.py reads: config (rounds), data, bindings
  <block>/{R,S}/round_00t/              selection.json, training.jsonl, adapter/, complete.json;
                                        pool.jsonl for t > 1
  <block>/{R,S}/finished.json           written when all rounds of the arm ran (an extension moves
                                        it to finished.N-rounds.json until its rounds are done)
  <block>/{R,S}/stopped.json            the round whose quota could not be filled

--resume continues a run with the same identity (config, code, data, pools,
adapter, h, seed, later pool size, study, phase): rounds with complete.json are
kept, an interrupted round directory is moved aside as round_00t.incomplete-N
and run again.  Sampling and training are seeded per round, but sampled decoding
on a GPU is not bitwise reproducible (evaluation_gpu_smoke.py records this), so a
rerun round may draw a different pool; like any pool, it is recorded and bound by
its sha256 in selection.json.

--extend-rounds (implies --resume) continues a complete run to this config's larger
`rounds`, from the round after its last.  The rounds already run are kept: no round depends
on how many follow it (its seeds are the block's and the round's), so they are the rounds a
run with the larger `rounds` would have run.  Nothing but `rounds` may differ from the run's
identity, except the code; the run's earlier identity and code are kept in experiment.json
"extensions", and each arm's run.json is brought to the new rounds (evaluate_multiround.py
evaluates the rounds it states), keeping what it said before (its finished.json becomes
finished.N-rounds.json until the new rounds are done).  h stays bound to the config the run
began with, as it is taken before any round.  Lowering rounds, or extending a run that is not
complete, is refused.  Every resume and extension recomputes the round-1 matching and refuses
to go on unless it is the run's (K and each block's subsets), so other code can never add
rounds that follow another selection.

Optional CONFIG audit section: after selection, R and S independently spend at
most audit.budget correctness-only queries per round. Audited errors are removed
and optional group weights are applied. The original matching is pre-audit only;
post-audit counts, rates, weights and optimizer steps are recorded separately.
Completed audits and their generated pools are cached under <arm>/audit_ledger
and reused on resume, regardless of GPU sampling nondeterminism; changed audit
inputs are refused rather than spending a second budget. An empty or
zero-weight subset takes zero steps and saves the unchanged adapter for the next round.

Exit status: 0 complete; 2 round-1 matching infeasible; 3 complete with stopped
arms; 1 refused before anything was written, or failed (experiment.json says why).
"""
import argparse
import json
import math
import sys
import time
from pathlib import Path

import one_step
from rsi.budgeted_auditing import apply_audit, enabled as auditing_enabled, positive_rows
from match_candidates import report_infeasible, write_matching
from rsi.common import (digest, environment, file_hash, load_config, read_json, read_jsonl, require_explicit,
                        seed_for, verify_dataset, write_json, write_jsonl)
from rsi.determinism import enable_determinism
from rsi.experiment import pin_config
from rsi.iterative import select_own_subset
from rsi.matching import K_LADDER, MatchingFailure, candidate_from_row, select_k_for_blocks, targets_for
from rsi.provenance import git_state
from rsi.reference_encode import check_parameter_binding, supervised_token_count
from rsi.tasks import judge
from run_matched_experiment import (check_branch_rows, check_hf_inputs, check_inputs_given, hardware,
                                    identity, load_tokenizer, read_block_pool, run_id)

SCHEMA_VERSION = "rsi-iterative/1"
ARMS = ("R", "S")
TRAINING_FIELDS = ("id", "task_id", "prompt", "response", "weight", "tokens")
JOINT_RULE = "joint matched construction over the shared round-1 pool"
OWN_RULE = ("own pool, own selection (rsi.iterative.select_own_subset): counts and rates matched, "
            "answers not shared")
H_NOTE = "projection of this round's Delta theta on h, the reference gradient at the shared initialisation"


def block_names(pool_dir):
    names = sorted(p.stem for p in Path(pool_dir).glob("*.jsonl"))
    if not names or names != ["b%02d" % i for i in range(len(names))]:
        raise SystemExit("--shared-pool must hold b00.jsonl, b01.jsonl, ... with no gaps, one per seed block; "
                         "%s holds %s.  Nothing was written." % (pool_dir, ", ".join(names) or "nothing"))
    return names


def lora_snapshot(model):
    return {name: p.detach().float().cpu().clone() for name, p in model.named_parameters()
            if "lora" in name.lower() and p.requires_grad}


def generate_pool(config, adapter, tasks, samples, seed):
    """Sample `samples` answers per task from the base with `adapter` loaded."""
    from rsi.backends import backend
    enable_determinism()
    model = backend(config, adapter=str(adapter))
    try:
        return model.generate(tasks, samples, seed)
    finally:
        model.close()


def train_arm(config, adapter_in, rows, out_adapter, seed, h_path):
    """One round's training from `adapter_in`: rsi.backends.HFBackend.train, then Delta theta and h^T Delta theta."""
    import torch
    from rsi.backends import backend
    from rsi.shared_adapter import verify_shared_adapter
    determinism = enable_determinism()
    model = backend(config, adapter=str(adapter_in), trainable=True)
    try:
        if (Path(adapter_in) / "shared_adapter.json").is_file():
            verify_shared_adapter(model.model, adapter_in, model.model_name, model.revision)
        h = torch.load(h_path, weights_only=True)
        before = lora_snapshot(model.model)
        check_parameter_binding(h, {n: tuple(t.shape) for n, t in before.items()}, "h at %s" % h_path)
        stats = model.train(rows, str(out_adapter), seed)
        if not stats.get("trained"):
            # Preserve the checkpoint chain even when auditing leaves no signal.
            model.model.save_pretrained(out_adapter)
            model.tokenizer.save_pretrained(out_adapter)
        after = lora_snapshot(model.model)
    finally:
        model.close()
    delta = {name: after[name] - before[name] for name in before}
    return dict(stats, delta_theta_norm=math.sqrt(sum(float((d.double() ** 2).sum()) for d in delta.values())),
                h_T_delta_theta=sum(float((h[n].double() * delta[n].double()).sum()) for n in delta),
                h_T_delta_theta_note=H_NOTE, determinism=determinism)


def round_pool(config, adapter, tasks, samples, seed, cache_path=None):
    """Persist generation for audited rounds; seeded GPU decoding can vary on retry."""
    if cache_path is None:
        return generate_pool(config, adapter, tasks, samples, seed)
    cache = Path(cache_path)
    adapter = Path(adapter)
    binding = {"config_hash": digest(config), "tasks_hash": digest(tasks), "samples": samples,
               "seed": seed, "adapter": str(adapter),
               "adapter_hash": digest({p.relative_to(adapter).as_posix(): file_hash(p)
                                       for p in sorted(adapter.rglob("*")) if p.is_file()})}
    if cache.exists():
        saved = read_json(cache)
        if saved["binding"] != binding or saved["pool_hash"] != digest(saved["pool"]):
            raise ValueError("Audit pool ledger inputs or payload changed: %s" % cache)
        return saved["pool"]
    pool = generate_pool(config, adapter, tasks, samples, seed)
    write_json(cache, {"schema": 1, "binding": binding, "pool": pool, "pool_hash": digest(pool)})
    return pool


def run_identity(resolved, dataset, pools, inputs, args, code_hash):
    return {"config_hash": digest(resolved), "source_hash": code_hash, "dataset_hash": digest(dataset),
            "pools": {name: binding["pool_id"] for name, (_, binding) in pools.items()},
            "shared_adapter": inputs["shared_adapter"]["parameter_hash"],
            "h_sha256": inputs["reference_gradient"]["h_sha256"], "seed": args.seed,
            "later_prompts": resolved["generation"]["prompts_per_pool"],
            "later_samples": resolved["generation"]["candidates"],
            "study_id": args.study_id, "phase": args.phase}


def prepare_out(out, resume):
    """The run to continue in `out` (its experiment.json), or None when `out` is new or empty."""
    if not out.exists() or not any(out.iterdir()):
        return None
    if not resume:
        raise SystemExit("--out %s is not empty; pass --resume to continue the same run.  Nothing was "
                         "overwritten." % out)
    try:
        return read_json(out / "experiment.json")
    except (OSError, ValueError):
        raise SystemExit("--out %s holds no readable experiment.json; cannot resume." % out)


def started_rounds(out, rounds):
    """The rounds the run in `out` began with, which its h is bound to; `rounds` for a new run."""
    try:
        previous = read_json(Path(out) / "experiment.json")
    except (OSError, ValueError):
        return rounds
    extensions = previous.get("extensions") or []
    return extensions[0]["from_rounds"] if extensions else previous.get("rounds", rounds)


def check_continuation(out, previous, run_ident, rounds, extend, at_previous_rounds, code):
    """None when `previous` is this run (a resume).  When `extend` and `previous` is this run complete
    at fewer rounds -- `at_previous_rounds` is this run's identity at its rounds -- the record of
    the extension to `rounds`.  Anything else is refused."""
    old = previous.get("identity") or {}
    if old == run_ident:
        return None
    ran = previous.get("rounds")
    if extend and isinstance(ran, int) and ran < rounds:
        if previous.get("status") not in ("complete", "complete_with_stopped_arms"):
            raise SystemExit("--out %s holds a %d-round run with status %r: finish it first with the code and the "
                             "config it began with (--resume), then extend it.  Nothing was overwritten."
                             % (out, ran, previous.get("status")))
        other = sorted(k for k in set(old) | set(at_previous_rounds)
                       if k != "source_hash" and old.get(k) != at_previous_rounds.get(k))
        if other:
            raise SystemExit("--out %s cannot be extended to %d rounds: besides rounds, this run differs in %s.  "
                             "Nothing was overwritten." % (out, rounds, ", ".join(other)))
        return {"from_rounds": ran, "to_rounds": rounds, "at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
                "previous_identity": old, "previous_code": previous.get("code"), "code": code,
                "source_changed": old.get("source_hash") != code["source_hash"]}
    differs = sorted(k for k in set(old) | set(run_ident) if old.get(k) != run_ident.get(k))
    hint = ""
    if isinstance(ran, int) and ran < rounds and not extend:
        hint = ("  It holds a %d-round run and this config has rounds %d; to continue it to %d rounds, pass "
                "--extend-rounds." % (ran, rounds, rounds))
    raise SystemExit("--out %s cannot resume: this run differs in %s.  Nothing was overwritten.%s"
                     % (out, ", ".join(differs), hint))


def extend_arm_manifest(arm_dir, resolved, config, config_file, extension):
    """Brings an arm's run.json, written at fewer rounds, to this run's rounds (evaluate_multiround.py
    evaluates the rounds its config states), keeping what it stated before in its "extensions".  The
    arm's finished.json becomes finished.N-rounds.json: it has not run all its rounds until the new
    ones are done, when finished.json is written again."""
    path = arm_dir / "run.json"
    if not path.is_file():
        return
    manifest = read_json(path)
    ran = manifest["config"]["rounds"]
    if ran == resolved["rounds"]:
        return
    if ran > resolved["rounds"]:
        raise RuntimeError("%s states %d rounds, more than this run's %d" % (path, ran, resolved["rounds"]))
    finished, kept = arm_dir / "finished.json", arm_dir / ("finished.%d-rounds.json" % ran)
    if finished.is_file():
        finished.rename(kept)
    manifest.setdefault("extensions", []).append({
        "from_rounds": ran, "to_rounds": resolved["rounds"], "at": extension["at"], "code": extension["code"],
        "previous_config_hash": manifest["bindings"]["config_hash"],
        "previous_config_file": manifest["bindings"]["config_file"],
        "previous_finished": read_json(kept) if kept.is_file() else None})
    manifest.update(config=resolved, requested_config=config)
    manifest["bindings"] = dict(manifest["bindings"], config_hash=digest(resolved), config_file=config_file)
    write_json(path, manifest)


def set_aside(round_dir):
    """Move an interrupted round directory aside; it is never deleted or reused."""
    n = 1
    while round_dir.with_name("%s.incomplete-%d" % (round_dir.name, n)).exists():
        n += 1
    round_dir.rename(round_dir.with_name("%s.incomplete-%d" % (round_dir.name, n)))


def main(args):
    config = load_config(args.config, args.seed, "hf")
    try:
        require_explicit(args.config, "iterative")
    except ValueError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    one_step.check_isolation(config, allow_auditing=True)
    missing = identity(args.study_id, args.phase)
    if missing:
        raise SystemExit("--study-id and --phase are required (RUN_COST_SPEC 1).  Nothing was written.")
    check_inputs_given("hf", args.shared_adapter, args.reference_gradient)
    data = Path(args.data).resolve()
    dataset = verify_dataset(data)
    resolved = pin_config(config)
    extend = getattr(args, "extend_rounds", False)
    resume = args.resume or extend
    # The base, the shared adapter and h first, as in every matched entry: a run on the wrong base
    # says so before anything about its data.  h is taken before any round and bound to the config
    # the run began with, so a run extended to more rounds checks it at the rounds it began with.
    begun = started_rounds(args.out, resolved["rounds"]) if resume else resolved["rounds"]
    inputs = check_hf_inputs(dict(resolved, rounds=begun), dataset, args.shared_adapter, args.reference_gradient)
    rounds = resolved["rounds"]
    for t in range(1, rounds + 1):
        if "train_%03d.jsonl" % t not in dataset["files"]:
            raise SystemExit("%s has no train_%03d.jsonl for round %d.  Nothing was written." % (data, t, t))
    if rounds > 1:
        available = min(len(read_jsonl(data / ("train_%03d.jsonl" % t))) for t in range(2, rounds + 1))
        later_prompts = resolved["generation"]["prompts_per_pool"]
        if later_prompts is not None and later_prompts > available:
            raise SystemExit("generation.prompts_per_pool is %d, but train_002 ... train_%03d have only %d prompts.  "
                             "Nothing was written." % (later_prompts, rounds, available))
    try:
        targets = targets_for(dataset["task"])
    except ValueError as exc:
        raise SystemExit("%s.  Nothing was written." % exc) from exc
    names = block_names(args.shared_pool)
    tasks1 = {t["id"]: t for t in read_jsonl(data / "train_001.jsonl")}
    # Round-1 pools come from the pool config: their size (candidates, prompts) is not this config's,
    # but every round samples from one distribution, so their sampling settings must be this
    # config's (read_block_pool's `sampling`), and every block must have been drawn alike.
    pools = {name: read_block_pool(Path(args.shared_pool) / (name + ".jsonl"), "hf", tasks1, base=resolved,
                                   dataset=dataset, sampling=resolved["generation"])
             for name in names}
    drawn = {name: binding["meta"].get("generation") for name, (_, binding) in pools.items()}
    if len({json.dumps(g, sort_keys=True) for g in drawn.values()}) > 1:
        raise SystemExit("The round-1 pools were not drawn alike: %s.  Draw them again with one pool config "
                         "(scripts/multiround: delete $WORK/pools).  Nothing was written."
                         % "; ".join("%s %s" % (name, json.dumps(g, sort_keys=True)) for name, g in sorted(drawn.items())))
    env, git = environment(), git_state()
    run_ident = run_identity(resolved, dataset, pools, inputs, args, env["source_hash"])
    out = Path(args.out).resolve()
    previous = prepare_out(out, resume)
    code = {"git_commit": git["git_commit"], "git_dirty": git["git_dirty"], "source_hash": env["source_hash"]}
    extension = None
    if previous is not None:
        ran = previous.get("rounds")
        at_ran = (run_identity(dict(resolved, rounds=ran), dataset, pools, inputs, args, env["source_hash"])
                  if isinstance(ran, int) else {})
        extension = check_continuation(out, previous, run_ident, rounds, extend, at_ran, code)

    tokenizer = load_tokenizer(resolved)

    def counted(rows):
        return {r["id"]: supervised_token_count(tokenizer, r["response"]) for r in rows}

    judgements, tokens = {}, {}
    for name, (rows, _) in pools.items():
        judgements[name] = {r["id"]: judge(tasks1[r["task_id"]], r["response"]) for r in rows}
        tokens[name] = counted(rows)
    candidates = {name: [candidate_from_row(r, tasks1[r["task_id"]], judgements[name][r["id"]], tokens[name][r["id"]])
                         for r in rows] for name, (rows, _) in pools.items()}
    try:
        result = select_k_for_blocks(candidates, seed=args.seed, k_ladder=K_LADDER, target_signatures=targets,
                                     **resolved["matching"])
    except MatchingFailure as failure:
        raise SystemExit("Matching failed, not infeasible: %s.  Nothing was written." % failure)
    k = result["K"]
    if previous is not None and previous.get("K") is not None:
        # The rounds a run has trained follow its round-1 matching (K and each block's subsets): a resume or an
        # extension under other code must reproduce it, or the rounds it adds would follow another selection.
        recorded = {name: read_json(out / name / "R" / "run.json")["bindings"]["matching"] for name in names
                    if (out / name / "R" / "run.json").is_file()}
        changed = sorted(name for name, m in recorded.items()
                         if not result["feasible"] or m != {"K": k, "block_hash": result["per_block"][name]["hash"]})
        if k != previous["K"] or changed:
            raise SystemExit("--out %s: this code's round-1 matching is not the run's (K %s, the run's %s%s).  Nothing "
                             "was overwritten." % (out, k, previous["K"],
                                                   "; other subsets in %s" % ", ".join(changed) if changed else ""))
    round1 = {}
    if result["feasible"]:
        for name, (rows, _) in pools.items():
            matched = result["per_block"][name]
            round1[name] = {arm: one_step.load_branch_rows(matched[arm], tasks1, rows, judgements[name], tokens[name])
                            for arm in ARMS}
            check_branch_rows(name, round1[name]["R"], round1[name]["S"], k, **resolved["matching"])
    training = resolved["training"]
    steps = math.ceil(k / training["effective_batch_size"]) * training["epochs"] if k else None

    out.mkdir(parents=True, exist_ok=True)
    experiment = {"schema_version": SCHEMA_VERSION, "status": "running", "exit_code": None, "backend": "hf",
                  "DEMO_ONLY": False, "study_id": args.study_id, "phase": args.phase, "identity": run_ident,
                  "task": dataset["task"], "target_signatures": list(targets),
                  "blocks": names, "arms": list(ARMS), "rounds": rounds, "K": k, "steps_per_round": steps,
                  "matching_settings": resolved["matching"],
                  "auditing": {"enabled": auditing_enabled(resolved["audit"]), "config": resolved["audit"],
                               "budget_unit": "correctness queries per arm per round",
                               "matching_constraints_stage": "pre_audit",
                               "steps_note": "steps_per_round is pre-audit; complete.json records actual expected steps"},
                  "later_pool": {"prompts": resolved["generation"]["prompts_per_pool"],
                                 "samples": resolved["generation"]["candidates"],
                                 "source": "config generation.prompts_per_pool and generation.candidates",
                                 "prompt_rule": "the first prompts_per_pool prompts of train_00t (null: all), "
                                                "the same for both arms"},
                  "attempts": (previous or {}).get("attempts", 0) + 1,
                  "extensions": list((previous or {}).get("extensions") or []) + ([extension] if extension else []),
                  "completed": [], "stopped": [], "code": code}
    write_json(out / "experiment.json", experiment)
    config_file = {"path": str(Path(args.config).resolve()), "sha256": file_hash(args.config)}
    try:
        if experiment["extensions"]:
            for name in names:
                for arm in ARMS:
                    extend_arm_manifest(out / name / arm, resolved, config, config_file, experiment["extensions"][-1])
        write_matching(out / "matching", names, list(K_LADDER), result,
                       metadata={"task": dataset["task"], "target_signatures": list(targets)})
        if not result["feasible"]:
            write_json(out / "experiment.json", dict(experiment, status="infeasible", exit_code=2))
            report_infeasible(result)
            return 2
        bindings = {"config_hash": digest(resolved), "config_file": config_file,
                    "model": {"name": resolved["model"], "revision": resolved["revision"]},
                    "tokenizer": {"name": resolved["model"], "revision": resolved["revision"]},
                    "token_counts": "rsi.reference_encode.supervised_token_count (response tokens + EOS)",
                    "base_pin": inputs["base_pin"], "shared_adapter": inputs["shared_adapter"],
                    "reference_gradient": inputs["reference_gradient"], "hardware": hardware("hf"),
                    "data": {"path": str(data), "manifest_sha256": file_hash(data / "manifest.json"),
                             "dataset_hash": digest(dataset)}}
        for name in names:
            stopped = set()
            for arm in ARMS:
                arm_dir = out / name / arm
                if not (arm_dir / "run.json").exists():
                    write_json(arm_dir / "run.json", {
                        "schema_version": SCHEMA_VERSION, "study_id": args.study_id, "phase": args.phase,
                        "block_id": name, "branch": arm, "run_id": run_id(args.study_id, name, arm),
                        "config": resolved, "requested_config": config, "dataset_hash": digest(dataset),
                        "data_path": str(data), "environment": env, "calibration": None, "DEMO_ONLY": False,
                        "task": dataset["task"], "seed": args.seed, "evaluation": "pending",
                        "auditing": experiment["auditing"],
                        "bindings": dict(bindings, pool=pools[name][1],
                                         matching={"K": k, "block_hash": result["per_block"][name]["hash"]})})
                if (arm_dir / "stopped.json").exists():  # stopped by an earlier attempt
                    stopped.add(arm)
                    experiment["stopped"].append("%s/%s" % (name, arm))
            for t in range(1, rounds + 1):
                for arm in ARMS:
                    if arm in stopped:
                        continue
                    status = run_round(out, name, arm, t, args, resolved, data, k, steps, inputs,
                                       round1.get(name), counted, targets, result["per_block"][name]["audit"])
                    if status == "stopped":
                        stopped.add(arm)
                        experiment["stopped"].append("%s/%s" % (name, arm))
                    else:
                        experiment["completed"].append("%s/%s/round_%03d" % (name, arm, t))
                    write_json(out / "experiment.json", experiment)
            for arm in ARMS:
                if arm not in stopped:
                    last = read_json(out / name / arm / ("round_%03d" % rounds) / "complete.json")
                    write_json(out / name / arm / "finished.json",
                               {"rounds": rounds, "cumulative_audit_queries": last["cumulative_audit_queries"],
                                "calibration_queries": 0})
    except BaseException as exc:
        write_json(out / "experiment.json", dict(experiment, status="failed",
                                                 exit_code=1 if isinstance(exc, Exception) else None,
                                                 failure_reason="%s: %s" % (type(exc).__name__, exc)))
        raise
    final = 3 if experiment["stopped"] else 0
    write_json(out / "experiment.json", dict(experiment, status="complete_with_stopped_arms" if final else "complete",
                                             exit_code=final))
    print("wrote", out / "experiment.json")
    return final


def run_round(out, block, arm, t, args, resolved, data, k, steps, inputs, round1_rows, counted, targets,
              round1_counts=None):
    """One arm-round.  Returns "done" or "stopped"."""
    arm_dir = out / block / arm
    round_dir = arm_dir / ("round_%03d" % t)
    if (round_dir / "complete.json").is_file():
        return "done"
    if round_dir.exists():
        set_aside(round_dir)
    round_dir.mkdir(parents=True)
    started = time.time()
    adapter_in = Path(inputs["shared_adapter"]["path"]) if t == 1 else arm_dir / ("round_%03d" % (t - 1)) / "adapter"
    previous = read_json(arm_dir / ("round_%03d" % (t - 1)) / "complete.json") if t > 1 else {}
    if t == 1:
        rows = round1_rows[arm]
        by_task = {task["id"]: task for task in read_jsonl(data / "train_001.jsonl")}
        pool_counts = round1_counts
        write_json(round_dir / "selection.json", {"rule": JOINT_RULE, "round": t, "arm": arm, "K": k,
                                                  "ids": [r["id"] for r in rows],
                                                  "matching": "matching/matched_subsets.json"})
    else:
        tasks = read_jsonl(data / ("train_%03d.jsonl" % t))[:resolved["generation"]["prompts_per_pool"]]
        by_task = {task["id"]: task for task in tasks}
        sample_seed = seed_for(args.seed, block, t, "generation")
        pool_cache = arm_dir / "audit_ledger" / ("round_%03d.pool.json" % t) if auditing_enabled(resolved["audit"]) else None
        if pool_cache is not None and (pool_cache.parent / ("round_%03d.json" % t)).exists() and not pool_cache.exists():
            raise ValueError("Missing cached pool for completed audit; refusing resampling: %s" % pool_cache)
        pool = round_pool(resolved, adapter_in, tasks, resolved["generation"]["candidates"], sample_seed, pool_cache)
        write_jsonl(round_dir / "pool.jsonl", pool)
        judgements = {r["id"]: judge(by_task[r["task_id"]], r["response"]) for r in pool}
        tokens = counted(pool)
        candidates = [candidate_from_row(r, by_task[r["task_id"]], judgements[r["id"]], tokens[r["id"]]) for r in pool]
        selection = select_own_subset(candidates, arm, k, args.seed, (block, t, arm), target_signatures=targets,
                                      error_fraction=resolved["matching"]["error_fraction"])
        write_json(round_dir / "selection.json", {
            "rule": OWN_RULE, "round": t, "arm": arm, "K": k, "ids": selection["ids"],
            "certificate": selection["certificate"], "audit": selection["audit"],
            "pool": {"path": "pool.jsonl", "sha256": file_hash(round_dir / "pool.jsonl"), "rows": len(pool),
                     "prompts": len(tasks), "samples": resolved["generation"]["candidates"], "seed": sample_seed,
                     "sampled_from": adapter_in.relative_to(out).as_posix()}})
        if not selection["feasible"]:
            write_json(arm_dir / "stopped.json", {"round": t, "reason": selection["certificate"]["reason"]})
            print("block=%s arm=%s round=%d status=stopped (%s)" % (block, arm, t, selection["certificate"]["reason"]),
                  flush=True)
            return "stopped"
        rows = one_step.load_branch_rows(selection["ids"], by_task, pool, judgements, tokens)
        pool_counts = selection["audit"]
    selected_examples = len(rows)
    audit = {"queries": 0, "groups": {}}
    if auditing_enabled(resolved["audit"]):
        write_jsonl(round_dir / "pre_audit.jsonl", [{key: r[key] for key in TRAINING_FIELDS} for r in rows])
        rows, audit = apply_audit(rows, by_task, resolved["audit"],
                                  seed_for(args.seed, block, "budgeted_auditing"), t, pool_counts,
                                  previous=previous.get("audit_groups", {}),
                                  cache_path=arm_dir / "audit_ledger" / ("round_%03d.json" % t))
        write_json(round_dir / "audit.json", audit)
    write_jsonl(round_dir / "training.jsonl", [{key: r[key] for key in TRAINING_FIELDS} for r in rows])
    training = resolved["training"]
    expected_steps = math.ceil(len(positive_rows(rows)) / training["effective_batch_size"]) * training["epochs"]
    stats = train_arm(resolved, adapter_in, rows, round_dir / "adapter", seed_for(args.seed, block, t, "training"),
                      inputs["h_path"])
    if stats.get("steps") != expected_steps:
        raise RuntimeError("Arm %s/%s round %d took %r optimizer steps, not %d" % (block, arm, t, stats.get("steps"), expected_steps))
    write_json(round_dir / "complete.json", {
        "round": t, "block_id": block, "branch": arm, "adapter": "round_%03d/adapter" % t, "examples": len(rows),
        "selected_examples": selected_examples, "expected_optimizer_steps": expected_steps,
        "adapter_in": adapter_in.relative_to(out).as_posix() if t > 1 else "shared_adapter",
        "training": stats, "seconds": time.time() - started,
        "audit_queries": audit["queries"],
        "cumulative_audit_queries": previous.get("cumulative_audit_queries", 0) + audit["queries"],
        "audit_groups": audit["groups"]})
    print("block=%s arm=%s round=%d status=done examples=%d steps=%d" % (block, arm, t, len(rows), stats["steps"]),
          flush=True)
    return "done"


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--config", required=True, help="e.g. configs/matched_iterative.json (rounds, steps, sampling)")
    p.add_argument("--data", required=True, help="Dataset root with train_001 ... train_00R")
    p.add_argument("--shared-pool", required=True, help="Round-1 pools b00.jsonl, b01.jsonl, ... with their .meta.json")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--study-id", required=True)
    p.add_argument("--phase", required=True, choices=("pilot", "main"))
    p.add_argument("--shared-adapter", required=True)
    p.add_argument("--reference-gradient", required=True, help="compute_reference_gradient.py output for this config")
    p.add_argument("--resume", action="store_true", help="Continue a run with the same identity in --out")
    p.add_argument("--extend-rounds", action="store_true",
                   help="Continue the complete run in --out to this config's larger rounds; implies --resume")
    a = p.parse_args()
    sys.exit(main(a))
