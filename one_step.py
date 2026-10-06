#!/usr/bin/env python3
"""One-step entry: the R, S and null comparison for one already-matched pool.

`HANDOFF.md` deliverable B.4: "all `K` examples accumulated, exactly one
`optimizer.step()`, dropout 0, uniform weights, auditing off, identical optimizer
state, learning rate, prompt order, and clip threshold across branches."

Three arms, not two (`HANDOFF.md` order of work, step 4): **R, S, and the
un-updated null**.  The null arm is the zero-step reference the two trained
branches are read against; without it a branch contrast has no floor, and the
review set's scripts that loop `for branch in ("R", "S")` silently drop it.

What the null arm does and does not do
--------------------------------------
It runs the same construction, the same forward passes and the same recorder as
R and S -- so the diagnostics are structurally identical, and a reader can see
that the *only* difference is the update -- but it never calls
`optimizer.step()` and it never saves an adapter.  Its `Delta theta` is
therefore exactly zero and its `h^T Delta theta` is exactly zero.  Those zeros
are recorded as measured values, and the arm is labelled `null` everywhere, so
they cannot be mistaken for a trained result.

Isolation rules enforced here (each one is a hard error, never a warning):
  * dropout 0, uniform weight 1.0 -- checked against the config, not assumed;
  * auditing off (policy `none`, budget 0);
  * both branches seeded from one `shared_adapter/` artefact, verified by
    adapter parameter hash and base model / revision before any forward pass;
  * the same clip threshold and learning rate on every arm;
  * one `optimizer.step()` per arm.

These are the standalone CLI's unaudited baseline rules. The newer experiment
runners opt into budgeted auditing separately, then call run_arm with retained
examples and nonnegative weights. In that extension the one-step gradient is
sum_i w_i grad(loss_i) / sum_i w_i; a subset with no positive weights takes zero
steps and saves an unchanged adapter. The frozen null still uses unaudited R.

The `mock` backend does not train (see `MockBackend.train`), so with it the
gradient diagnostics are recorded as `skipped` with a reason rather than
invented.  That is deliberate: `KNOWN_GAPS.md` forbids substituting mock numbers
for real ones, and a fabricated gradient norm would be exactly that.
"""
import argparse
import json
from pathlib import Path

from rsi.common import digest, load_config, read_json, read_jsonl, require_explicit, write_json
from rsi.budgeted_auditing import enabled as auditing_enabled, positive_rows
from rsi.determinism import enable_determinism
from rsi.tasks import judge

# One place where the isolation policy lives, so a caller cannot half-enforce it.
REQUIRED_DROPOUT = 0.0
REQUIRED_WEIGHT = 1.0
CLIP_THRESHOLD = 1.0

# The arms in run order: (arm, matched subset it reads).  null reads R's subset:
# R's rows, forward passes and recorder, but no optimizer.step().
ARMS = (("R", "R"), ("S", "S"), ("null", "R"))


def check_isolation(config, allow_auditing=False):
    """Refuse to run a comparison that is not actually isolated."""
    training = config["training"]
    problems = []
    if training["lora_dropout"] != REQUIRED_DROPOUT:
        problems.append("lora_dropout is %r, must be %r for a one-step comparison"
                        % (training["lora_dropout"], REQUIRED_DROPOUT))
    auditing_enabled(config["audit"])
    if not allow_auditing and (config["audit"]["policy"] != "none" or config["audit"]["budget"]):
        problems.append("auditing is on (%s / %s); the one-step comparison runs with auditing off"
                        % (config["audit"]["policy"], config["audit"]["budget"]))
    if problems:
        raise ValueError("One-step isolation violated: " + "; ".join(problems))


def load_branch_rows(subset, tasks, candidates, judgements, tokens):
    """Build the training rows for one branch from the matched subset ids.

    Every row carries weight 1.0 (the matched construction guarantees it) and the
    ground-truth label, which the diagnostics need to split G_C from G_E.  The
    label is used *only* for that split; it never enters the loss.

    Rows come in task-id order, not subset order.  The subsets are sorted by
    candidate id, and R and S may hold different error candidates for a task, so
    subset order can put that task's prompt at different positions in the two
    branches; B.4 asks for the same prompt order.  R and S have the same task
    set with one row per task, so task-id order is one order for both.
    """
    by_id = {c["id"]: c for c in candidates}
    rows = []
    for candidate_id in sorted(subset, key=lambda i: (by_id[i]["task_id"], i)):
        candidate = by_id[candidate_id]
        if candidate.get("truncated"):
            # Matching never selects one (rsi.matching); a subset that holds one is not this pool's.
            raise ValueError("%s was cut at max_new_tokens; truncated answers are never trained on" % candidate_id)
        task = tasks[candidate["task_id"]]
        truth = judgements[candidate_id]
        rows.append({"id": candidate_id, "task_id": candidate["task_id"],
                     "prompt": task["prompt"], "response": candidate["response"],
                     "weight": REQUIRED_WEIGHT, "correct": bool(truth["correct"]),
                     "error": truth["error"], "tokens": tokens[candidate_id]})
    return rows


def run_arm(branch, rows, config, adapter_dir, output, seed, recorder_kwargs=None):
    """Run one arm: forward, one update (except null), diagnostics, one file.

    Returns a small record; the heavy diagnostics go to disk under `output`.
    """
    from rsi.backends import backend

    positive_rows(rows)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    record = {"branch": branch, "examples": len(rows), "updates": 0 if branch == "null" else 1,
              "weight_uniform": all(r["weight"] == REQUIRED_WEIGHT for r in rows),
              "clip_threshold": CLIP_THRESHOLD, "seed": seed}
    if config["backend"] == "mock":
        # MockBackend has no gradients; record the skip instead of a number.
        record["diagnostics"] = "skipped"
        record["reason"] = "mock backend has no trainable parameters; gradients are not measurable here"
        record["trained"] = False
        write_json(output / "arm.json", record)
        return record
    trained = _run_hf_arm(branch, rows, config, adapter_dir, output, seed, recorder_kwargs or {})
    record.update(trained)
    record["updates"] = trained["steps"]
    write_json(output / "arm.json", record)
    return record


def _run_hf_arm(branch, rows, config, adapter_dir, output, seed, recorder_kwargs):
    import torch

    from rsi.backends import backend
    from rsi.gradient_recorder import GradientRecorder

    determinism = enable_determinism()  # before the first GPU matrix product (rsi/determinism.py)
    model = backend(config, adapter=adapter_dir, trainable=True)
    try:
        wrapped = model.model
        from rsi.shared_adapter import adapter_parameter_hash, verify_shared_adapter
        # The adapter hash leaves the base out, so the base this backend loaded is
        # checked against the record by model ID and revision.
        try:
            verify_shared_adapter(wrapped, adapter_dir, model.model_name, model.revision)
        except ValueError as exc:
            raise ValueError("Branch %s did not start from the shared adapter: %s" % (branch, exc)) from exc
        wrapped.train()
        wrapped.config.use_cache = False
        optimizer = torch.optim.AdamW([p for p in wrapped.parameters() if p.requires_grad],
                                      lr=config["training"]["learning_rate"], weight_decay=0.0)
        recorder = GradientRecorder(wrapped, **recorder_kwargs)
        recorder.before_update()
        # ONE step over all K examples: zero once, then add each example's
        # contribution (w_i/sum(w)) * grad loss_i into .grad, so the single clip and the
        # single optimizer.step() see the full K-example gradient.  Zeroing per
        # example instead would leave only the last example's gradient for the
        # step.  Each contribution is taken on its own (autograd.grad) so the
        # recorder gets it exactly, not as a difference of running sums.
        torch.manual_seed(seed)
        from rsi.backends import response_losses
        params = [(n, p) for n, p in wrapped.named_parameters() if p.requires_grad]
        optimizer.zero_grad(set_to_none=True)
        active = positive_rows(rows)
        denominator = sum(float(r["weight"]) for r in active)
        losses = []
        for row in active:
            prefix = model.prompt_ids(row["prompt"])
            suffix = model.tokenizer.encode(row["response"], add_special_tokens=False) + [model.tokenizer.eos_token_id]
            ids = torch.tensor([prefix + suffix], device=model.device)
            labels = torch.full_like(ids, -100)
            labels[0, len(prefix):] = torch.tensor(suffix, device=model.device)
            logits = wrapped(input_ids=ids, attention_mask=torch.ones_like(ids)).logits
            loss = response_losses(logits, labels).sum()  # loss_i, this example's response-token mean
            if not torch.isfinite(loss):
                raise FloatingPointError("Non-finite one-step loss")
            contribution = torch.autograd.grad(loss * (float(row["weight"]) / denominator),
                                               [p for _, p in params])
            for (_, p), g in zip(params, contribution):
                if p.grad is None:
                    p.grad = g.detach().clone()
                else:
                    p.grad.add_(g)
            # The recorder takes the scaled contribution and the unscaled loss_i.
            recorder.record_per_sample_gradient(row["id"], row["correct"],
                                                {n: g.detach() for (n, _), g in zip(params, contribution)},
                                                float(loss.detach()))
            losses.append((float(loss.detach()), float(row["weight"])))
        # An empty/zero-weight audited subset is a measured no-op, not an AdamW step.
        before = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                  for n, p in params}
        torch.nn.utils.clip_grad_norm_([p for p in wrapped.parameters() if p.requires_grad],
                                       CLIP_THRESHOLD, error_if_nonfinite=True)
        after = {n: p.grad.detach().clone() if p.grad is not None else torch.zeros_like(p)
                 for n, p in params}
        from rsi.diagnostic_hooks import check_gradient_clipping
        fired = check_gradient_clipping(before, after)
        recorder.record_batch_gradient(before, after, fired)
        update = branch != "null" and bool(active)
        if update:
            optimizer.step()
        recorder.after_update()
        recorder.save(str(output / "diagnostics.json"), step=0, branch=branch, seed=seed)
        if branch != "null":
            wrapped.save_pretrained(output / "adapter")
        return {"trained": update, "steps": int(update),
                "mean_loss": sum(loss * weight for loss, weight in losses) / denominator if denominator else None,
                "positive_weight_examples": len(active), "weight_sum": denominator,
                "skip_reason": "no positive-weight examples after auditing" if not active else None,
                "parameter_hash": adapter_parameter_hash(wrapped), "determinism": determinism}
    finally:
        model.close()


def load_matched(path, block):
    """Read one block out of a `matched_subsets.json` written by match_candidates.py.

    That file is keyed by block (`per_block`), because the ladder must be
    feasible for *every* block at once; a one-step run still trains a single
    block, so the block has to be named rather than guessed.  Reading `R`/`S`
    straight off the top level was the older, wrong shape and raised KeyError.
    """
    document = read_json(path)
    per_block = document.get("per_block")
    if not isinstance(per_block, dict) or not per_block:
        raise SystemExit("%s has no per_block section; run match_candidates.py to produce it" % path)
    if block is None:
        if len(per_block) != 1:
            raise SystemExit("%s holds %d blocks (%s); pass --block to choose one"
                             % (path, len(per_block), ", ".join(sorted(per_block))))
        block = next(iter(per_block))
    if block not in per_block:
        raise SystemExit("Block %r is not in %s; it holds %s" % (block, path, ", ".join(sorted(per_block))))
    return block, per_block[block]


def main(args):
    config = load_config(args.config, args.seed, args.backend)
    if config["backend"] == "hf":
        require_explicit(args.config, "one_step")  # every value a run uses is in its config file
    check_isolation(config)
    tasks = {t["id"]: t for t in read_jsonl(Path(args.data) / "train_001.jsonl")}
    candidates = read_jsonl(args.pool)
    block, matched = load_matched(args.matched, args.block)
    drawn = {key: matched.get("certificate", {}).get(key, default)
             for key, default in (("error_fraction", 0.25), ("token_tolerance", 0))}
    if drawn != config["matching"]:
        raise SystemExit("%s block %s was matched with %s, but %s states %s; match again with this config "
                         "(match_candidates.py --config)" % (args.matched, block, drawn, args.config, config["matching"]))
    judgements = {c["id"]: judge(tasks[c["task_id"]], c["response"]) for c in candidates}
    tokens = {c["id"]: c["completion_tokens"] for c in candidates}

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report = {"pool": str(args.pool), "matched": str(args.matched), "block": block, "seed": args.seed,
              "K": matched.get("audit", {}).get("K_R"), "arms": {}}
    for branch, key in ARMS:
        subset = matched[key]
        rows = load_branch_rows(subset, tasks, candidates, judgements, tokens)
        report["arms"][branch] = run_arm(branch, rows, config, args.shared_adapter,
                                         out / branch, args.seed)
        print("arm=%s examples=%d updates=%d" % (branch, len(rows), report["arms"][branch]["updates"]),
              flush=True)
    # The matched construction must give R and S the same task set and per-task supervised
    # token counts within matching.token_tolerance; a one-step comparison over anything else is void.
    report["matched_audit"] = matched.get("audit", {})
    write_json(out / "one_step.json", report)
    print("wrote", out / "one_step.json")
    return report


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", required=True)
    p.add_argument("--data", required=True, help="Dataset root containing train_001.jsonl")
    p.add_argument("--pool", required=True, help="Candidate pool JSONL from sample_candidates.py")
    p.add_argument("--matched", required=True, help="matched_subsets.json from match_candidates.py")
    p.add_argument("--block", help="Block to train; required when the file holds more than one")
    p.add_argument("--shared-adapter", required=True, help="shared_adapter/ directory")
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--backend", choices=["hf", "mock"])
    a = p.parse_args()
    main(a)
