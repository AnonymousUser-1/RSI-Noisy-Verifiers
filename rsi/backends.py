from __future__ import annotations

import gc
import json
import math
from pathlib import Path

from .common import digest, read_json, rng_for, write_json
from .tasks import mock_wrong_answer, reference_answer, shortest_path


def resolve_revision(model, revision):
    if Path(model).is_dir():
        raise ValueError("Use a Hugging Face model ID with a pinned revision for experiment runs")
    if len(revision) == 40 and all(c in "0123456789abcdef" for c in revision.lower()):
        return revision
    from huggingface_hub import model_info
    return model_info(model, revision=revision).sha


def lora_config_from_training(training):
    """Build the PEFT LoRA config from the config's training values.

    Single source of truth for the matched-dynamics adapter: rank, alpha, dropout and
    target modules come from the config (configs/README.md), with plain alpha/r scaling
    (`use_rslora=False`).  Every value is read from the resolved config; nothing about the adapter is
    hardcoded here, so a config that retunes LoRA is visible in the run manifest.
    Kept free of torch/peft imports so the protocol is testable without a GPU
    environment; `peft.LoraConfig` is imported lazily.
    """
    from peft import LoraConfig
    return LoraConfig(
        task_type="CAUSAL_LM", r=training["lora_rank"], lora_alpha=training["lora_alpha"],
        lora_dropout=training["lora_dropout"], bias="none",
        target_modules=list(training["target_modules"]),
        use_rslora=False)


# Passed to every chat-template render.  A template reads only the variables it names, so
# each entry is inert for the other model families:
#   enable_thinking -- Qwen3: False keeps a thinking block out of the reply.
#   date_string     -- Llama 3.x writes "Today Date: <date>" into its system header and,
#                      unless given a date, takes the day the code runs (strftime_now), so one
#                      prompt would tokenize differently from day to day.  26 Jul 2024 is those
#                      templates' own fallback when no clock is available.
CHAT_TEMPLATE_KWARGS = {"enable_thinking": False, "date_string": "26 Jul 2024"}


def chat_prompt_ids(tokenizer, prompt):
    """Token ids of `prompt` as the one user turn of a chat, ending where the reply starts."""
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": prompt}], tokenize=True,
        add_generation_prompt=True, **CHAT_TEMPLATE_KWARGS)


def response_losses(logits, labels):
    """Per-example mean response NLL, ignoring prompt and padding tokens."""
    import torch
    import torch.nn.functional as F
    shifted = labels[:, 1:]
    losses = F.cross_entropy(logits[:, :-1].float().transpose(1, 2), shifted,
                             ignore_index=-100, reduction="none")
    counts = shifted.ne(-100).sum(dim=1)
    if torch.any(counts == 0):
        raise ValueError("Training example has no supervised response tokens")
    return losses.sum(dim=1) / counts


# Windows only.  The WDDM driver lets CUDA allocations spill into system RAM, so PyTorch's cache
# grows past the card instead of being freed and reused, until an allocation outside the cache fails
# with "CUDA error: out of memory": a Qwen3-1.7B training round on an 8 GiB RTX 4070 Laptop failed
# that way at 5.3 GiB allocated and 14.8 GiB reserved.  Capping the cache below the card makes
# PyTorch free and reuse cached blocks instead (the same round then peaks at 7.6 GiB reserved).
# It changes no computation.  Linux drivers do not spill, so nothing is set there.
CUDA_CACHE_FRACTION = 0.95


def limit_cuda_cache(torch, device, platform=None):
    """Cap PyTorch's CUDA cache at CUDA_CACHE_FRACTION of the card on Windows; returns the cap set."""
    import sys
    if (platform or sys.platform) != "win32" or not device.startswith("cuda"):
        return None
    index = torch.device(device).index
    # The call takes an indexed device: "cuda" alone means the current one.
    torch.cuda.set_per_process_memory_fraction(
        CUDA_CACHE_FRACTION, torch.cuda.current_device() if index is None else index)
    return CUDA_CACHE_FRACTION


def repetition_stop(torch, width, span, max_period):
    """generation.repetition_stop as a transformers StoppingCriteria: a row stops once its last `span`
    generated tokens repeat with a period of at most `max_period` tokens -- a loop, which would only
    run to max_new_tokens and hold its whole batch there.  `lengths[row]` is how many tokens the row
    had generated when it was stopped (-1: not stopped).  A finished answer never ends in such a tail
    (span >= 2 * max_period, so the shortest one is two whole periods), so it never stops one."""
    from transformers import StoppingCriteria

    class RepetitionStop(StoppingCriteria):
        lengths = None

        def __call__(self, input_ids, scores, **kwargs):
            rows = input_ids.shape[0]
            if self.lengths is None:
                self.lengths = torch.full((rows,), -1, dtype=torch.long, device=input_ids.device)
            done = torch.zeros(rows, dtype=torch.bool, device=input_ids.device)
            generated = input_ids[:, width:]
            if generated.shape[1] >= span:
                tail = generated[:, -span:]
                for period in range(1, max_period + 1):
                    done |= (tail[:, period:] == tail[:, :-period]).all(dim=1)
                self.lengths = torch.where(done & (self.lengths < 0), generated.shape[1], self.lengths)
            return done

    return RepetitionStop()


class HFBackend:
    def __init__(self, config, adapter=None, model_name=None, revision=None, trainable=False):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.torch, self.config = torch, config
        self.device = config["device"]
        if self.device.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable. Run HF experiments on the GPU server; use --backend mock only for a smoke test")
        limit_cuda_cache(torch, self.device)
        name, rev = model_name or config["model"], revision or config["revision"]
        self.tokenizer = AutoTokenizer.from_pretrained(name, revision=rev)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        # Llama tokenizers delete spaces before punctuation on decode; the response that is judged
        # and re-encoded for training must be the text that was generated (Qwen3 already is).
        self.tokenizer.clean_up_tokenization_spaces = False
        self.model = AutoModelForCausalLM.from_pretrained(
            name, revision=rev, torch_dtype=getattr(torch, config["dtype"]),
            attn_implementation="sdpa", low_cpu_mem_usage=True).to(self.device)
        self.model_name, self.revision = name, rev
        if adapter:
            from peft import PeftModel
            self.model = PeftModel.from_pretrained(self.model, str(adapter), is_trainable=trainable)
        self.model.eval()

    def prompt_ids(self, prompt):
        if not self.tokenizer.chat_template:
            raise ValueError("A chat template is required for this backend")
        return chat_prompt_ids(self.tokenizer, prompt)

    def padded(self, rows, left=False):
        torch = self.torch
        width = max(map(len, rows))
        ids = torch.full((len(rows), width), self.tokenizer.pad_token_id, dtype=torch.long, device=self.device)
        mask = torch.zeros_like(ids)
        for i, row in enumerate(rows):
            start = width - len(row) if left else 0
            ids[i, start:start+len(row)] = torch.tensor(row, dtype=torch.long, device=self.device)
            mask[i, start:start+len(row)] = 1
        return {"input_ids": ids, "attention_mask": mask}

    def generate(self, tasks, candidates, seed):
        torch, cfg = self.torch, self.config["generation"]
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        entries = [(task, sample) for task in tasks for sample in range(candidates)]
        results = []
        self.model.eval()
        self.model.config.use_cache = True
        for offset in range(0, len(entries), cfg["batch_size"]):
            batch = entries[offset:offset+cfg["batch_size"]]
            prompts = [self.prompt_ids(task["prompt"]) for task, _ in batch]
            if any(len(p) + cfg["max_new_tokens"] > cfg["max_sequence_length"] for p in prompts):
                raise ValueError("Prompt + generation allowance exceeds max_sequence_length; change the task/config, do not truncate the problem")
            inputs = self.padded(prompts, left=True)
            width = inputs["input_ids"].shape[1]
            # Temperature 0 is greedy decoding (evaluation); the model's own sampling defaults are
            # overridden so they cannot apply.
            decoding = (dict(do_sample=True, temperature=cfg["temperature"], top_p=cfg["top_p"], top_k=cfg["top_k"])
                        if cfg["temperature"] > 0 else dict(do_sample=False, temperature=None, top_p=None, top_k=None))
            rule, looping = cfg.get("repetition_stop"), None
            if rule:
                from transformers import StoppingCriteriaList
                looping = repetition_stop(torch, width, rule["span"], rule["max_period"])
                decoding["stopping_criteria"] = StoppingCriteriaList([looping])
            with torch.inference_mode():
                outputs = self.model.generate(**inputs, **decoding,
                                              max_new_tokens=cfg["max_new_tokens"],
                                              pad_token_id=self.tokenizer.pad_token_id)
            stopped = (looping.lengths.tolist() if looping is not None and looping.lengths is not None
                       else [-1] * len(batch))
            for (task, sample), sequence, prompt, loop_at in zip(batch, outputs[:, width:], prompts, stopped):
                tokens = sequence.tolist()
                stop_ids = self.model.generation_config.eos_token_id or self.tokenizer.eos_token_id
                stop_ids = stop_ids if isinstance(stop_ids, list) else [stop_ids]
                stop = next((i+1 for i, token in enumerate(tokens) if token in stop_ids), None)
                if loop_at >= 0 and (stop is None or stop > loop_at):
                    # Stopped while repeating a block (generation.repetition_stop): a loop that would
                    # only have run to max_new_tokens.  Cut where it was stopped (the padding after
                    # it may be the eos id), and marked like an answer cut at the cap.
                    tokens, truncated = tokens[:loop_at], True
                else:
                    # No stop id within max_new_tokens: the answer was cut, not finished.  It is kept
                    # in the pool and judged, but marked, so that nothing trains on it.
                    truncated = stop is None
                    tokens = tokens if truncated else tokens[:stop]
                results.append({"id": digest([task["id"], sample]), "task_id": task["id"], "sample": sample,
                                "response": self.tokenizer.decode(tokens, skip_special_tokens=True).strip(),
                                "completion_tokens": len(tokens), "prompt_tokens": len(prompt),
                                "truncated": truncated})
            print("generated %d/%d" % (len(results), len(entries)), flush=True)
        return results

    def score_verdicts(self, prompts):
        """P(CORRECT | {CORRECT, INCORRECT}); scores entire multi-token completions."""
        torch, cfg = self.torch, self.config["generation"]
        prepared = []
        for prompt in prompts:
            prefix = self.prompt_ids(prompt)
            for answer in ("CORRECT", "INCORRECT"):
                suffix = self.tokenizer.encode(answer, add_special_tokens=False)
                ids = prefix + suffix
                if len(ids) > cfg["max_sequence_length"]:
                    raise ValueError("Verifier prompt too long; refusing silent truncation")
                prepared.append((ids, len(prefix)))
        scores = []
        self.model.eval()
        for offset in range(0, len(prepared), cfg["batch_size"]):
            batch = prepared[offset:offset+cfg["batch_size"]]
            inputs = self.padded([ids for ids, _ in batch])
            with torch.inference_mode():
                logits = self.model(**inputs).logits
                for i, (ids, start) in enumerate(batch):
                    positions = logits[i, start-1:len(ids)-1].float().log_softmax(-1)
                    targets = torch.tensor(ids[start:], device=self.device)
                    scores.append(float(positions.gather(1, targets[:, None]).sum().cpu()))
        return [1 / (1 + math.exp(max(-700, min(700, scores[i+1] - scores[i]))))
                for i in range(0, len(scores), 2)]

    def train(self, rows, output, seed):
        import torch
        from peft import PeftModel, get_peft_model
        cfg = self.config["training"]
        rows = [r for r in rows if r["weight"] > 0]
        if not rows:
            return {"trained": False, "steps": 0, "mean_loss": None}
        torch.manual_seed(seed)
        if not isinstance(self.model, PeftModel):
            self.model = get_peft_model(self.model, lora_config_from_training(cfg))
        encoded = []
        for r in rows:
            prefix = self.prompt_ids(r["prompt"])
            suffix = self.tokenizer.encode(r["response"], add_special_tokens=False) + [self.tokenizer.eos_token_id]
            ids = prefix + suffix
            if len(ids) > self.config["generation"]["max_sequence_length"]:
                raise ValueError("Training sequence too long; refusing truncation of supervision")
            encoded.append((ids, [-100]*len(prefix) + suffix, float(r["weight"])))
        self.model.config.use_cache = False
        if cfg["gradient_checkpointing"]:
            self.model.enable_input_require_grads()
            self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        optimizer = torch.optim.AdamW([p for p in self.model.parameters() if p.requires_grad],
                                     lr=cfg["learning_rate"], weight_decay=0.0)
        self.model.train()
        losses, steps = [], 0
        rng = rng_for(seed, "training_order")
        for epoch in range(cfg["epochs"]):
            order = list(range(len(encoded)))
            rng.shuffle(order)
            for offset in range(0, len(order), cfg["effective_batch_size"]):
                effective = [encoded[i] for i in order[offset:offset+cfg["effective_batch_size"]]]
                denominator = sum(r[2] for r in effective)
                optimizer.zero_grad(set_to_none=True)
                total = 0.0
                for j in range(0, len(effective), cfg["batch_size"]):
                    micro = effective[j:j+cfg["batch_size"]]
                    inputs = self.padded([r[0] for r in micro])
                    labels = torch.full_like(inputs["input_ids"], -100)
                    for k, (_, target, _) in enumerate(micro):
                        labels[k, :len(target)] = torch.tensor(target, device=self.device)
                    output_logits = self.model(**inputs).logits
                    per_example = response_losses(output_logits, labels)
                    weights = torch.tensor([r[2] for r in micro], device=self.device)
                    loss = (per_example * weights).sum() / denominator
                    if not torch.isfinite(loss):
                        raise FloatingPointError("Non-finite loss")
                    loss.backward()
                    total += float(loss.detach().cpu())
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0, error_if_nonfinite=True)
                optimizer.step()
                steps += 1
                losses.append(total)
            print("epoch %d/%d, steps=%d, loss=%.5f" % (epoch+1, cfg["epochs"], steps, losses[-1]), flush=True)
        self.model.save_pretrained(output)
        self.tokenizer.save_pretrained(output)
        self.model.eval()
        self.model.config.use_cache = True
        return {"trained": True, "steps": steps, "mean_loss": sum(losses)/len(losses),
                "examples": len(rows), "response_tokens_per_epoch": sum(sum(x != -100 for x in r[1]) for r in encoded),
                "training_tokens_per_epoch": sum(len(r[0]) for r in encoded)}

    def close(self):
        del self.model
        gc.collect()
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


class MockBackend:
    """Oracle-assisted fake model for plumbing tests ONLY. Never a research baseline."""
    def __init__(self, config, adapter=None, **kwargs):
        self.config = config
        self.p = read_json(Path(adapter) / "mock.json")["p"] if adapter else 0.55

    def generate(self, tasks, candidates, seed):
        rng, result = rng_for(seed, "mock_generation"), []
        for task in tasks:
            for i in range(candidates):
                if rng.random() < self.p:
                    answer = reference_answer(task)
                elif task["task"] == "graph":
                    path = shortest_path(task)
                    answer = json.dumps([path[0], path[1], path[0]] + path[1:]) if rng.random() < 0.7 else "[]"
                elif task["task"] == "arithmetic":
                    answer = str(-int(reference_answer(task))) if rng.random() < 0.7 else "incorrect"
                else:
                    answer = mock_wrong_answer(task, rng)
                result.append({"id": digest([task["id"], i]), "task_id": task["id"], "sample": i,
                               "response": answer, "completion_tokens": len(answer.split()),
                               "prompt_tokens": len(task["prompt"].split()), "truncated": False})
        return result

    def score_verdicts(self, prompts):
        return [0.1 + 0.8 * (int(digest(prompt)[:8], 16) / (2**32-1)) for prompt in prompts]

    def train(self, rows, output, seed):
        active = [r for r in rows if r["weight"] > 0]
        if not active:
            return {"trained": False, "steps": 0, "mean_loss": None}
        self.p = min(0.9, self.p + 0.01)  # Deliberately synthetic, not estimated learning.
        write_json(Path(output) / "mock.json", {"p": self.p, "DEMO_ONLY": True})
        return {"trained": True, "steps": 1, "mean_loss": None, "examples": len(active)}

    def close(self):
        pass


def backend(config, adapter=None, **kwargs):
    return (MockBackend if config["backend"] == "mock" else HFBackend)(config, adapter=adapter, **kwargs)
