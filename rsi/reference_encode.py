from __future__ import annotations

"""The single encode used by the reference gradient and by the training arms.

`h = grad R_reference` is only comparable with the recorded `Delta theta` if both
sides turn a (prompt, response) pair into tokens the same way.  That means one
construction, not two that happen to agree today:

    prefix = model.prompt_ids(prompt)
    suffix = model.tokenizer.encode(response, add_special_tokens=False) + [eos]
    input_ids = prefix + suffix
    labels    = [-100]*len(prefix) + suffix

which is exactly what `rsi/backends.py` does for training (`HFBackend.train`) and
what `one_step.py::_run_hf_arm` does for the one-step arms.  `rsi/backends.py` was
frozen by T003/T006a (PR #14 changed its chat-template keywords and decode setting, not
these lines), so instead of a second copy of the
three lines -- which would be free to drift the moment either side is touched --
the construction lives here and `tests/test_reference_encode.py` pins it against
the frozen encode lines read out of `rsi/backends.py` (a stub tokenizer on both
sides; the real tokenizer runs only on the GPU box).

The last position of `labels` is the EOS token, never -100: `response_losses`
shifts labels by one, so `labels[:, 1:]` must still carry the EOS for every
response token to be supervised.  `assert_supervised_last_token` states that as
a check rather than leaving it to a reader.

Deliberately free of torch/transformers/peft imports: the encode is testable on a
CPU-only box, and nothing here decides which model or device is used.
"""


def encode_example(model, prompt, response):
    """Encode one (prompt, response) pair the way the training arms do.

    Returns `(input_ids, labels, prefix_length)`.  `model` is any object with
    `.prompt_ids(prompt)` and `.tokenizer` -- in practice an `HFBackend`, but a
    stub with the same two attributes is enough to test this function.
    """
    prefix = model.prompt_ids(prompt)
    suffix = model.tokenizer.encode(response, add_special_tokens=False) + [model.tokenizer.eos_token_id]
    return prefix + suffix, [-100] * len(prefix) + suffix, len(prefix)


def supervised_token_count(tokenizer, response):
    """The number of supervised tokens `encode_example` gives `response`: its tokens plus EOS.

    This, not the pool's generation-time `completion_tokens`, is the length the
    matching holds equal across R and S, because it is what the arms train on.
    The two differ when decoding with `skip_special_tokens` and `strip()` does
    not round-trip (GPU pilot 2026-10-01: 27 of 16,384 Qwen3-1.7B answers).
    Tokenizer only, so the matching can count without loading the model.
    """
    return len(tokenizer.encode(response, add_special_tokens=False)) + 1


def check_parameter_binding(h, expected, what="h"):
    """Refuse an h whose parameters are not the ones the recorder will project onto.

    `GradientRecorder` loads h from disk and dot-products it with its own
    `Delta theta`; `_dot_product` checks the key **set** and then multiplies
    by key, so a name that matches with a tensor of the wrong shape is a silent
    wrong number rather than an error.  The projection is defined by the LoRA
    parameter names and shapes, not by the loss: a different loss averages the
    same coordinates differently, it does not move them.  This states that
    binding as a check -- names, shapes, and order.  `_dot_product`'s number does
    not depend on the order; the order is checked because the record lists h's
    parameters in order and the acceptance item names it.

    `expected` is a name -> shape mapping, in order, taken from the same adapter
    the arms use.
    """
    have = {name: tuple(t.shape) for name, t in h.items()}
    want = {name: tuple(shape) for name, shape in expected.items()}
    missing = sorted(set(want) - set(have))
    extra = sorted(set(have) - set(want))
    if missing or extra:
        raise ValueError(
            "%s is not in the recorder's coordinates: missing %s, unexpected %s; "
            "the projection h^T Delta theta is defined on the adapter's LoRA parameters only"
            % (what, missing, extra))
    wrong = {name: (have[name], want[name]) for name in sorted(have) if have[name] != want[name]}
    if wrong:
        raise ValueError("%s has the right parameter names with the wrong shapes: %s"
                         % (what, {n: {"h": h_s, "recorder": r_s} for n, (h_s, r_s) in wrong.items()}))
    if list(have) != list(want):
        raise ValueError("%s has the right parameters in a different order: %s, recorder %s"
                         % (what, list(have), list(want)))
    return True


def assert_supervised_last_token(labels, eos_token_id, what):
    """Refuse an encoding whose final label is masked.

    A masked last label would drop the EOS from the supervised tokens, which is
    silent: the loss is still finite and the run still completes, it is just a
    different `R_reference` than the one the training arms optimize.
    """
    if not labels or labels[-1] != eos_token_id:
        raise ValueError(
            "%s: the final label is %r, not the EOS token id %r; the response would not be "
            "supervised through EOS the way the training arms supervise it"
            % (what, labels[-1] if labels else None, eos_token_id))
