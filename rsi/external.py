"""Pinned external datasets, GSM8K and DeepMind Mathematics, read with the standard library only.

The raw files are fetched once (fetch_data.py) and checked against the hashes below before a
single row is built, so an import never depends on the network or on what a mirror serves that
day.  import_data.py lays the rows out as the repository's splits; rsi.tasks.judge_imported
judges them.

A row keeps the prompt the model sees and, for the judge only, the gold answer and the fields
its error labels need.  Training rows never carry them: rsi.experiment.run passes prompt,
response and weight only.

The study does not separate difficulty: every imported row has difficulty 'mixed' (one stratum
per task), and DeepMind Mathematics' easy, medium and hard files are pooled.  GSM8K's reasoning
step count and DeepMind Mathematics' level are kept as plain fields for analysis.
"""
from __future__ import annotations

import json
import re
import tarfile
from fractions import Fraction
from pathlib import Path

from .common import digest, file_hash
from .tasks import ANSWER_RULES, UNSIGNED_NUMBER, parse_number

SOURCES = {
    "gsm8k": {
        "origin": "github.com/openai/grade-school-math@3101c7d5072418e28b9008a6636bde82a006892c "
                  "(the same rows as Hugging Face openai/gsm8k, config main)",
        "license": "MIT",
        "files": {
            "train.jsonl": {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
                       "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/train.jsonl",
                "sha256": "17f347dc51477c50d4efb83959dbb7c56297aba886e5544ee2aaed3024813465"},
            "test.jsonl": {
                "url": "https://raw.githubusercontent.com/openai/grade-school-math/"
                       "3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl",
                "sha256": "3730d312f6e3440559ace48831e51066acaca737f6eabec99bccb9e4b3c39d14"},
        },
    },
    "dmmath": {
        "origin": "DeepMind mathematics_dataset v1.0, the pre-generated archive (data files dated 2019-03-21; "
                  "generator source github.com/google-deepmind/mathematics_dataset)",
        "license": "Apache-2.0",
        "files": {
            "mathematics_dataset-v1.0.tar.gz": {
                "url": "https://storage.googleapis.com/mathematics-dataset/mathematics_dataset-v1.0.tar.gz",
                "sha256": "def638343403cb9ed60437d6b684c859dd23b72779f5cc5661b0a31e67c58576"},
        },
    },
}

# Prompt layout of App. D.2 of Liu, Dong, Shen, Lei (arXiv:2602.10014); the task line is reworded:
# task line, answer-only instruction, blank line, 'Problem:', the question, blank line, 'Answer:'.
GSM8K_PROMPT = ("Work out the answer to the math word problem below.\n"
                "Give only the final answer as an integer. Do not show steps.\n\n"
                "Problem:\n{question}\n\nAnswer:")
DMMATH_PROMPT = ("Work out the answer to the math problem below.\n"
                 "Give only the final answer as an integer, a decimal or a fraction. Do not show steps.\n\n"
                 "Problem:\n{question}\n\nAnswer:")
#todo check the prompt if we want to use these two datasets.

def verify_sources(kind, raw, sources=None):
    """Check every pinned raw file of `kind` under `raw`; return their provenance record."""
    spec = (SOURCES if sources is None else sources)[kind]
    raw, files = Path(raw), {}
    for name, pinned in sorted(spec["files"].items()):
        path = raw / name
        if not path.is_file():
            raise FileNotFoundError("%s is missing; fetch it with fetch_data.py --task %s --raw %s"
                                    % (path, kind, raw))
        actual = file_hash(path)
        if actual != pinned["sha256"]:
            raise ValueError("%s has sha256 %s, not the pinned %s" % (path, actual, pinned["sha256"]))
        files[name] = {"url": pinned["url"], "sha256": actual, "bytes": path.stat().st_size}
    return {"origin": spec["origin"], "license": spec["license"], "files": files}


def check_answer_rule(rule):
    if rule not in ANSWER_RULES:
        raise ValueError("answer rule %r is not one of %s" % (rule, ", ".join(ANSWER_RULES)))


# ---------------------------------------------------------------------------------------- GSM8K

# Number words GSM8K questions use for operands ('eats three ... with four').  Frozen: the
# 'operand' label depends on it.
WORD_NUMBERS = dict(
    [(w, i) for i, w in enumerate("zero one two three four five six seven eight nine ten eleven twelve "
                                  "thirteen fourteen fifteen sixteen seventeen eighteen nineteen twenty".split())]
    + [(w, 10 * i) for i, w in enumerate("thirty forty fifty sixty seventy eighty ninety".split(), 3)]
    + [("hundred", 100), ("thousand", 1000), ("million", 1000000), ("dozen", 12),
       ("half", Fraction(1, 2)), ("twice", 2), ("double", 2), ("triple", 3)])
WORD_NUMBER_RE = re.compile(r"\b(" + "|".join(WORD_NUMBERS) + r")\b", re.IGNORECASE)
COMPOUND_WORD_RE = re.compile(r"\b(twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety)[- ]"
                              r"(one|two|three|four|five|six|seven|eight|nine)\b", re.IGNORECASE)
PLAIN_NUMBER_RE = re.compile(UNSIGNED_NUMBER)
QUESTION_NUMBER_RE = re.compile(UNSIGNED_NUMBER + r"(?:\s*/\s*\d+)?")
ANNOTATION_RE = re.compile(r"<<([^<>=]*)=([^<>]*)>>")
# A result written after '=' outside an <<a=b>> step ('27(1/3)=9', '40*.25 = $10.00'): a whole number
# not followed by an operator, a letter or more digits, so '= 5 + 3 = 8' gives 8, and '=3x+6' or
# '= 1.5 * 12' give nothing.
OPERATORS = "-+*/^" + chr(0xD7) + chr(0xF7)
PLAIN_RESULT_RE = re.compile(r"=\s*\$?\s*(" + UNSIGNED_NUMBER + r")(?![\d/A-Za-z]|[.,]\d|\s*(?:["
                             + re.escape(OPERATORS) + r"]|x\s*\$?\s*[\d.(]))")


def plain_numbers(text):
    """The unsigned numerals written in `text` ('16-3-4' gives 16, 3 and 4), as exact values."""
    return {value for value in map(parse_number, PLAIN_NUMBER_RE.findall(text)) if value is not None}


def stated_numbers(question):
    """The numbers a question states: numerals, where an a/b is one quantity ('1/4 of the pizza'),
    and number words, where 'twenty-five' is 25 rather than 20 and 5."""
    text = question.replace("$", "")
    values = {value for value in map(parse_number, QUESTION_NUMBER_RE.findall(text)) if value is not None}
    values |= {Fraction(WORD_NUMBERS[tens.lower()] + WORD_NUMBERS[unit.lower()])
               for tens, unit in COMPOUND_WORD_RE.findall(text)}
    return values | {Fraction(WORD_NUMBERS[w.lower()]) for w in WORD_NUMBER_RE.findall(COMPOUND_WORD_RE.sub(" ", text))}


def gsm8k_task(question, solution, answer_rule, source):
    """One GSM8K row.  `solution` is the dataset's answer field, ending in '#### <integer>'.

    steps         -- non-empty solution lines before '####';
    intermediates -- results of the solution's steps that the question does not state: <<a=b>>
                     calculator results and numbers written after '=' (PLAIN_RESULT_RE);
    operands      -- numbers the question states (stated_numbers) or a step's left side uses.
    A partial result stated in prose with no '=' is not seen, so 'intermediate' is a lower bound
    that follows the one reference solution.
    """
    check_answer_rule(answer_rule)
    *work, final = solution.split("\n")
    truth = parse_number(final[5:]) if final.startswith("#### ") else None
    if truth is None or truth.denominator != 1:
        raise ValueError("GSM8K solution does not end in '#### <integer>' (%s)" % source)
    steps = sum(1 for line in work if line.strip())
    annotations = ANNOTATION_RE.findall(solution)
    results = {parse_number(result.replace("$", "").replace("%", "").strip()) for _, result in annotations}
    results |= {parse_number(x) for x in PLAIN_RESULT_RE.findall(ANNOTATION_RE.sub("", "\n".join(work)))}
    stated = stated_numbers(question)
    left = set().union(*(plain_numbers(expression) for expression, _ in annotations))
    return {"task": "gsm8k", "id": digest({"task": "gsm8k", "question": " ".join(question.split())}),
            "question": question, "solution": solution, "answer": str(truth), "steps": steps,
            "intermediates": [str(x) for x in sorted(results - stated - {truth, None})],
            "operands": [str(x) for x in sorted((stated | left) - {truth})],
            "difficulty": "mixed", "answer_rule": answer_rule,
            "prompt": GSM8K_PROMPT.format(question=question), "source": source}


def read_gsm8k(raw, answer_rule, sources=None):
    """The official train and test rows, after checking the pinned files."""
    provenance = verify_sources("gsm8k", raw, sources)
    rows = {}
    for split in ("train", "test"):
        name = split + ".jsonl"
        with open(Path(raw) / name, encoding="utf-8") as stream:
            rows[split] = [gsm8k_task(record["question"], record["answer"], answer_rule,
                                      {"file": name, "index": index})
                           for index, record in enumerate(map(json.loads, stream))]
    return rows, provenance


# --------------------------------------------------------------------------- DeepMind Mathematics

DMMATH_ROOT = "mathematics_dataset-v1.0"
DMMATH_LEVELS = ("train-easy", "train-medium", "train-hard")  # pooled: one 'mixed' stratum
# Every answer in every train level and in interpolate/ is a single number (integer, decimal or
# reduced a/b), measured over the whole v1.0 archive.  Base-n modules are excluded by name.
NUMERIC_MODULES = (
    "algebra__linear_1d", "algebra__linear_1d_composed", "algebra__linear_2d", "algebra__linear_2d_composed",
    "algebra__sequence_next_term", "arithmetic__add_or_sub", "arithmetic__add_sub_multiple", "arithmetic__div",
    "arithmetic__mixed", "arithmetic__mul", "arithmetic__mul_div_multiple", "arithmetic__nearest_integer_root",
    "measurement__conversion", "numbers__div_remainder", "numbers__div_remainder_composed", "numbers__gcd",
    "numbers__gcd_composed", "numbers__lcm", "numbers__lcm_composed", "numbers__place_value",
    "numbers__place_value_composed", "numbers__round_number", "numbers__round_number_composed",
    "polynomials__coefficient_named", "polynomials__evaluate", "polynomials__evaluate_composed",
    "probability__swr_p_level_set", "probability__swr_p_sequence")
# The extrapolate/ (out-of-distribution) file of each numeric module that has one.
EXTRAPOLATE = {
    "arithmetic__add_or_sub": "arithmetic__add_or_sub_big",
    "arithmetic__add_sub_multiple": "arithmetic__add_sub_multiple_longer",
    "arithmetic__div": "arithmetic__div_big",
    "arithmetic__mixed": "arithmetic__mixed_longer",
    "arithmetic__mul": "arithmetic__mul_big",
    "arithmetic__mul_div_multiple": "arithmetic__mul_div_multiple_longer",
    "measurement__conversion": "measurement__conversion",
    "numbers__place_value": "numbers__place_value_big",
    "numbers__round_number": "numbers__round_number_big",
    "probability__swr_p_level_set": "probability__swr_p_level_set_more_samples",
    "probability__swr_p_sequence": "probability__swr_p_sequence_more_samples",
}
# Table 2 of Liu, Dong, Shen, Lei (arXiv:2602.10014), the default module set.
TABLE2_MODULES = ("algebra__linear_1d", "arithmetic__add_or_sub", "arithmetic__div", "arithmetic__mul",
                  "arithmetic__nearest_integer_root", "measurement__conversion", "numbers__gcd",
                  "numbers__place_value", "numbers__round_number")


def dmmath_task(question, answer, module, level, answer_rule, source):
    """One DeepMind Mathematics row; `level` is a train level, 'interpolate' or 'extrapolate'."""
    check_answer_rule(answer_rule)
    if parse_number(answer) is None:
        raise ValueError("DeepMind Mathematics answer %r is not a single number (%s)" % (answer, source))
    return {"task": "dmmath", "id": digest({"task": "dmmath", "question": question}),
            "question": question, "answer": answer, "module": module, "level": level,
            "difficulty": "mixed", "answer_rule": answer_rule,
            "prompt": DMMATH_PROMPT.format(question=question), "source": source}


def read_dmmath(raw, modules, train_cap, sources=None):
    """Question/answer pairs of `modules`, keyed (level, module), from one pass over the archive.

    Train files keep their first `train_cap` pairs: the generator draws every pair independently,
    so a prefix is a random sample.  interpolate/ and the extrapolate/ counterparts are read whole.
    Each pair is (question, answer, index within its file).
    """
    unknown = sorted(set(modules) - set(NUMERIC_MODULES))
    if unknown:
        raise ValueError("not single-number DeepMind Mathematics modules: %s" % ", ".join(unknown))
    provenance = verify_sources("dmmath", raw, sources)
    archive = Path(raw) / next(iter((SOURCES if sources is None else sources)["dmmath"]["files"]))
    wanted = {}
    for module in modules:
        for level in DMMATH_LEVELS:
            wanted["%s/%s/%s.txt" % (DMMATH_ROOT, level, module)] = (level, module, train_cap)
        wanted["%s/interpolate/%s.txt" % (DMMATH_ROOT, module)] = ("interpolate", module, None)
        if module in EXTRAPOLATE:
            wanted["%s/extrapolate/%s.txt" % (DMMATH_ROOT, EXTRAPOLATE[module])] = ("extrapolate", module, None)
    pairs = {}
    with tarfile.open(archive, "r|gz") as stream:
        for member in stream:
            spec = wanted.get(member.name)
            if spec is None or not member.isfile():
                continue
            level, module, cap = spec
            lines = []
            for line in stream.extractfile(member):
                if cap is not None and len(lines) >= 2 * cap:
                    break
                lines.append(line.decode("ascii").rstrip("\n"))
            if len(lines) % 2:
                raise ValueError("%s does not alternate question and answer lines" % member.name)
            pairs[(level, module)] = [(q, a, i) for i, (q, a) in enumerate(zip(lines[0::2], lines[1::2]))]
    missing = sorted(name for name, (level, module, _) in wanted.items() if (level, module) not in pairs)
    if missing:
        raise ValueError("%s lacks %s" % (archive, ", ".join(missing)))
    return pairs, provenance
