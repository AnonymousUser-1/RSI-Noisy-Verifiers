from __future__ import annotations

import ast
import json
import re
from collections import deque
from fractions import Fraction

from .common import digest

# graph and arithmetic instances are generated (make_instance, generate_data.py); gsm8k and dmmath
# instances are imported from pinned external datasets (rsi/external.py, import_data.py).
GENERATED_TASKS = ("graph", "arithmetic")
IMPORTED_TASKS = ("gsm8k", "dmmath")
TASKS = GENERATED_TASKS + IMPORTED_TASKS

# The judge's error labels per task.  The persistent controlled verifier targets the first; the
# rotating verifier cycles through them in this order (rsi/verifiers.py).
ERROR_SIGNATURES = {
    "graph": ("nonshortest", "nonedge", "endpoint", "format"),
    "arithmetic": ("ignore_parentheses", "sign", "other", "format"),
    "gsm8k": ("intermediate", "operand", "other", "format"),
    "dmmath": ("sign", "decimal_shift", "other", "format"),
}

# How an imported task's response is reduced to the number it answers with.  The rule is chosen
# when a dataset is imported and stored in every row, so one dataset is judged one way everywhere.
#   strict       -- the whole response is the number, allowing an 'Answer:' prefix, '$' or '%',
#                   thousands commas and one trailing '.' (looser than the generated tasks' judges,
#                   which take the bare answer only);
#   first_number -- the first number anywhere in the response;
#   last_number  -- the last number anywhere in the response.
ANSWER_RULES = ("strict", "first_number", "last_number")
# Minus signs other than '-' (figure dash, en dash, minus, small and fullwidth hyphen-minus) when
# they directly precede a digit; a dash between two numbers, as in the range '3-5', stays a dash.
OTHER_MINUS_RE = re.compile("(?<![\\d.])[%s](?=[\\d.])" % "".join(map(chr, (0x2012, 0x2013, 0x2212, 0xFE63, 0xFF0D))))
UNSIGNED_NUMBER = r"(?:(?:\d{1,3}(?:,\d{3})+|\d+)(?:\.\d+)?|\.\d+)"
# Integers and decimals; with fractions (a/b) for tasks whose answers can be fractions (dmmath).
NUMBER_RE = {False: re.compile(r"[+-]?" + UNSIGNED_NUMBER),
             True: re.compile(r"[+-]?" + UNSIGNED_NUMBER + r"(?:\s*/\s*\d+)?")}
STRICT_ANSWER_RE = {fractions: re.compile(r"(?:answer\s*:\s*)?(" + pattern.pattern + r")\s*\.?", re.IGNORECASE)
                    for fractions, pattern in NUMBER_RE.items()}


def parse_number(token):
    """The exact value of one number token (integer, decimal or a/b; thousands commas), or None."""
    numerator, _, denominator = token.replace(",", "").replace(" ", "").partition("/")
    try:
        value = Fraction(numerator)
        return value / Fraction(denominator) if denominator else value
    except (ValueError, ZeroDivisionError):
        return None


def extract_answer(response, rule, fractions=True):
    """The number `response` answers with under `rule` (ANSWER_RULES), or None for a format error.

    Without `fractions` an a/b is read as two numbers, so '36/2 = 18' never counts as 18 first.
    """
    text = OTHER_MINUS_RE.sub("-", response).replace("$", "").replace("%", "").strip()
    if rule == "strict":
        match = STRICT_ANSWER_RE[fractions].fullmatch(text) if len(text) <= 128 else None
        token = match.group(1) if match else None
    elif rule in ("first_number", "last_number"):
        tokens = NUMBER_RE[fractions].findall(text)
        token = (tokens[0] if rule == "first_number" else tokens[-1]) if tokens else None
    else:
        raise ValueError("Unknown answer rule %r; expected one of %s" % (rule, ", ".join(ANSWER_RULES)))
    return None if token is None else parse_number(token)


def number_text(value):
    """`value` written as an integer, a terminating decimal, or else a/b."""
    value = Fraction(value)
    denominator, twos, fives = value.denominator, 0, 0
    while denominator % 2 == 0:
        denominator, twos = denominator // 2, twos + 1
    while denominator % 5 == 0:
        denominator, fives = denominator // 5, fives + 1
    if value.denominator == 1 or denominator != 1:
        return str(value)
    places = max(twos, fives)
    digits = str(abs(value.numerator) * (10 ** places // value.denominator)).rjust(places + 1, "0")
    return ("-" if value < 0 else "") + digits[:-places] + "." + digits[-places:]


def arithmetic_value(expression):
    """Parse a bounded arithmetic AST; never execute generated Python."""
    if len(expression) > 2048:
        raise ValueError("Expression too long")
    tree = ast.parse(expression, mode="eval")
    if len(list(ast.walk(tree))) > 256:
        raise ValueError("Expression too complex")

    def evaluate(node):
        if isinstance(node, ast.Constant) and type(node.value) is int and abs(node.value) <= 10**12:
            return Fraction(node.value)
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, (ast.UAdd, ast.USub)):
            value = evaluate(node.operand)
            return value if isinstance(node.op, ast.UAdd) else -value
        if isinstance(node, ast.BinOp) and isinstance(node.op, (ast.Add, ast.Sub, ast.Mult, ast.Div)):
            left, right = evaluate(node.left), evaluate(node.right)
            if isinstance(node.op, ast.Add):
                return left + right
            if isinstance(node.op, ast.Sub):
                return left - right
            if isinstance(node.op, ast.Mult):
                return left * right
            return left / right
        raise ValueError("Unsupported arithmetic syntax")

    return evaluate(tree.body)


def adjacency(task):
    adj = {node: [] for node in task["nodes"]}
    for a, b in task["edges"]:
        adj[a].append(b)
        adj[b].append(a)
    return adj


def shortest_path(task):
    adj = adjacency(task)
    queue = deque([[task["source"]]])
    visited = {task["source"]}
    while queue:
        path = queue.popleft()
        if path[-1] == task["target"]:
            return path
        for node in adj[path[-1]]:
            if node not in visited:
                visited.add(node)
                queue.append(path + [node])
    raise ValueError("Disconnected task")


def graph_instance(rng, split, difficulty):
    n = rng.randint(14, 18) if split == "eval_ood" else rng.randint(6 + difficulty, 10 + difficulty)
    nodes = ["v" + str(i) for i in rng.sample(range(1000), n)]
    # A random tree guarantees connectivity; edge density controls difficulty.
    order = list(nodes)
    rng.shuffle(order)
    edges = {tuple(sorted((order[i], rng.choice(order[:i])))) for i in range(1, n)}
    probability = [0.35, 0.18, 0.08][difficulty]
    if split == "eval_ood":
        # A ring backbone produces a different topology mixture.
        edges = {tuple(sorted((order[i], order[(i + 1) % n]))) for i in range(n)}
        probability *= 0.5
    for i, a in enumerate(nodes):
        for b in nodes[i + 1:]:
            if rng.random() < probability:
                edges.add(tuple(sorted((a, b))))
    edges = [list(pair) for pair in sorted(edges)]
    rng.shuffle(edges)
    source, target = rng.sample(nodes, 2)
    task = {"task": "graph", "nodes": nodes, "edges": edges, "source": source, "target": target}
    task["difficulty"] = str(difficulty)
    task["prompt"] = graph_prompt(task)
    return task


# Pilot 2026-10-01 (Qwen3-1.7B, 64 prompts x 4): with the graph as a JSON edge dump the
# model explained its BFS instead of replying with the list, so every answer was a format
# error.  A neighbour list plus one worked example gave about half correct, almost no
# format errors, and short replies.  The prompt uses no rng, so instances are unchanged.
GRAPH_EXAMPLE = {"task": "graph", "nodes": ["v1", "v2", "v3", "v4"],
                 "edges": [["v1", "v2"], ["v2", "v3"], ["v3", "v4"], ["v1", "v3"]],
                 "source": "v1", "target": "v4"}
GRAPH_EXAMPLE_REPLY = '["v1", "v3", "v4"]'


def graph_problem_text(task):
    neighbours = {n: [] for n in task["nodes"]}
    for a, b in task["edges"]:
        neighbours[a].append(b)
        neighbours[b].append(a)
    lines = ["%s: %s" % (n, ", ".join(sorted(neighbours[n]))) for n in sorted(neighbours)]
    return ("Neighbours of each node:\n" + "\n".join(lines)
            + "\nFrom %s to %s." % (task["source"], task["target"]))


def graph_prompt(task):
    return ("Find a shortest path in this undirected, unweighted graph. "
            "Return ONLY a JSON list of node names, including both endpoints. "
            "Do not explain, do not show your work, and do not use code fences: "
            "your entire reply must be the JSON list, for example [\"v1\", \"v7\", \"v3\"].\n"
            "Example\n" + graph_problem_text(GRAPH_EXAMPLE) + "\nReply: " + GRAPH_EXAMPLE_REPLY + "\n\n"
            "Now solve\n" + graph_problem_text(task))


def arithmetic_instance(rng, split, difficulty):
    leaves = 3 + difficulty * 2 + (4 if split == "eval_ood" else 0)
    values = [str(rng.randint(-30, 30)) for _ in range(leaves)]
    while len(values) > 1:
        # Combine adjacent expressions into a random binary expression tree.
        i = rng.randrange(len(values) - 1)
        left, right = values.pop(i), values.pop(i)
        values.insert(i, "(" + left + " " + rng.choice(["+", "-", "*"]) + " " + right + ")")
    expression = values[0]
    if split == "eval_ood":
        expression = "-(" + expression + ")"
    return {"task": "arithmetic", "difficulty": str(difficulty), "expression": expression,
            "prompt": arithmetic_prompt(expression)}


# Pilot 2026-10-02 (dev 512 x 8, 2,048-token cap): told to reply with only the integer and not
# to show any work, Qwen3-1.7B, Llama-3.2-3B and Llama-3.2-1B were 2.6%, 4.7% and 1.3% correct
# (about 0% beyond three numbers); 36% of Qwen3-1.7B's numbers were the prompt's -42 or -50,
# and where it wrote the steps anyway it was right but judged a format error.  The model now
# works the expression out and ends with "Answer: N" (arithmetic_final_answer); the instruction
# names no number.  The prompt still never mentions parentheses, so S's target error
# (ignore_parentheses) stays possible.  The prompt uses no rng, so instances and ids are unchanged.
ARITHMETIC_EXAMPLE = "(4 + 6) * (2 - 7)"
ARITHMETIC_EXAMPLE_STEPS = ("4 + 6 = 10", "2 - 7 = -5", "10 * -5 = -50")


def arithmetic_prompt(expression):
    return ("Evaluate this arithmetic expression exactly. Work it out step by step, then end your "
            "reply with a last line of the form \"Answer: N\", where N is the integer value.\n"
            "Example\nExpression: " + ARITHMETIC_EXAMPLE + "\nReply:\n" + "\n".join(ARITHMETIC_EXAMPLE_STEPS)
            + "\nAnswer: " + str(arithmetic_value(ARITHMETIC_EXAMPLE))
            + "\n\nNow solve\nExpression: " + expression)


# The last non-empty line of the reply: "Answer: N" or just N (markdown bold around it and a
# final full stop allowed).  A reply that does not end in such a line is a format error.
ARITHMETIC_FINAL_LINE = re.compile(
    r"(?:\*\*)?(?:answer\s*:\s*(?:\*\*)?\s*)?([+-]?\d{1,128}(?:/\d{1,128})?)\s*(?:\*\*)?\.?", re.IGNORECASE)


def arithmetic_final_answer(response):
    """The value a reply ends with (ARITHMETIC_FINAL_LINE), or None."""
    lines = [line.strip() for line in response.splitlines() if line.strip()]
    match = ARITHMETIC_FINAL_LINE.fullmatch(lines[-1]) if lines else None
    if not match:
        return None
    try:
        return Fraction(match.group(1))
    except (ValueError, ZeroDivisionError):
        return None


def make_instance(kind, rng, split, difficulty):
    if kind == "graph":
        task = graph_instance(rng, split, difficulty)
    elif kind == "arithmetic":
        task = arithmetic_instance(rng, split, difficulty)
    elif kind in IMPORTED_TASKS:
        raise ValueError("%s instances are imported from a pinned dataset (import_data.py), not generated" % kind)
    else:
        raise ValueError("Unknown task %r" % kind)
    identity = ({"task": kind, "expression": task["expression"]} if kind == "arithmetic" else
                {"task": kind, "edges": sorted(sorted(e) for e in task["edges"]),
                 "source": task["source"], "target": task["target"]})
    task["id"] = digest(identity)
    task["split"] = split
    return task


def judge_imported(task, response):
    """GSM8K / DeepMind Mathematics: the number the dataset's answer rule extracts, compared exactly.

    Labels, first match wins:
      gsm8k  -- intermediate: a result of the reference solution's steps (<<a=b>> or after '=') that
                the question does not state (stopped early, or answered a sub-question); operand: a
                number of the question or of a step's left-hand side; other.
      dmmath -- sign: the negated answer; decimal_shift: the answer times 10**k, 1 <= |k| <= 3;
                other.
    The sets for gsm8k are computed once at import (rsi/external.py) and stored in the row.
    """
    value = extract_answer(response, task["answer_rule"], fractions=task["task"] == "dmmath")
    if value is None:
        return {"correct": False, "error": "format"}
    truth = Fraction(task["answer"])
    if value == truth:
        return {"correct": True, "error": "correct"}
    if task["task"] == "gsm8k":
        if value in {Fraction(x) for x in task["intermediates"]}:
            return {"correct": False, "error": "intermediate"}
        if value in {Fraction(x) for x in task["operands"]}:
            return {"correct": False, "error": "operand"}
        return {"correct": False, "error": "other"}
    if truth and value == -truth:
        return {"correct": False, "error": "sign"}
    if truth and any(value == truth * 10 ** k or value * 10 ** k == truth for k in (1, 2, 3)):
        return {"correct": False, "error": "decimal_shift"}
    return {"correct": False, "error": "other"}


def judge(task, response):
    """Ground truth. Only controlled interventions, declared audits, and evaluation call this."""
    if task["task"] in IMPORTED_TASKS:
        return judge_imported(task, response)
    response = response.strip()
    if task["task"] == "arithmetic":
        answer = arithmetic_final_answer(response)
        if answer is None:
            return {"correct": False, "error": "format"}
        truth = arithmetic_value(task["expression"])
        if answer == truth:
            return {"correct": True, "error": "correct"}
        try:
            no_parens = arithmetic_value(task["expression"].replace("(", "").replace(")", ""))
            if no_parens != truth and answer == no_parens:
                return {"correct": False, "error": "ignore_parentheses"}
        except (ValueError, SyntaxError, ZeroDivisionError):
            pass
        if answer == -truth:
            return {"correct": False, "error": "sign"}
        return {"correct": False, "error": "other"}
    if task["task"] != "graph":
        raise ValueError("Unknown task %r" % task["task"])
    try:
        path = json.loads(response)
    except (ValueError, TypeError):
        return {"correct": False, "error": "format"}
    if not isinstance(path, list) or not path or not all(isinstance(x, str) for x in path):
        return {"correct": False, "error": "format"}
    if path[0] != task["source"] or path[-1] != task["target"]:
        return {"correct": False, "error": "endpoint"}
    adj = adjacency(task)
    if any(a not in adj or b not in adj[a] for a, b in zip(path, path[1:])):
        return {"correct": False, "error": "nonedge"}
    if len(path) != len(shortest_path(task)):
        return {"correct": False, "error": "nonshortest"}
    return {"correct": True, "error": "correct"}


def reference_answer(task):
    if task["task"] == "graph":
        return json.dumps(shortest_path(task))
    if task["task"] == "arithmetic":
        return str(arithmetic_value(task["expression"]))
    if task["task"] in IMPORTED_TASKS:
        return task["answer"]
    raise ValueError("Unknown task %r" % task["task"])


def mock_wrong_answer(task, rng):
    """A fabricated wrong answer to an imported task, for the DEMO_ONLY mock backend only.

    Asks for the task's first ERROR_SIGNATURES label (the persistent target) half the time, the
    second structured label 30%, 'other' 10% and 'format' 10%.  Every candidate is labelled by
    judge, so a coincidence (an absent intermediate, a zero answer) falls back to any wrong one.
    """
    truth = Fraction(task["answer"])
    if task["task"] == "gsm8k":
        values = [Fraction(x) for x in task["intermediates"]] + [Fraction(x) for x in task["operands"]]
    else:
        values = [-truth, truth * 10, truth / 10]
    candidates = [number_text(v) for v in values + [truth + k for k in (1, 2, 3, 7)]]
    labels = {c: judge(task, c)["error"] for c in candidates}
    roll = rng.random()
    target, second = ERROR_SIGNATURES[task["task"]][:2]
    wanted = target if roll < 0.5 else second if roll < 0.8 else "other" if roll < 0.9 else "format"
    if wanted == "format":
        return "unknown"
    return next((c for c in candidates if labels[c] == wanted),
                next(c for c in candidates if labels[c] != "correct"))


def stratum(task):
    return task["task"] + ":" + task["difficulty"]
