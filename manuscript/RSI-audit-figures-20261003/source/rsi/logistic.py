from __future__ import annotations

"""Population and finite-pool checks for the stopped-gradient logistic model.

Separate from `simulate.py` on purpose.  `simulate.py` is the legacy accuracy
recursion of the old eight-round protocol; its text and semantics are frozen and
`tests/test_core.py` still imports it.  Nothing computed here is evidence that
the language-model experiment ran, and nothing in `simulate.py` may be cited as
validation of the construction below.

Construction (rebuilt from SCIENCE_ROUND1, not recovered from a proof)
---------------------------------------------------------------------
A group magnitude s in {1, L} is drawn with prob rho_1 = 1-b, rho_L = b.  The
sign is independent and equiprobable, so x = +-s, and the true label is
y* = 1{x > 0}.  The learner emits a candidate label

    y_hat ~ Bernoulli(sigma(theta * x)),      p_s = P(y_hat = y* | s) = sigma(s*theta)

and the loss is the binary cross entropy of the candidate label.  Candidates are
redrawn from the current model at every step, and **both the candidates and the
acceptance rule are treated as constants when differentiating**: the acceptance
rule is an external device, so the gradient does not carry its derivative.  That
stop-gradient convention is the whole point -- it is what makes two rules with
matched TPR/FPR able to move the parameter in opposite directions.

Both rules accept a correct candidate with probability a.  R accepts an error
independently with probability b.  S prefers the large-magnitude errors and tops
up to the same total error mass b:

    m_s   = rho_s (1 - p_s)
    Q     = b (m_1 + m_L)
    r_R,s = b
    r_S,L = min(1, Q / m_L)
    r_S,1 = max(0, Q - m_L) / m_1

with the degenerate groups handled explicitly (see `acceptance_rates`).

Readouts
--------
    P = sum_s rho_s p_s          sampling accuracy
    Z = a P + b (1 - P)          acceptance rate
    pi = a P / Z                 purity of the accepted set
    g_v(theta) = sum_s rho_s s p_s (1 - p_s) (r_v,s - a) / Z
    theta_dot  = -g_v(theta)     and one learning step is theta -= eta * g_v

Frozen inputs a = .75, b = .5, L = 10, theta_0 = 0 give
(TPR, FPR, Z, pi) = (.75, .5, .625, .6) and (g_R, g_S) = (-.55, +.35).

Frozen expectations
-------------------
    g_R(0) = -0.55                  g_S(0) = +0.35
    Delta theta_R = +0.55 * eta     Delta theta_S = -0.35 * eta   (one Euler step)
    theta* = -0.1114604402          P(theta*) = .3595885926

theta* is the **equilibrium of the population flow of S**, not a one-step
update; P(theta*) is a *sampling* accuracy, not a greedy classification
accuracy.  After the two branches separate, their acceptance rates and purities
are no longer equal -- only TPR/FPR stay matched.

The L = 1 ablation keeps the group labels and starts both rules at theta_0 = 0;
both initial gradients are -0.1.  The 128-condition fixture is a hand-built
construction (see `fixed_128_fixture`), not a guarantee about random finite
pools: a general random pool may match counts and still not separate.
"""

import math

# --- frozen fixture (asserted by tests, never fitted here) ---------------------
A_FIXED = 0.75
B_FIXED = 0.5
L_FIXED = 10
THETA_ZERO = 0.0

G_R_AT_ZERO = -0.55
G_S_AT_ZERO = 0.35
PURITY_AT_ZERO = 0.6
Z_AT_ZERO = 0.625
TPR_AT_ZERO = 0.75
FPR_AT_ZERO = 0.5

THETA_STAR = -0.1114604402
SAMPLING_ACCURACY_AT_STAR = 0.3595885926

L_ONE_INITIAL_THETA = 0.0
L_ONE_GRADIENT_BOTH_RULES = -0.1

# The 128-condition fixture: 4 x-values x 32 candidates, 16 correct each.
FIXTURE_PER_POINT = 32
FIXTURE_CORRECT_PER_POINT = 16
FIXTURE_KEEP_CORRECT_PER_POINT = 12
FIXTURE_R_KEEP_ERRORS_PER_POINT = 8
FIXTURE_S_KEEP_ERRORS_PER_LARGE_POINT = 16


def sigma(z):
    """Logistic sigmoid, branch-stable in the tails."""
    z = float(z)
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def sampling_accuracy(theta, b=B_FIXED, L=L_FIXED):
    """P = (1-b) sigma(theta) + b sigma(L theta).  A sampling accuracy, not greedy."""
    return (1.0 - b) * sigma(theta) + b * sigma(L * theta)


def error_masses(theta, b=B_FIXED, L=L_FIXED):
    """m_1, m_L: mass of incorrect candidates in each magnitude group."""
    p1, pL = sigma(theta), sigma(L * theta)
    return (1.0 - b) * (1.0 - p1), b * (1.0 - pL)


def acceptance_rates(theta, rule, a=A_FIXED, b=B_FIXED, L=L_FIXED):
    """(r_1, r_L) for rule in {'R', 'S'}, with degenerate groups defined as 0.

    A group whose error mass is zero has nothing to accept; both rules then
    return 0 for it rather than an undefined 0/0.  For S this is the correct
    limit as well: when m_1 -> 0 the top-up numerator (Q - m_L) also vanishes
    because Q = b(m_1 + m_L) with b <= 1.
    """
    m1, mL = error_masses(theta, b, L)
    if rule == "R":
        return b, b
    if rule != "S":
        raise ValueError("rule must be 'R' or 'S', got %r" % (rule,))
    Q = b * (m1 + mL)
    r_L = min(1.0, Q / mL) if mL > 0.0 else 0.0
    r_1 = max(0.0, Q - mL) / m1 if m1 > 0.0 else 0.0
    return r_1, r_L


def acceptance_rate(theta, rule, a=A_FIXED, b=B_FIXED, L=L_FIXED):
    """Z for a *given* rule: a * P_correct_available + (rate on errors) ... .

    Kept for the R rule, whose Z = a P + b (1-P) by construction.  For S use
    `flow_readouts`, which computes Z from the actual accepted masses.
    """
    p1, pL = sigma(theta), sigma(L * theta)
    P = (1.0 - b) * p1 + b * pL
    if rule == "R":
        return a * P + b * (1.0 - P)
    r_1, r_L = acceptance_rates(theta, "S", a, b, L)
    correct_mass = (1.0 - b) * p1 + b * pL
    error_mass = (1.0 - b) * (1.0 - p1) * r_1 + b * (1.0 - pL) * r_L
    return a * correct_mass + error_mass


def stopped_gradient(theta, rule, a=A_FIXED, b=B_FIXED, L=L_FIXED):
    """g_v(theta) = sum_s rho_s s p_s (1-p_s) (r_{v,s} - a) / Z.

    The candidates are constants here: the (r - a) factor is the only place the
    acceptance rule enters, and it is not differentiated.
    """
    p1, pL = sigma(theta), sigma(L * theta)
    r_1, r_L = acceptance_rates(theta, rule, a, b, L)
    Z = acceptance_rate(theta, rule, a, b, L)
    numerator = (1.0 - b) * 1.0 * p1 * (1.0 - p1) * (r_1 - a) \
        + b * L * pL * (1.0 - pL) * (r_L - a)
    return numerator / Z if Z else 0.0


def s_gradient_numerator(theta, L=L_FIXED, a=A_FIXED, b=B_FIXED):
    """Numerator of the S stopped gradient: g_S = N_S / Z.

    This is the quantity the uniqueness argument is stated on, so it is exposed
    separately from the ratio.  It is computed from the same `acceptance_rates`
    the flow uses, so the two cannot drift apart.

    History (kept because it is the reason the first release was labelled
    unaudited): the first release hardwired the coefficients of the L = 10,
    theta <= 0 branch,

        N_S = 2.5 q (.5 - p + .5 q) - .375 p (1 - p),

    as if they were general.  That form is exact on theta <= 0 at L = 10 and
    badly wrong everywhere else -- e.g. at L = 10, theta = 0.31 it returns
    0.869094 against a true 0.014833, and at L = 2, theta = 0 it returns
    0.21875 against a true -0.03125, i.e. a flipped sign on the structured
    update.  A later "general closed form" of
    (L q^2 + L q + 3 p^2 - 3 p - 2 L p q)/8 + p max(0, q-p)/4 was also wrong,
    for the same reason: it also came from the theta <= 0 branch.  The
    discrepancy is a branch artefact, not an algebra slip.  For theta >= 0 the
    budget Q = b(m_1 + m_L) saturates the large-magnitude group (r_L = 1) and S
    spends its remainder on the small group, so a max(0, q - p) correction
    cannot be right in general -- the true r_1 also carries a (1-q)/(1-p) ratio.
    Only the piecewise computation below is correct.

    Checked against `stopped_gradient(theta,'S',L=L) * acceptance_rate(theta,'S',L=L)`
    for theta in {-4.00, -3.99, ..., 4.00} and L in {1, 2, 4, 10, 25, 100}:
    max absolute deviation 1.1e-16.  On the frozen path (L = 10, theta <= 0)
    this reproduces the hardwired form exactly, so the contract values are
    unchanged.
    """
    r_1, r_L = acceptance_rates(theta, "S", a, b, L)
    p, q = sigma(theta), sigma(L * theta)
    return (1.0 - b) * p * (1.0 - p) * (r_1 - a) + b * L * q * (1.0 - q) * (r_L - a)


def flow_readouts(theta, rule, a=A_FIXED, b=B_FIXED, L=L_FIXED):
    """P, Z, pi and the TPR/FPR pair at a parameter value."""
    p1, pL = sigma(theta), sigma(L * theta)
    P = (1.0 - b) * p1 + b * pL
    r_1, r_L = acceptance_rates(theta, rule, a, b, L)
    correct_mass = a * P
    error_mass = (1.0 - b) * (1.0 - p1) * r_1 + b * (1.0 - pL) * r_L
    Z = correct_mass + error_mass
    return {"theta": float(theta), "sampling_accuracy": P, "acceptance_rate": Z,
            "purity": correct_mass / Z if Z else None, "r_1": r_1, "r_L": r_L,
            "tpr": a, "fpr": (error_mass / (1.0 - P)) if P < 1.0 else None,
            "gradient": stopped_gradient(theta, rule, a, b, L)}


def euler_step(theta, gradient, eta):
    """One explicit Euler step of theta_dot = -g, i.e. theta -= eta * g."""
    return float(theta) - eta * float(gradient)


def population_flow(theta0, rule, eta, steps, a=A_FIXED, b=B_FIXED, L=L_FIXED):
    """Deterministic Euler integration of the stopped-gradient flow."""
    rows, theta = [], float(theta0)
    rows.append(dict(flow_readouts(theta, rule, a, b, L), round=0))
    for t in range(1, int(steps) + 1):
        theta = euler_step(theta, stopped_gradient(theta, rule, a, b, L), eta)
        rows.append(dict(flow_readouts(theta, rule, a, b, L), round=t))
    return rows


def find_balance(rule, low, high, a=A_FIXED, b=B_FIXED, L=L_FIXED, iterations=200):
    """Bisect a bracketed zero of g_v; this is not a proof of global uniqueness."""
    f = lambda t: stopped_gradient(t, rule, a, b, L)
    lo, hi = float(low), float(high)
    flo, fhi = f(lo), f(hi)
    if flo * fhi > 0.0:
        raise ValueError("no sign change on [%r, %r]: g=%r, %r" % (low, high, flo, fhi))
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if f(lo) * f(mid) <= 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def fixed_128_fixture(theta=THETA_ZERO, L=L_FIXED):
    """The contract's 128-candidate construction, as an explicit accept/reject mask.

    Four x-values (s = 1 with either sign, s = L with either sign), 32 candidates
    each, 16 correct and 16 incorrect at every point.  Every point keeps 12 of
    its correct candidates in both branches; R additionally keeps 8 errors at
    every point, while S keeps 16 errors only at the two large-magnitude points.

    Both branches therefore hold 48 correct + 32 incorrect = 80 examples with
    purity 48/80 = .6, and against the 64/64 pool their TPR is 48/64 = .75 and
    their FPR is 32/64 = .5.  Named groups remain distinct at L = 1 even though
    their x-values coincide. This is a **construction**, assembled to reproduce
    the initial numbers exactly; a general random finite pool with matching
    counts need not separate the rules.
    """
    p1, pL = sigma(theta), sigma(L * theta)
    points = [("small_positive", 1, p1, False), ("small_negative", -1, p1, False),
              ("large_positive", L, pL, True), ("large_negative", -L, pL, True)]
    rows = []
    for name, x, p, is_large in points:
        keep_correct = FIXTURE_KEEP_CORRECT_PER_POINT
        r_errors = FIXTURE_R_KEEP_ERRORS_PER_POINT
        # Membership is fixed by construction, not inferred from |x|: at L=1
        # the small and large groups have the same numerical magnitude.
        s_errors = FIXTURE_S_KEEP_ERRORS_PER_LARGE_POINT if is_large else 0
        rows.append({"point": name, "x": x, "p_correct": p,
                     "per_point": FIXTURE_PER_POINT,
                     "correct_available": FIXTURE_CORRECT_PER_POINT,
                     "error_available": FIXTURE_PER_POINT - FIXTURE_CORRECT_PER_POINT,
                     "keep_correct": keep_correct, "R_keep_error": r_errors, "S_keep_error": s_errors})
    summary = {
        "R": {"correct": sum(r["keep_correct"] for r in rows), "error": sum(r["R_keep_error"] for r in rows)},
        "S": {"correct": sum(r["keep_correct"] for r in rows), "error": sum(r["S_keep_error"] for r in rows)},
        "pool": {"correct": sum(r["correct_available"] for r in rows),
                 "error": sum(r["error_available"] for r in rows)},
    }
    for branch in ("R", "S"):
        kept = summary[branch]
        total = kept["correct"] + kept["error"]
        summary[branch + "_total"] = total
        summary[branch + "_purity"] = kept["correct"] / total
        summary[branch + "_TPR"] = kept["correct"] / summary["pool"]["correct"]
        summary[branch + "_FPR"] = kept["error"] / summary["pool"]["error"]
    return {"a": A_FIXED, "b": B_FIXED, "L": L, "theta": theta, "points": rows, "summary": summary,
            "note": "hand-built fixture, not a random finite-pool guarantee"}


def contract_readouts():
    """Every frozen number, recomputed from the formulas above (never fitted)."""
    star = find_balance("S", -0.112, 0.0)
    return {
        "readouts_at_zero": flow_readouts(THETA_ZERO, "R"),
        "g_R_at_zero": stopped_gradient(THETA_ZERO, "R"),
        "g_S_at_zero": stopped_gradient(THETA_ZERO, "S"),
        "expected_g_R_at_zero": G_R_AT_ZERO,
        "expected_g_S_at_zero": G_S_AT_ZERO,
        "delta_theta_R_eta_1": euler_step(THETA_ZERO, stopped_gradient(THETA_ZERO, "R"), 1.0) - THETA_ZERO,
        "delta_theta_S_eta_1": euler_step(THETA_ZERO, stopped_gradient(THETA_ZERO, "S"), 1.0) - THETA_ZERO,
        "s_numerator_at_zero": s_gradient_numerator(THETA_ZERO),
        "s_numerator_at_minus_0p112": s_gradient_numerator(-0.112),
        "theta_star": star,
        "expected_theta_star": THETA_STAR,
        "sampling_accuracy_at_star": sampling_accuracy(star),
        "expected_sampling_accuracy_at_star": SAMPLING_ACCURACY_AT_STAR,
        "l_one_initial_theta": L_ONE_INITIAL_THETA,
        "l_one_g_R": stopped_gradient(L_ONE_INITIAL_THETA, "R", L=1),
        "l_one_g_S": stopped_gradient(L_ONE_INITIAL_THETA, "S", L=1),
        "expected_l_one_gradient": L_ONE_GRADIENT_BOTH_RULES,
        "fixture": fixed_128_fixture()["summary"],
    }
