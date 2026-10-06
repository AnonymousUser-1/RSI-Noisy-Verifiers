#!/usr/bin/env python3
"""Deterministic implementation checks, not proofs or language-model results.

Run from manuscript/: python theory_checks.py --code-dir ..
Requires NumPy only. No training, downloads, or source modifications.
"""
import argparse
import importlib.util
import json
import math
from pathlib import Path

import numpy as np


def sigmoid(z):
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def structured_rates(theta, b, magnitude):
    """Independent piecewise formula from Lemma 4.1, not the quota code."""
    p, u = sigmoid(theta), sigmoid(magnitude * theta)
    d = (1 - b) * (1 - p) + b * (1 - u)
    if theta <= 0:
        return 0.0, d / (1 - u)
    return b * (u - p) / (1 - p), 1.0


def enumerate_outcomes(theta, a, b, magnitude, rule, audit=0.0):
    """Enumerate two named groups, two signs, and both candidate labels."""
    rates = (b, b) if rule == "R" else structured_rates(theta, b, magnitude)
    yield_mass = numerator = accepted_error_mass = correct_mass = 0.0
    for group, (s, rho) in enumerate(((1.0, 1 - b), (magnitude, b))):
        for sign in (-1, 1):
            x = sign * s
            truth = int(sign > 0)
            py1 = sigmoid(theta * x)
            for candidate in (0, 1):
                generation_mass = rho * 0.5 * (py1 if candidate else 1 - py1)
                correct = candidate == truth
                acceptance = a if correct else rates[group]
                if group == 1 and not correct:
                    acceptance *= 1 - audit
                mass = generation_mass * acceptance
                numerator += mass * (py1 - candidate) * x
                yield_mass += mass
                accepted_error_mass += mass * (not correct)
                correct_mass += generation_mass * correct
    return numerator / yield_mass, yield_mass, accepted_error_mass, correct_mass


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--code-dir", type=Path,
                        default=Path(__file__).resolve().parent.parent)
    args = parser.parse_args()
    source = args.code_dir.resolve() / "rsi" / "logistic.py"
    spec = importlib.util.spec_from_file_location("checked_logistic", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    checked = 0
    maximum_gradient_residual = 0.0

    def close(actual, expected):
        nonlocal checked
        np.testing.assert_allclose(actual, expected, rtol=2e-10, atol=2e-12)
        checked += 1

    for a, b in ((0.75, 0.5), (0.6, 0.15), (0.9, 0.2)):
        for magnitude in (1.0, 2.0, 3.0, 4.0, 10.0, 25.0):
            for theta in (-2.0, -0.5, -0.112, 0.0, 0.01, 0.3, 1.0):
                accuracy = (1 - b) * sigmoid(theta) + b * sigmoid(magnitude * theta)
                close(module.sampling_accuracy(theta, b=b, L=magnitude), accuracy)
                for rule in ("R", "S"):
                    gradient, z, errors, p = enumerate_outcomes(theta, a, b, magnitude, rule)
                    actual = module.stopped_gradient(theta, rule, a=a, b=b, L=magnitude)
                    maximum_gradient_residual = max(maximum_gradient_residual,
                                                    abs(actual - gradient))
                    close(actual, gradient)
                    close(module.acceptance_rate(theta, rule, a=a, b=b, L=magnitude), z)
                    close(errors, b * (1 - p))
                    close(z, a * p + b * (1 - p))
            average_magnitude = 1 - b + b * magnitude
            close(module.stopped_gradient(0, "R", a=a, b=b, L=magnitude),
                  (b - a) * average_magnitude / (2 * (a + b)))
            close(module.stopped_gradient(0, "S", a=a, b=b, L=magnitude),
                  (b * magnitude - a * average_magnitude) / (2 * (a + b)))

        # Sufficiency on a finite grid is a consistency check, not universal proof.
        for q in (1 - a, (2 - a) / 2, 1.0):
            for magnitude in (2.0, 10.0, 25.0):
                for theta in (-2.0, -0.5, 0.0, 0.3, 1.0):
                    audited, _, _, _ = enumerate_outcomes(theta, a, b, magnitude, "S", q)
                    assert audited < 0, (a, b, magnitude, theta, q, audited)
                    checked += 1
        q = (1 - a) / 2
        magnitude = 2 * a * (1 - b) / (b * (1 - a - q))
        audited, _, _, _ = enumerate_outcomes(0, a, b, magnitude, "S", q)
        expected = (b * magnitude * (1 - a - q) - a * (1 - b)) / (2 * (a + b * (1 - q)))
        close(audited, expected)
        assert audited > 0
        checked += 1
        preaudit_high = b * (a + 1) / 2
        close(q * preaudit_high / ((a + b) / 2), q * b * (a + 1) / (a + b))

        # Equality at the initial contrast threshold gives a zero vector field.
        # Global constancy of the resulting flow uses uniqueness in the proof.
        critical_magnitude = a * (1 - b) / (b * (1 - a - q))
        critical_gradient, _, _, _ = enumerate_outcomes(
            0, a, b, critical_magnitude, "S", q)
        close(critical_gradient, 0)

        # Explicit negative-tail and conditional-average bounds from Appendix A.
        # Choose a strict margin in the exponential tail condition.
        for magnitude in (2.0, 4.0, 10.0, 25.0):
            tail_theta = min(-1.0, (math.log(a * (1 - b) / (4 * b * magnitude))
                                   - 1.0) / (magnitude - 1))
            upper_numerator = (-a * (1 - b) * math.exp(tail_theta) / 4
                               + b * magnitude * math.exp(magnitude * tail_theta))
            assert upper_numerator < 0
            checked += 1
            for audit in (0.0, (1 - a) / 2, 1 - a, 1.0):
                gradient, z, _, _ = enumerate_outcomes(
                    tail_theta, a, b, magnitude, "S", audit)
                numerator = gradient * z
                assert numerator <= upper_numerator + 2e-12
                assert gradient < 0
                assert abs(gradient) <= magnitude + 2e-12
                checked += 3
            for theta in (-3.0, 0.0, 0.3):
                for rule in ("R", "S"):
                    for audit in ((0.0,) if rule == "R" else (0.0, 1 - a, 1.0)):
                        gradient, _, _, _ = enumerate_outcomes(
                            theta, a, b, magnitude, rule, audit)
                        assert abs(gradient) <= magnitude + 2e-12
                        checked += 1

    # Bisection is a numerical root check. Uniqueness is neither used nor tested.
    root = module.find_balance("S", -0.2, 0, a=0.75, b=0.5, L=10)
    close(root, -0.11146044018345)
    close(module.sampling_accuracy(root), 0.35958859256212)
    for theta in np.linspace(root + 1e-7, 0, 50):
        assert module.stopped_gradient(theta, "S") > 0
        checked += 1

    # Finite matched subsets: identical correct gradients cancel before clipping.
    rng = np.random.default_rng(2027)

    # Conditional acceptance probabilities suffice to evaluate the covariance
    # identity by total expectation. No sampled masks or second moments enter.
    p0, tpr, fpr = 0.35, 0.7, 0.2
    gradients_c = rng.normal(size=(3, 7))
    gradients_e = rng.normal(size=(5, 7))
    rates_rc = np.full(3, tpr)
    rates_re = np.full(5, fpr)
    rates_sc = rates_rc + np.array([0.2, -0.3, 0.1])
    rates_se = rates_re + np.array([-0.1, 0.1, 0.0, 0.05, -0.05])
    close(rates_sc.mean(), tpr)
    close(rates_se.mean(), fpr)
    z = p0 * tpr + (1 - p0) * fpr
    gr_cov = (p0 * (rates_rc[:, None] * gradients_c).mean(axis=0)
              + (1 - p0) * (rates_re[:, None] * gradients_e).mean(axis=0)) / z
    gs_cov = (p0 * (rates_sc[:, None] * gradients_c).mean(axis=0)
              + (1 - p0) * (rates_se[:, None] * gradients_e).mean(axis=0)) / z

    def conditional_covariance(rate_difference, gradients):
        return ((rate_difference[:, None] * gradients).mean(axis=0)
                - rate_difference.mean() * gradients.mean(axis=0))

    covariance_gap = (p0 * conditional_covariance(rates_sc - rates_rc, gradients_c)
                      + (1 - p0) * conditional_covariance(
                          rates_se - rates_re, gradients_e)) / z
    close(gs_cov - gr_cov, covariance_gap)

    c, e, dimension = 12, 4, 7
    correct = rng.normal(size=(c, dimension))
    errors_r = rng.normal(size=(e, dimension))
    errors_s = rng.normal(size=(e, dimension))
    k = c + e
    gr = (correct.sum(axis=0) + errors_r.sum(axis=0)) / k
    gs = (correct.sum(axis=0) + errors_s.sum(axis=0)) / k
    close(gs - gr, (errors_s - errors_r).sum(axis=0) / k)
    bound = np.linalg.norm(errors_s - errors_r, axis=1).sum() / k
    assert np.linalg.norm(gs - gr) <= bound + 1e-12
    checked += 1

    # Smooth-loss remainder for arbitrary displacements, not assumed SGD updates.
    matrix = rng.normal(size=(dimension, dimension))
    hessian = (matrix + matrix.T) / 2
    smoothness = np.linalg.norm(hessian, ord=2)
    point = rng.normal(size=dimension)
    linear = rng.normal(size=dimension)
    h = hessian @ point + linear

    def risk(x):
        return 0.5 * x @ hessian @ x + linear @ x

    displacements = [rng.normal(size=dimension) * 0.01 for _ in range(2)]
    losses = [risk(point + d) - risk(point) for d in displacements]
    projections = [h @ d for d in displacements]
    bounds = [smoothness * (d @ d) / 2 for d in displacements]
    for loss, projection, remainder in zip(losses, projections, bounds):
        assert abs(loss - projection) <= remainder + 1e-12
        if abs(projection) > remainder:
            assert np.sign(loss) == np.sign(projection)
        checked += 1
    assert abs((losses[1] - losses[0]) - (projections[1] - projections[0])) <= sum(bounds) + 1e-12
    checked += 1

    print(json.dumps({"status": "passed", "checks": checked,
                      "max_gradient_residual": maximum_gradient_residual,
                      "theta_star_L10": root,
                      "sampling_accuracy_at_root": module.sampling_accuracy(root),
                      "source": str(source),
                      "scope": "deterministic numerical consistency; not GPU experiments or proofs"},
                     indent=2))


if __name__ == "__main__":
    main()
