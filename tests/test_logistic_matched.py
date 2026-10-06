"""Regression tests for the matched-dynamics candidate modules.

Deliberately a separate file from `test_core.py`, whose text and semantics are
frozen and cover the legacy eight-round protocol.  Nothing here trains and
nothing here is evidence that the language-model experiment ran.
"""

from __future__ import annotations

import unittest
from fractions import Fraction

from rsi.logistic import (
    A_FIXED,
    B_FIXED,
    G_R_AT_ZERO,
    G_S_AT_ZERO,
    L_FIXED,
    PURITY_AT_ZERO,
    SAMPLING_ACCURACY_AT_STAR,
    THETA_STAR,
    THETA_ZERO,
    Z_AT_ZERO,
    acceptance_rate,
    acceptance_rates,
    contract_readouts,
    fixed_128_fixture,
    s_gradient_numerator,
    sigma,
    stopped_gradient,
)


class ContractTest(unittest.TestCase):
    """The frozen numbers, recomputed from the formulas rather than restated."""

    def test_frozen_readouts(self):
        self.assertAlmostEqual(stopped_gradient(THETA_ZERO, "R"), G_R_AT_ZERO, places=12)
        self.assertAlmostEqual(stopped_gradient(THETA_ZERO, "S"), G_S_AT_ZERO, places=12)
        self.assertAlmostEqual(acceptance_rate(THETA_ZERO, "R"), Z_AT_ZERO, places=12)
        self.assertAlmostEqual(acceptance_rate(THETA_ZERO, "S"), Z_AT_ZERO, places=12)

    def test_theta_star_and_sampling_accuracy(self):
        r = contract_readouts()
        self.assertAlmostEqual(r["theta_star"], THETA_STAR, places=9)
        self.assertAlmostEqual(r["sampling_accuracy_at_star"], SAMPLING_ACCURACY_AT_STAR, places=9)

    def test_l_one_is_degenerate_for_both_rules(self):
        r = contract_readouts()
        self.assertAlmostEqual(r["l_one_g_R"], -0.1, places=12)
        self.assertAlmostEqual(r["l_one_g_S"], -0.1, places=12)


class FixedFixtureTest(unittest.TestCase):
    """Check the retained finite pool, independently of population formulas."""

    def test_l_one_keeps_named_groups_distinct(self):
        points = fixed_128_fixture(L=1)["points"]
        self.assertEqual(len(points), 4)
        self.assertEqual({p["x"] for p in points}, {-1, 1})
        self.assertEqual(
            {p["point"]: p["S_keep_error"] for p in points},
            {"small_positive": 0, "small_negative": 0,
             "large_positive": 16, "large_negative": 16},
        )

    def test_counts_and_rates_match_for_degenerate_and_separated_groups(self):
        for magnitude in (1, 2, 3, 10):
            with self.subTest(L=magnitude):
                summary = fixed_128_fixture(L=magnitude)["summary"]
                self.assertEqual(summary["pool"], {"correct": 64, "error": 64})
                for rule in ("R", "S"):
                    self.assertEqual(summary[rule], {"correct": 48, "error": 32})
                    self.assertEqual(summary[rule + "_total"], 80)
                    self.assertEqual(summary[rule + "_purity"], 0.6)
                    self.assertEqual(summary[rule + "_TPR"], 0.75)
                    self.assertEqual(summary[rule + "_FPR"], 0.5)

    def test_retained_bce_gradients_match_initial_ablation_values(self):
        expected = {
            1: (Fraction(-1, 10), Fraction(-1, 10)),
            2: (Fraction(-3, 20), Fraction(-1, 20)),
            3: (Fraction(-1, 5), Fraction(0)),
            10: (Fraction(-11, 20), Fraction(7, 20)),
        }
        for magnitude, gradients in expected.items():
            fixture = fixed_128_fixture(theta=0, L=magnitude)
            for rule, gradient in zip(("R", "S"), gradients):
                with self.subTest(L=magnitude, rule=rule):
                    # At theta=0, d BCE / d theta = (1/2 - label) * x.
                    # Sum actual retained candidate derivatives, then normalize
                    # by their actual count; do not call the population formula
                    # to derive this independent finite-pool expectation.
                    total, derivative_sum = 0, Fraction(0)
                    for point in fixture["points"]:
                        x = point["x"]
                        truth = int(x > 0)
                        correct = point["keep_correct"]
                        error = point[rule + "_keep_error"]
                        derivative_sum += correct * (Fraction(1, 2) - truth) * x
                        derivative_sum += error * (Fraction(1, 2) - (1 - truth)) * x
                        total += correct + error
                    self.assertEqual(derivative_sum / total, gradient)
                    self.assertAlmostEqual(
                        float(gradient), stopped_gradient(0, rule, L=magnitude), places=12
                    )


class SgGradientNumeratorTest(unittest.TestCase):
    """`s_gradient_numerator` must equal `stopped_gradient * acceptance_rate`.

    The first release hardwired `2.5 q (.5 - p + .5 q) - .375 p (1 - p)`, the
    `L = 10`, `theta <= 0` branch.  That was exact at `theta <= 0` and wrong
    above it -- at the default `L`, not only at non-default ones.  These tests
    are the guard that the piecewise form cannot silently regress.
    """

    def _reference(self, theta, L):
        return stopped_gradient(theta, "S", L=L) * acceptance_rate(theta, "S", L=L)

    def test_matches_flow_across_theta_and_L(self):
        worst = 0.0
        worst_at = None
        for i in range(-400, 401):
            theta = i / 100.0
            for L in (1, 2, 4, 10, 25, 100):
                err = abs(s_gradient_numerator(theta, L=L) - self._reference(theta, L))
                if err > worst:
                    worst, worst_at = err, (theta, L)
        self.assertLess(worst, 1e-12, "worst deviation %r at %r" % (worst, worst_at))

    def test_positive_theta_is_the_case_the_old_form_broke(self):
        # At L = 10, theta = 0.31 the old hardwired form returned 0.869094
        # against a true 0.014833; at theta = 1.0 it returned 0.598537 against
        # -0.024528, a sign inversion on the frozen contract's own setting.
        for theta, expected in ((0.31, 0.01483320730723725), (1.0, -0.024528044020856947)):
            got = s_gradient_numerator(theta)
            self.assertAlmostEqual(got, expected, places=12)
            self.assertAlmostEqual(got, self._reference(theta, L_FIXED), places=12)

    def test_l_two_no_longer_flips_sign(self):
        # The value recorded in the hand-off checkpoint: 0.21875 was returned
        # where -0.03125 is correct.
        self.assertAlmostEqual(s_gradient_numerator(THETA_ZERO, L=2), -0.03125, places=12)

    def test_frozen_path_values_unchanged(self):
        self.assertAlmostEqual(s_gradient_numerator(THETA_ZERO), 0.21875, places=12)
        r = contract_readouts()
        self.assertAlmostEqual(r["s_numerator_at_zero"], 0.21875, places=12)
        self.assertAlmostEqual(r["s_numerator_at_minus_0p112"], -0.000601863087, places=11)

    def test_sign_is_positive_at_zero_and_negative_at_theta_star(self):
        # These are regression values, not a proof of global root uniqueness.
        self.assertGreater(s_gradient_numerator(THETA_ZERO), 0.0)
        self.assertLess(s_gradient_numerator(THETA_STAR), 0.0)


class AcceptanceRuleTest(unittest.TestCase):
    """Degenerate groups are defined, and S preserves the error budget."""

    def test_s_matches_r_error_mass_at_zero(self):
        r1, rL = acceptance_rates(THETA_ZERO, "R")
        s1, sL = acceptance_rates(THETA_ZERO, "S")
        # Both spend total error probability b per group mass; S just reweights.
        m1 = (1.0 - B_FIXED) * (1.0 - sigma(THETA_ZERO))
        mL = B_FIXED * (1.0 - sigma(L_FIXED * THETA_ZERO))
        self.assertAlmostEqual(s1 * m1 + sL * mL, B_FIXED * (m1 + mL), places=12)

    def test_rule_name_is_validated(self):
        with self.assertRaises(ValueError):
            acceptance_rates(THETA_ZERO, "X")


if __name__ == "__main__":
    unittest.main()
