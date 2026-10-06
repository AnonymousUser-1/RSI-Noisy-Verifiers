"""Regression tests for matching.py fixes (Issue B defects 1-2).

Tests the three fix scenarios from review:
A. K=16 correct-only pool should produce 16 correct + 0 error (not 20+4)
B. Error-only tasks are eligible for error role
C. Multiple correct-candidate lengths with error bucket at different length
"""
import pytest
from rsi.matching import Candidate, construct_matched_subsets


def test_fix_A_no_extra_correct_for_error_role():
    """Fix A: error role should not add correct candidates.

    Pool: 16 tasks, each with 1 correct + 2 errors (all tokens=8)
    Expected: K=16, C=12, E=4 (not K=20, C=16, E=4)
    """
    candidates = []
    for i in range(16):
        task_id = f"t{i:02d}"
        # Correct candidate
        candidates.append(Candidate(
            candidate_id=f"{task_id}-c",
            task_id=task_id,
            response=f"{task_id}-c",
            correct=True,
            error="",
            tokens=8
        ))
        # Target error
        candidates.append(Candidate(
            candidate_id=f"{task_id}-e-target",
            task_id=task_id,
            response=f"{task_id}-e-target",
            correct=False,
            error="nonshortest",
            tokens=8
        ))
        # Non-target error
        candidates.append(Candidate(
            candidate_id=f"{task_id}-e-other",
            task_id=task_id,
            response=f"{task_id}-e-other",
            correct=False,
            error="invalid",
            tokens=8
        ))

    result = construct_matched_subsets(candidates, k=16, seed=0)

    assert result["feasible"], f"Should be feasible: {result.get('certificate', {})}"

    audit = result["audit"]
    assert audit["K_R"] == 16, f"R should have 16 candidates, got {audit['K_R']}"
    assert audit["K_S"] == 16, f"S should have 16 candidates, got {audit['K_S']}"
    assert audit["C_R"] == 12, f"R should have 12 correct, got {audit['C_R']}"
    assert audit["C_S"] == 12, f"S should have 12 correct, got {audit['C_S']}"
    assert audit["E_R"] == 4, f"R should have 4 errors, got {audit['E_R']}"
    assert audit["E_S"] == 4, f"S should have 4 errors, got {audit['E_S']}"
    assert audit["ratio_C_to_E"], "C:E ratio should be 3:1"
    assert audit["precision"] == 0.75, f"Precision should be 0.75, got {audit['precision']}"


def test_fix_B_error_only_tasks_eligible():
    """Fix B: tasks with only errors (no correct) are eligible for error role.

    Pool: 12 correct-only + 4 error-only, K=16
    Expected: feasible, 12 correct + 4 error from error-only tasks
    """
    candidates = []

    # 12 correct-only tasks
    for i in range(12):
        task_id = f"C{i:02d}"
        candidates.append(Candidate(
            candidate_id=f"{task_id}-c",
            task_id=task_id,
            correct=True,
            error="",
            tokens=8,
            response=f"{task_id}-c"
        ))

    # 4 error-only tasks (each with target + other error, tokens=8)
    for i in range(4):
        task_id = f"E{i:02d}"
        candidates.append(Candidate(
            candidate_id=f"{task_id}-e-target",
            task_id=task_id,
            correct=False,
            error="nonshortest",
            tokens=8,
            response=f"{task_id}-e-target"
        ))
        candidates.append(Candidate(
            candidate_id=f"{task_id}-e-other",
            task_id=task_id,
            correct=False,
            error="invalid",
            tokens=8,
            response=f"{task_id}-e-other"
        ))

    result = construct_matched_subsets(candidates, k=16, seed=0)

    assert result["feasible"], f"Should be feasible: {result.get('certificate', {})}"

    audit = result["audit"]
    assert audit["K_R"] == 16
    assert audit["K_S"] == 16
    assert audit["C_R"] == 12
    assert audit["C_S"] == 12
    assert audit["E_R"] == 4
    assert audit["E_S"] == 4

    # Verify the 4 error tasks are used
    r_tasks = {cid.split('-')[0] for cid in result["R"]}
    s_tasks = {cid.split('-')[0] for cid in result["S"]}
    error_tasks = {f"E{i:02d}" for i in range(4)}
    assert error_tasks.issubset(r_tasks), "Error-only tasks should be in R"
    assert error_tasks.issubset(s_tasks), "Error-only tasks should be in S"


def test_fix_C_multiple_correct_buckets_error_at_different_length():
    """Fix C: correct bucket != error bucket should work.

    Pool: 12 correct-only (tokens=7 or 8) + 4 tasks with correct(tokens=7,8)
          and errors(tokens=8 only)
    Expected: error role uses the common error bucket (tokens=8),
              not KeyError on correct bucket 7
    """
    candidates = []

    # 12 correct-only tasks (tokens=8)
    for i in range(12):
        task_id = f"C{i:02d}"
        candidates.append(Candidate(
            candidate_id=f"{task_id}-c",
            task_id=task_id,
            correct=True,
            error="",
            tokens=8,
            response=f"{task_id}-c"
        ))

    # 4 tasks with multiple correct candidates (tokens=7,8) and errors (tokens=8)
    for i in range(4):
        task_id = f"B{i:02d}"
        # Correct candidate, tokens=7
        candidates.append(Candidate(
            candidate_id=f"{task_id}-a-c7",
            task_id=task_id,
            correct=True,
            error="",
            tokens=7,
            response=f"{task_id}-a-c7"
        ))
        # Correct candidate, tokens=8
        candidates.append(Candidate(
            candidate_id=f"{task_id}-z-c8",
            task_id=task_id,
            correct=True,
            error="",
            tokens=8,
            response=f"{task_id}-z-c8"
        ))
        # Target error, tokens=8
        candidates.append(Candidate(
            candidate_id=f"{task_id}-target8",
            task_id=task_id,
            correct=False,
            error="nonshortest",
            tokens=8,
            response=f"{task_id}-target8"
        ))
        # Other error, tokens=8
        candidates.append(Candidate(
            candidate_id=f"{task_id}-other8",
            task_id=task_id,
            correct=False,
            error="invalid",
            tokens=8,
            response=f"{task_id}-other8"
        ))

    result = construct_matched_subsets(candidates, k=16, seed=0)

    assert result["feasible"], f"Should be feasible: {result.get('certificate', {})}"

    audit = result["audit"]
    assert audit["K_R"] == 16
    assert audit["K_S"] == 16
    assert audit["C_R"] == 12
    assert audit["C_S"] == 12
    assert audit["E_R"] == 4
    assert audit["E_S"] == 4

    # All correct candidates should use tokens=8 (common signal bucket)
    from rsi.matching import _views
    pool_candidates = {c.candidate_id: c for c in candidates}
    for cid in result["R"]:
        c = pool_candidates[cid]
        assert c.tokens == 8, f"All selected candidates should have tokens=8, got {c.tokens} for {cid}"


def test_fix_C_negative_infeasible_case():
    """Fix C negative: truly infeasible case should still fail.

    Pool: errors at different lengths (target@8, other@9) with no common bucket.
    Expected: infeasible
    """
    candidates = []

    # 12 correct-only tasks (tokens=8)
    for i in range(12):
        task_id = f"C{i:02d}"
        candidates.append(Candidate(
            candidate_id=f"{task_id}-c",
            task_id=task_id,
            correct=True,
            error="",
            tokens=8,
            response=f"{task_id}-c"
        ))

    # 4 tasks with correct (tokens=8) and errors at different lengths
    for i in range(4):
        task_id = f"E{i:02d}"
        candidates.append(Candidate(
            candidate_id=f"{task_id}-c",
            task_id=task_id,
            correct=True,
            error="",
            tokens=8,
            response=f"{task_id}-c"
        ))
        # Target error at tokens=8
        candidates.append(Candidate(
            candidate_id=f"{task_id}-target8",
            task_id=task_id,
            correct=False,
            error="nonshortest",
            tokens=8,
            response=f"{task_id}-target8"
        ))
        # Other error at tokens=9 (different bucket)
        candidates.append(Candidate(
            candidate_id=f"{task_id}-other9",
            task_id=task_id,
            correct=False,
            error="invalid",
            tokens=9,
            response=f"{task_id}-other9"
        ))

    result = construct_matched_subsets(candidates, k=16, seed=0)

    # Should be infeasible because there's no common error bucket with both target and other
    assert not result["feasible"], "Should be infeasible (no shared signal bucket)"
    cert = result["certificate"]
    assert not cert["condition_error_tasks"], "Should fail error task condition"
