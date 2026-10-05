"""Smoke tests for the autonomous-loop plumbing.

Guards the docs/PLAN.md format that scripts/claim.py and scripts/mark.py depend
on, and the claim script's safety checks. Out of scope: research logic. Runs
without torch or outputs/, so it is safe on any machine and in CI.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REQUIRED_FIELDS = {"status", "owner", "claimed_at", "deps", "source", "done-when"}
# "cut" is a terminal status for a step retired by human decision (e.g. one
# dropped for scope): like "done" it is finished and never re-claimed, but it
# did not satisfy its done-when.
VALID_BASE_STATUSES = {"todo", "claimed", "done", "blocked", "cut"}


def _load_claim_module():
    spec = importlib.util.spec_from_file_location("claim", ROOT / "scripts" / "claim.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["claim"] = mod
    spec.loader.exec_module(mod)
    return mod


def test_plan_ledger_parses_and_is_consistent():
    claim = _load_claim_module()
    lines = (ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8").splitlines()
    blocks = claim.parse(lines)
    assert blocks, "docs/PLAN.md has no step blocks"

    ids = [b["id"] for b in blocks]
    assert len(ids) == len(set(ids)), "duplicate step ids in docs/PLAN.md"

    id_set = set(ids)
    for b in blocks:
        missing = REQUIRED_FIELDS - set(b["fields"])
        assert not missing, f"step {b['id']} missing fields: {missing}"
        status = claim.status_of(b)
        base = status.split(" ", 1)[0].split("(", 1)[0]
        assert base in VALID_BASE_STATUSES | {"in_review"}, f"step {b['id']} has odd status {status!r}"
        for dep in claim.deps_of(b):
            assert dep in id_set, f"step {b['id']} depends on unknown step {dep!r}"


def test_plan_has_a_frontier_or_is_human_gated():
    """The ledger must never truly deadlock. A healthy frontier is: an eligible
    todo, something in flight (claimed/in_review), or a blocked step a human can
    unblock. A *real* deadlock is only when unfinished todo steps remain but
    none is eligible, nothing is in flight, and nothing is blocked — i.e. work
    is stuck with no human-resolvable path. A legitimate pause waiting on human
    input (everything either done or blocked) is not a deadlock."""
    claim = _load_claim_module()
    lines = (ROOT / "docs" / "PLAN.md").read_text(encoding="utf-8").splitlines()
    blocks = claim.parse(lines)
    statuses = [claim.status_of(b) for b in blocks]
    if all(s in {"done", "cut"} for s in statuses):  # every step finished or retired
        return
    in_flight = any(s.startswith(("claimed", "in_review")) for s in statuses)
    human_gated = any(s.startswith("blocked") for s in statuses)
    assert claim.pick(blocks) is not None or in_flight or human_gated, (
        "docs/PLAN.md is deadlocked: unfinished todo steps but none eligible, "
        "nothing in flight, and nothing blocked for a human to unblock"
    )


def test_claim_rejects_unknown_args_before_touching_git(monkeypatch):
    # `claim.py --help` used to perform a real claim and a hard reset.
    claim = _load_claim_module()

    def no_git(*a, **k):
        raise AssertionError("git must not be called")

    monkeypatch.setattr(claim, "git", no_git)
    monkeypatch.setattr(sys, "argv", ["claim.py", "--help"])
    assert claim.main() == 2


def test_claim_refuses_with_dirty_tracked_files(monkeypatch):
    claim = _load_claim_module()
    calls = []

    class R:
        stdout = " M docs/PLAN.md\n"

    def fake_git(*a, **k):
        calls.append(a)
        if a[0] == "status":
            return R()
        raise AssertionError(f"reset/checkout must not run: {a}")

    monkeypatch.setattr(claim, "git", fake_git)
    monkeypatch.setattr(sys, "argv", ["claim.py"])
    assert claim.main() == 2
    assert calls == [("status", "--porcelain", "--untracked-files=no")]
