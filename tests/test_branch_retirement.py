"""Unit-check exact branch retirement safety rules; never contact or mutate GitHub."""
from __future__ import annotations

from scripts.retire_merged_branches import EXPECTED_BRANCHES, eligible


def test_only_audited_original_heads_are_eligible():
    assert len(EXPECTED_BRANCHES)==26
    for name,sha in EXPECTED_BRANCHES.items():
        allowed,why=eligible(branch=name,current_sha=sha,protected=False,ahead_by=0)
        assert allowed,(name,why)


def test_branch_retirement_fails_closed_on_changed_head_or_new_work():
    name,sha=next(iter(EXPECTED_BRANCHES.items()))
    assert not eligible(branch=name,current_sha="0"*40,protected=False,ahead_by=0)[0]
    assert not eligible(branch=name,current_sha=sha,protected=False,ahead_by=1)[0]
    assert not eligible(branch=name,current_sha=sha,protected=True,ahead_by=0)[0]
    assert not eligible(branch="main",current_sha=sha,protected=False,ahead_by=0)[0]
    assert not eligible(branch="feature/unknown",current_sha=sha,protected=False,ahead_by=0)[0]


def test_cleanup_never_targets_main():
    assert "main" not in EXPECTED_BRANCHES
    assert all(name.startswith(("phase2/","phase3/","codex/","fix/"))
               for name in EXPECTED_BRANCHES)
