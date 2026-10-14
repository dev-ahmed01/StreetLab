# StreetLab: October 14, 2026 branch-retirement decision

## Why 27 branches remained after 20 merged PRs

The repository contained 27 refs: the current `main` and 26 historical development refs. All 20 GitHub PRs were merged, but GitHub had retained their branch refs. Several earlier PRs were squash-merges; their original source-tip commits therefore did not appear as ancestors of `main` even though the PR changes were incorporated. Three older experimental branch names also pointed to two distinct unmerged commit tips.

## Preserve history without overwriting current code

Commit `8e60f6e51463db7fc44fb8734764844b5ce7cd3e` is an **ours-tree, history-only, multi-parent merge**. It preserves the original 9 Phase 2/3 PR source heads and two distinct experimental heads in the ancestry of `main`, while retaining the exact application tree from `8bfad82bc3d58f81e35a5c89fa1584d1124ccd12`. It did not replace modern `decision_lab.py` with an obsolete experiment, nor reintroduce broken `BranchPolicy` / `DecisionKind` imports.

The post-merge GitHub comparison confirmed **ahead_by=0 on all 26 non-main refs**. These source commits can therefore be reached via `main` even when their original branch names are removed.

## Safe deletion controls

`scripts/retire_merged_branches.py` freezes the 26 exact branch names and last-audited Git commit SHAs. It never deletes `main`, protected refs, unknown refs, changed ref heads or refs containing any commits outside the latest default branch. It re-fetches the target SHA immediately before requesting deletion. If the remote changed or GitHub rejected deletion, the action reports failure rather than claiming success.

The dedicated GitHub Actions workflow `retire-merged-branches.yml` performs clean-source and local HTTP regression checks before using a scoped `contents: write` token to delete only those safe old refs. GitHub's deletion endpoint does not offer an atomic SHA lease, so the script uses a final pre-delete ref read; do not let other contributors push to these legacy refs while this workflow runs.

Branch deletion does **not** delete old commit history, rewrite historical Git data, remove binary blobs from historical commits, or imply successful frozen W04 inference or independent real-junction measurements. Repository history may remain large until a separately coordinated and authorized Git history compaction.

## Post-retirement target

Only `main` should remain as an active GitHub branch. Begin future changes from `main` in a new focused branch, add CI and PR review, merge reviewed changes and remove each completed branch. Enable repository rulesets protecting `main` and requiring passing CI checks before further production work.

**This is code/history housekeeping, not physical scientific sign-off.**
