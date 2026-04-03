# Hints

- Before Iter 1, run `ncu` on the baseline kernel to guide the first direction.
- If 3 consecutive iterations show no improvement, run `ncu` to re-profile, use WebSearch for new ideas, and review `ITERATIONS.md` for patterns. Plan before continuing.

## Agent Behavior Controls

- **NEVER** use `git reset`, `git rebase`, or `git commit --amend`. The ONLY rollback mechanism is `git revert`. This is verified automatically by `kernelhub sync-git` — violations cause the entire run to be rejected.
- Before each commit, mentally verify: "Does this commit ADD to the linear chain? Or does it REPLACE/DELETE existing commits?" Only the former is acceptable.
