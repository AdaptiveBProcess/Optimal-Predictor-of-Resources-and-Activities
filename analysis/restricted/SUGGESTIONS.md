# Suggestions: outcome restrictions & masking

## Max-trace-length loop guard (not implemented)

**Problem.** Neither `NucleusMaskFunction` nor `TransitionRestrictionMaskFunction`
checks whether the allowed options can still reach END. A tight mask or a
restriction can trap a case in a loop:

- *Deterministic dead loop* (mask alone). AC_CRE:
  `Transferir creditos homologables -> itself 0.48, END 0.36, Transferir Creditos 0.16`.
  With `top_k=1, p_min_end=0.5` only the self-loop survives, so the case never ends and
  the episode never terminates (it ends only when `max_cases` cases complete).
- *Stochastic near-loop* (restriction). BPIC17 R1 `{"W_Call after offers": ["END"]}`:
  DRL-DRL repeats the call ~93% of the time, about 15 extra calls per case, and runs slow to a crawl.

**Proposal.** A per-case maximum trace length of **10x the longest trace in
the training log**. Once a case passes it, the activity mask allows only END. A counter
(`n_forced_end`) is printed per run and saved next to `restriction_counts.csv`, so
a hit shows up in the results instead of as a hung run.

**Where.** It could be a decorator mask (like `TransitionRestrictionMaskFunction`)
using `len(case.activity_history)`, which needs `trace_length` in
`ActivityMaskContext`. Or it could be a check in `BusinessProcessEnvironment.get_activity_mask`.
The decorator keeps the env untouched and matches the existing pattern.

**Sizing.** 10x is loose on purpose. It should never fire in a healthy run and
only catches real loops. Any non-zero `n_forced_end` should be reported as a
failure of the mask/rule combination, not as normal behaviour.
