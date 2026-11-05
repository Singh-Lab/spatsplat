# BARseq PASTE2 and MLP-MAE rigid controls

Entry point: `notebooks/registration_controls-paste2_mlp_barseq.ipynb`.
This is a separate exploratory extension. The accepted publication notebook and
figure are unchanged. It does not require another notebook to be executed.

## Run order

1. Use the registration environment and existing BARseq download.
2. Add `paste2==1.0.1`. In the already configured environment:
   `python -m pip install --no-deps paste2==1.0.1`.
   A fresh environment can use `requirements-registration-controls.txt`.
3. Run with `STAGE='development'` first: section 19, seven cases, ten methods.
4. Inspect failures, estimated overlap, field reconstruction, and parameter counts.
   Resolve numerical issues and freeze settings before follow-up evaluation.
5. Set `STAGE='evaluation'` for sections 15, 23, 27: 21 cases, ten methods.

The original eight methods load their existing caches when compatible. Missing
fits are allowed by the RUN flags. The two added methods have separate caches;
turning their RUN flags off requires completed caches. No oracle overlap is
provided to PASTE2. Data loading, metrics, and plots are all notebook cells.

## Design

PASTE2 rigid tests a partial-overlap registration algorithm. Its overlap is
estimated by its official model-selection routine using training data only.
GLM-PCA costs are also training-only. The final partial coupling is converted to
a proper rigid transformation and applied to every evaluation coordinate. This
forbids reflections and evaluates held-out observations independently of the
transported subset. It is not a comparison restricted to PASTE2's selected matches.

MLP-MAE rigid uses the library's `CountMLP`, trained independently per view with
count MAE, hidden width 128, 300 epochs and at least 2000 updates. Each frozen
field is registered with the same symmetric cosine objective, 12 coarse rotation
starts, top-three refinement, 300 refinement steps, and checkpoint rule as
Splat rigid. Its rigid settings are copied directly from `RIGID_SETTINGS`.
Post-fit reconstruction and parameter counts are saved for both field types.
Their reconstruction losses differ; this is not a pure representation ablation.

The sample cap, gene panel, train/test membership, unequal-coverage construction,
known perturbation, and evaluation code match the standardized BARseq benchmark.
Test expression, true transforms, and overlap masks never enter fitting or overlap
selection. Development output must not be pooled with fixed follow-up results.

## PASTE2 implementation qualifications

The adapter pins the inspected PyPI release, `paste2==1.0.1`. The official
repository now points users to PASTE3; this experiment explicitly retains the
PASTE2 release to avoid silently changing the existing PASTE/POT environment.

Local function wrappers fix the obsolete sparse-array check and compute edge
disagreement without pandas chained assignment. They initialize the requested
partial mass rather than a unit-mass coupling. The remaining native optimizer
and official overlap-selection rule are retained; installed globals/files are
not patched. These qualifications must accompany any published PASTE2 comparison.

The adapter audits partial marginals, mass, objective descent, independently
recomputed partial-FGW cost, convergence, and rigid-map replay. Selection and
final alignment retain their respective native spatial normalization. The
training-derived GLM-PCA costs are reused. Selection attempts 20 overlap values,
so runtime can be substantial. Candidate couplings and audits are checkpointed
individually. Invalid or nonconverged candidates fail visibly, without substituting
true overlap, discarding difficult evaluation cases, or silently using another map.

Source references:
- [PASTE2 repository](https://github.com/raphael-group/paste2)
- [Overlap selection](https://github.com/raphael-group/paste2/blob/main/src/paste2/model_selection.py)
- [Partial-alignment implementation](https://github.com/raphael-group/paste2/blob/main/src/paste2/PASTE2.py)

## Files and outputs

Additional source files to commit alongside the existing publication dependency list:

```text
notebooks/registration_controls-paste2_mlp_barseq.ipynb
paper/workflows/barseq_additional_controls.py
requirements-registration-controls.txt
tests/test_barseq_additional_controls.py
docs/barseq_additional_controls.md
```

New caches/reports live in `notebooks/checkpoints/barseq_additional_controls/`.
Figures go to `figures/barseq_additional_controls/`. Reports include the protocol,
case inventory, method availability, completion status, score tables, additional
fit status with selected overlap, and field diagnostics. The main plot includes
all ten methods and does not hide failed/missing models: incomplete results block it.

Only syntax/import checks and small synthetic tests were run during preparation.
The PASTE2 solver smoke test used the downloaded wheel without installing it into
the user's environment. Full BARseq overlap selection, field fitting, and notebook
figures were not executed. Dataset-scale convergence and runtime remain to be checked.
# Numerical Stability Retries

The PASTE2 inner EMD solver uses a bounded iteration-limit retry: one million,
five million, then twenty million iterations. Only the explicit `numItermax`
warning triggers retry; other solver failures remain errors. Returned couplings
must satisfy balanced marginals before returning to the partial-transport solver.
This changes solver effort, not the objective or overlap-selection criterion.
The installed PASTE2/POT packages are not modified. Attempt budgets and warnings
are included in the candidate/final solver audits.

## Runtime Tracking

The fitting cell now displays runtime tables and saves `timing_events.jsonl` and
`timing_summary.csv` under the report directory. Timings use a monotonic wall
clock, synchronize CUDA, record hardware, and exclude evaluation/plotting.
PASTE2 includes cost preprocessing, numerical retries, overlap selection, and
alignment. MLP field fitting and rigid pose optimization are separate stages.
The Splat and CAST family timers include shared preparation and both affine
variants; the Splat rigid pose timer excludes reusable field fitting. These are
workload timings, not interchangeable per-method end-to-end speed rankings.
Completed fresh work, resumed work, failed attempts, and cache loading are
distinguished. Only completed fresh work enters the fit-minute summary.
Existing caches have unknown historical fit time, not zero fit time. No caches
are invalidated or deleted to obtain timing. New evaluation fits will acquire
timings automatically; reconstructing old full runtimes requires a fresh run.
Compare runs on the same machine with the same CPU/GPU settings. The append-only
event log retains failures and repeated attempts rather than hiding their cost.

PASTE2 GLM-PCA can diverge on the cropped BARseq training views. The adapter
tries ridge penalties 1, 10, 100, then 1000, using the same seed each time and
stopping at the first finite, informative cost matrix. Only numerical divergence
triggers a retry; no held-out metric or known overlap selects the penalty.
This is a disclosed stability modification, not the untouched default PASTE2.
Each new cost fit records its attempts in `glmpca_audit.json`. If all attempts
fail, the method remains unavailable and the complete comparison is blocked.
Exact caches from the preceding implementation are reused: previously successful
default-penalty PASTE2 fits and MLP fits do not need retraining.
