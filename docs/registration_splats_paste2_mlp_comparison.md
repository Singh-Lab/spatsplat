# Registration Comparison: Splats, PASTE2, and MLP-MAE

Updated: 2026-11-01.

## Scope and Provenance

This note summarizes **evaluation cases only**, not the development cases.

- Allen MERFISH: natural alignment of pairs 28/29 and 32/33, and controlled
  combined rotation plus smooth deformation of sections 28 and 32.
- BARseq: unequal-coverage experiments on sections 15, 23, and 27.
- Source: the exported plotting bundle
  `notebooks/checkpoints/registration_publication_controls/5b2016112ab99603ff0c/reports/plot_data/20261101T214518078609Z/`.
- Metrics: `scores.csv.gz`; runtime availability: `timings.csv.gz`.

The bundle contains the complete ten-method comparison. This note focuses on
Splat rigid, Splat affine, Splat non-affine, PASTE2 rigid, and MLP-MAE rigid.
Numbers below are descriptive means across evaluation cases, not significance
tests or confidence intervals. Crop directions within a section are not
independent biological replicates.

## Main Finding

**Splat rigid handles unequal coverage better on average than PASTE2 rigid and
MLP-MAE rigid in these BARseq evaluation cases.** Its advantage is modest at full
coverage and larger when the sampled regions differ. At the most severe coverage
mismatch, however, even splat registration remains inaccurate in absolute terms.

This supports a claim of relative robustness under the tested sampling mismatch,
not general superiority across registration tasks or datasets.

## BARseq: Unequal Coverage

Higher positional accuracy and coverage-weighted cosine are better.

| Shared parent tissue | Method | Mean positional accuracy | Mean coverage-weighted cosine |
|---|---|---:|---:|
| 100% | Splat rigid | **0.7477** | **0.5657** |
| 100% | PASTE2 rigid | 0.6941 | 0.5609 |
| 100% | MLP-MAE rigid | 0.1882 | 0.4240 |
| 60% | Splat rigid | **0.4711** | **0.5616** |
| 60% | PASTE2 rigid | 0.1582 | 0.4539 |
| 60% | MLP-MAE rigid | 0.1095 | 0.3433 |
| 20% | Splat rigid | **0.1299** | **0.2563** |
| 20% | PASTE2 rigid | 0.04408 | 0.008880 |
| 20% | MLP-MAE rigid | 0.03438 | 0.05425 |

Full coverage contributes three cases, one per section. Each partial-coverage
level contributes nine cases, comprising three crop directions per section.
The percentages describe the shared fraction of the parent tissue, not the
transported-mass parameter used internally by PASTE2. PASTE2 estimates that
parameter from training data rather than receiving the known overlap.

### Interpretation

- **100% overlap:** Splat rigid and PASTE2 are close, especially in expression
  agreement. MLP-MAE is substantially weaker.
- **60% overlap:** Splat rigid has a clearer advantage on both metrics. Its mean
  positional accuracy is about three times that of PASTE2, although accuracy is
  a nonlinear transformation of positional error and should not be described as
  a threefold reduction in error.
- **20% overlap:** Splat rigid retains more positional and expression agreement,
  but its mean per-case median positional error is still **11.06 reference-neighbor
  spacings**, versus **22.80** for PASTE2 and **33.19** for MLP-MAE. This is relative
  robustness, not reliable precise recovery in every case.

Coverage-weighted cosine accounts for spatial coverage as well as expression
agreement among matched neighborhoods. A low value can therefore reflect poor
spatial overlap, not merely dissimilar expression among the remaining matches.

### Additional Splat Variants

| Shared parent tissue | Method | Mean positional accuracy | Mean positional error, neighbor spacings | Mean coverage-weighted cosine |
|---|---|---:|---:|---:|
| 100% | Splat affine | 0.4022 | 3.140 | 0.5082 |
| 100% | Splat non-affine | 0.4711 | 1.975 | 0.5124 |
| 60% | Splat affine | 0.1896 | 6.071 | 0.4667 |
| 60% | Splat non-affine | 0.3041 | 3.267 | 0.5069 |
| 20% | Splat affine | 0.06583 | 16.42 | 0.1721 |
| 20% | Splat non-affine | 0.07301 | 15.29 | 0.1851 |

PASTE2 outperforms these more flexible splat variants at full BARseq coverage.
Splat rigid is the strongest splat variant on the reported BARseq averages;
additional transformation freedom does not automatically improve registration.

## Allen: Natural Alignment

| Method | Mean CCF division agreement | Mean coverage-weighted cosine |
|---|---:|---:|
| Splat rigid | 0.7193 | 0.5088 |
| Splat affine | 0.7216 | 0.5358 |
| Splat non-affine | 0.7225 | 0.5315 |
| PASTE2 rigid | 0.5184 | 0.4374 |
| MLP-MAE rigid | 0.6989 | 0.4847 |

The splat variants score better than the two added controls. However, the
unregistered CCF division agreement is already approximately **0.7189**.
Consequently, this task mainly demonstrates preservation of an already good
alignment, not a substantial improvement over the starting coordinates.

The PASTE2 mean includes both evaluation pairs after the numerical failures were
resolved. Earlier incomplete summaries could contain only the available pair
and should not be compared directly with this complete mean.

## Allen: Controlled Combined Distortion

Positional error is the mean of the per-case median errors, normalized by
reference-neighbor spacing. Lower is better.

| Method | Mean positional error, neighbor spacings | Mean positional accuracy | Mean coverage-weighted cosine |
|---|---:|---:|---:|
| Splat rigid | 3.369 | 0.2294 | 0.6203 |
| PASTE2 rigid | 5.118 | 0.1640 | 0.5693 |
| MLP-MAE rigid | 26.32 | 0.03661 | 0.3912 |
| Splat affine | **0.6390** | **0.6383** | **0.9128** |
| Splat non-affine | 0.8713 | 0.5513 | 0.8643 |

Among the rigid controls, Splat rigid performs better than PASTE2 and MLP-MAE.
MLP-MAE is approximately at the unregistered positional-error level of **26.33**
neighbor spacings.

Splat affine improves substantially, but has greater transformation flexibility
than PASTE2 rigid and MLP-MAE rigid. Its advantage cannot be attributed solely to
the field representation. The controlled distortion also contains a non-rigid
component that no rigid transform can fully undo.

## What the MLP Control Establishes

MLP-MAE and Splat rigid use the same expression-only rigid pose optimizer, but
their field-fitting objectives differ. MLP-MAE learns its fields using raw-count
MAE; the splat field-training objective is not identical.

The results show that **a continuous expression field alone is insufficient in
this implementation**: the MLP-MAE pipeline does not reproduce the splat pipeline's
registration performance. They do not isolate splat primitives as the sole cause,
nor establish that every MLP-based registration strategy would fail.

## Runtime

PASTE2 is substantially slower in the recorded runs. In the exported evaluation
bundle, its median fresh BARseq pipeline time is **3.948 minutes per case** across
21 cases. The corresponding MLP field and rigid-pose stage medians are
**0.03524 minutes** and **0.01755 minutes**, respectively. These are separate stage
medians, not an aggregated end-to-end median.

Fresh splat evaluation timings are **missing from this exported bundle**. Missing
times must not be treated as zero or replaced with cache-loading times. This bundle
therefore does not support a complete evaluation-time speedup estimate against
splats.

Further qualifications:

- PASTE2 includes expression preprocessing, numerical retries, overlap selection,
  and alignment.
- Splat family timing includes shared field preparation and both affine variants;
  Splat rigid pose is an additional stage that reuses those fields.
- PASTE2 runs on CPU, while MLP and splats use the configured GPU. This is a
  practical implementation comparison, not a hardware-matched algorithmic benchmark.
- Resumed fits are not counted as fresh end-to-end measurements.
- PASTE2 used documented numerical-stability and inner-solver iteration-limit
  retries. These change computational effort, not the evaluation targets or
  overlap-selection criterion.

## Suggested Paper Summary

> On BARseq unequal-coverage recovery, rigid splat-based registration achieved
> higher mean positional accuracy and coverage-weighted expression agreement than
> PASTE2 rigid and MLP-MAE rigid across the tested coverage levels. Differences
> were small at full coverage and larger under sampling mismatch, although severe
> mismatch remained challenging for all methods. On Allen MERFISH, splat-based
> registration preserved the already strong natural alignment and improved
> controlled distortion recovery relative to these rigid controls. These results
> support splats as a useful registration representation in the evaluated settings,
> without establishing universal superiority or isolating representation from
> field-training and transformation-model choices.

For experiment details and numerical modifications, see
`registration_publication_controls.md` and `barseq_additional_controls.md`.
