# Splat registration: methods, experiments, and metrics

Updated: 2026-10-31.

This document describes the two-dimensional standardized benchmark and final
publication figure in
`notebooks/registration_publication_reproducible-allen_barseq.ipynb`.
The self-contained notebook clones the experiment developed in
`notebooks/registration_standardized-allen_barseq.ipynb`, but does not execute or
read that notebook. It prepares its own inputs, fits missing models, evaluates
the results, and exports the figure. Existing compatible caches are optional.
The separate joint volumetric registration experiments are not included here.

For environment installation, data download commands, and the files to commit,
see [registration_reproduction.md](registration_reproduction.md).
For numerical outcomes, failures, and suggested manuscript wording, see
[registration_results_summary.md](registration_results_summary.md).

## 1. The central idea

For each section, fit a continuous gene-expression field made of spatial splats.
Then freeze the fields and optimize a coordinate transformation so that matching
locations have similar predicted expression profiles.

The pipeline is:

```text
Reference coordinates + expression -- fit reference splat field --+
                                                                 +-- optimize coordinate map
Query coordinates + expression ----- fit query splat field -------+
                                                                         |
                                                transform query measurements into reference frame
                                                                         |
                                                evaluate anatomy, positional recovery, and expression
```

Registration does not require one-to-one measurement identities between real
sections. It compares continuous fields evaluated at corresponding coordinates.
The splat fields are fitted independently, using a shared autoencoder
initialization within each pair. They are not jointly retrained during registration.

## 2. Building the expression fields

### Inputs and autoencoder

Each measurement has an x/y coordinate and a nonnegative expression vector over
the pair's shared gene panel. Coordinates are normalized into a common frame.
Expression profiles are formed by dividing each vector by its sum.

A shared profile autoencoder is trained on the two supplied sections. Its encoder
provides initial splat latents, and its decoder initializes each section's field
decoder. During field reconstruction, the section decoders can be fine-tuned;
the two fitted decoders are therefore not necessarily identical afterward.

The standardized settings use 512 splats **per section**, latent dimension 32,
hidden dimension 128, kernel sharpness 0.5, and 150 autoencoder epochs. Field
training uses a minimum budget of 1500 updates and a final MAE-refinement phase.
The notebook configuration and saved protocol are authoritative if these settings
are changed in a later run.

### Properties of a splat

Each splat has:

- A two-dimensional center.
- Two spatial scales and an in-plane rotation, defining an elliptical kernel.
- An opacity parameter affecting its spatial responsibility.
- A learned latent vector decoded into a gene-composition profile.
- A positive amplitude controlling its expression magnitude.

Let `w_k(x)` be the normalized spatial responsibility of splat `k`, `p_k` its
decoded gene-composition vector, and `a_k` its amplitude. Ignoring numerical
stabilizers, the predicted expression is:

```text
f(x) = sum_k w_k(x) * a_k * p_k
```

The kernel depends on center, spatial scales, rotation, opacity, and sharpness.
Expression is evaluated pointwise; this registration pipeline does not use a
nine-point measurement-neighborhood readout.

Initial latents use the encoded mean profiles of up to 16 nearby supplied fitting
measurements. This is initialization, not the later registration correspondence.

### Reconstruction objective

Field fitting combines composition cosine loss, log-total-expression loss,
off-tissue intensity regularization, and a center-support penalty. Expression MAE
is added during the final 25% refinement phase in the current configuration.
This is **not** an MAE-first, cosine-second schedule.

Once fitted, the reference and query fields are frozen. All three registration
variants reuse these exact fields. Registration optimizes coordinates, not a new
reconstruction against held-out expression.

## 3. The three splat registration variants

### Splat rigid

This fits one rotation and two translations. It cannot change size, shear, or
local shape, and it cannot reflect the tissue.

Write `T` for the query-to-reference transformation and `p_A(x)`, `p_B(y)` for
the normalized predicted profiles. The objective is symmetric:

```text
L_rigid = 0.5 * mean_y [1 - cosine(p_A(T(y)), p_B(y))]
        + 0.5 * mean_x [1 - cosine(p_B(T_inverse(x)), p_A(x))]
```

The implementation tests 12 rotation starts, initializes translation from the
measurement-cloud means, and refines the best three starts for 300 steps each.
It selects the lowest decoded-field objective checkpoint. The geometry-loss
weight is zero: geometry informs coordinates and initialization, but there is no
point-cloud matching term in the optimized rigid objective.

In this standardized extension, optimization and pose selection use supplied
fitting coordinates. Although the reused helper calls its selection input
`val`, it receives those fitting coordinates, not the reserved BARseq validation
or test observations. This is training-objective checkpoint selection, not an
independent validation estimate.

### Splat affine

This fits a 2-by-2 linear matrix and a two-dimensional translation. It allows
rotation, translation, scaling, and shear. Positive determinant is required for
selected checkpoints, excluding global reflections.

The implementation parameterizes a **reference-to-query** map `F`:

```text
F(x) = A x + t
L_field = mean_x [1 - cosine(p_A(x), p_B(F(x)))]
L_affine = L_field + lambda * [mean((A - I)^2) + mean(t^2)]
```

Expression comparison uses reference fitting coordinates whose transformed
positions remain in the normalized coordinate domain. Low valid fractions are
penalized. The regularizer favors identity; despite its historical
`rigid_reg_weight` name, it does not constrain the matrix to a rotation.

The map starts at identity. The current budget is 500 optimization steps, with
up to 2048 sampled reference coordinates. The lowest eligible training-objective
checkpoint is retained. Query measurements are displayed and evaluated through
the inverse map `F_inverse`.

### Splat non-affine

This starts from the selected affine map and adds a smooth displacement field:

```text
F_nonaffine(x) = A x + t + d(x)
```

The affine component is fixed during this refinement. `d(x)` is represented on
an 8-by-8 grid and evaluated by bilinear interpolation. The current refinement
budget is 400 steps.

Its objective combines:

- The same reference-to-query expression-profile matching term.
- Spatial smoothness of neighboring displacement-grid values.
- A penalty on displacement magnitude.
- Tissue-support and normalized-domain penalties.
- A splat-center anchor term.

Anchors are constructed after affine registration: active query splats are
matched to nearest active reference centers, then filtered by decoded-profile
cosine of at least 0.8. Their cosine values weight the center-distance penalty.
These are model-derived anchors, not anatomical labels or known correspondences.

Tissue support is estimated from fitting measurement coordinates, not histology.
The inverse is obtained iteratively, with contraction controls, inverse-roundtrip
checks, and an orientation guard used during checkpoint selection. These checks
are safeguards, not a proof that every map is anatomically correct or globally
free of local distortions.

### What is and is not controlled across variants

The three variants share field reconstructions and fitting inputs. However,
rigid uses multistart symmetric expression matching, whereas affine starts at
identity with a one-direction field objective, and non-affine adds support and
anchor terms. Therefore this is a comparison of practical registration variants,
**not a pure ablation of transformation degrees of freedom**.

Geometry enters through spatial field fitting, initialization, and, for
non-affine registration, explicit support and anchor penalties. It is inaccurate
to describe every splat variant as either geometry-only or expression-only.

## 4. Experiments

The complete protocol has 34 cases: six Allen cases (three section pairs, each
with a natural and a controlled case) and 28 BARseq cases (four sections, seven
coverage/direction cases each). Eight methods produce 272 case-method score rows
when all fits complete. Quantitative publication panels show the fixed evaluation
cases, not the Allen development pair or BARseq initial section 19.

Seed 1023 is the configured base seed. Each method uses the same settings across
the two datasets; there is no per-test-case selection of the best splat variant.
The development/initial examples informed exploratory work, so the full exercise
should not be characterized as a prospectively preregistered benchmark.

All tasks use the same eight-method roster:

| Method | Purpose |
|---|---|
| Unregistered | No additional alignment in the supplied frame |
| Geometry rigid ICP | Expression-blind point-cloud alignment |
| PASTE rigid | Expression/spatial transport-based rigid alignment |
| CAST affine | Learned-expression alignment with affine transformation |
| CAST non-affine | CAST with non-affine refinement |
| Splat rigid | Symmetric expression-field rigid registration |
| Splat affine | Affine expression-field registration |
| Splat non-affine | Regularized local refinement of splat affine |

Here the PASTE label describes this benchmark's rigid coordinate readout, not
the full range of analyses possible with PASTE. MLP and volumetric splat methods
are not part of this standardized roster. No method uses histology images or
CCF region labels for fitting.

### Baseline configuration and common evaluation

- Geometry rigid ICP uses the shared four-start implementation with 100 iterations.
- PASTE uses alpha 0.1, 400 outer iterations, and up to 1,000,000 inner iterations.
  The repository's corrected local solver is required; it is not an unmodified
  call to the installed PASTE optimization routine.
- CAST uses 400 representation-learning epochs, 1500 affine iterations, and 800
  non-affine iterations, with coordinate scale 1000 and reflections disabled.
  Its affine and non-affine outputs are both retained, not selected by test scores.
- CAST and PASTE run on CPU in this protocol. Splats use CUDA when available.

All methods receive the same fitting observations within a case. They do not
necessarily use the same internal representation or preprocessing. Saved native
transformations are applied to evaluation coordinates; held-out BARseq coordinates
are not mapped by interpolating training-point displacement vectors. Missing
methods are recorded explicitly rather than replaced with unregistered results.

### Experiment A: Real adjacent Allen MERFISH sections

- Development pair: query 25 to reference 24.
- Evaluation pairs: query 29 to reference 28; query 33 to reference 32.
- Neighboring sections are separated by 0.2 in the released z coordinate (mm).
- At most 3000 measurements per section are supplied.

Question: can expression-field registration improve agreement between genuine
adjacent sections?

There is no known exact pointwise transformation between different physical
sections. We therefore measure CCF annotation agreement and observed-expression
agreement, not ground-truth positional error.

These sections are already approximately aligned in the supplied coordinate
frame. A strong unregistered baseline and modest gains are expected limitations
of this task, not evidence that registration is generally unnecessary.

Allen fits all supplied observations. Its expression score is descriptive and
transductive, not held-out expression prediction. Region annotations are withheld
from fitting and used only for evaluation and visualization.

### Experiment B: Allen controlled rotation plus smooth deformation

Create a query copy of one section with the same expression and measurement
identities, then alter its coordinates by:

1. Rotating 20 degrees about the normalized center.
2. Translating by `(0.035, -0.025)`.
3. Adding `0.025 * sin(2*pi*y)` to x, using the transformed y coordinate.

Section 24 is the development control; sections 28 and 32 are evaluation
controls. The original coordinates provide known registration targets, withheld
from fitting. The displacement values above are normalized-coordinate units,
not micrometers.

Question: can the methods recover a known combined misalignment, including a
smooth local deformation?

This is an optimistic recovery control: expression vectors are identical across
the two copies. It is not equivalent to registering biologically different slices.
Rigid methods cannot exactly invert the local deformation by construction.

### Experiment C: BARseq unequal sampling coverage

Initial section: 19. Fixed follow-up sections: 15, 23, 27.

For each section, sample up to 8000 parent measurements and split them into two
disjoint acquisition pools. Assign train/validation/test subsets (60/20/20)
before cropping. Crop opposite ends of the two views along a chosen direction.
Apply a 35-degree rotation and translation `(0.06, -0.04)` to the query.

| Retained fraction per view | Nominal shared fraction of parent | Crop directions |
|---|---|---|
| 100% | 100% | One full-coverage case |
| 80% | 60% | 0, 60, 120 degrees |
| 60% | 20% | 0, 60, 120 degrees |

Opposite 80%-coverage views share approximately `0.8 + 0.8 - 1 = 0.6` of the
parent extent. These are nominal overlap fractions; realized measurement counts
can differ because sampling and tissue density are uneven.

Question: can expression help when partial sampling makes point-cloud shapes
misleading, even though the underlying coordinate transformation is rigid?

Only training observations are supplied to all fitters. The reserved validation
observations are unused by this standardized protocol. The known transformation,
shared-region mask, and held-out test expression are not supplied to fitting.
Scores use held-out observations within the true shared region. They do not
evaluate correctness of unmatched regions outside it.

There are seven cases per section. Follow-up box plots contain three cases at
full coverage and nine at each partial-coverage level. These are repeated cases
within three sections, not 21 independent biological replicates.

### Assay differences

Allen uses released nonnegative log2 expression values, whereas BARseq uses raw
counts. CAST receives the Allen values directly and CP10k/log1p-transformed
BARseq values. Each pair has its own shared gene panel. Consequently, do not pool
absolute expression scores across the two assays or call Allen values raw counts.

## 5. Metrics

### CCF division agreement: natural Allen sections

After registration, compute an annotation-blind spatial optimal-transport coupling
between reference and mapped query measurements. The coupling uses Euclidean
coordinate cost and uniform measurement marginals. It is a common evaluation
readout for every method, not another registration transformation and not PASTE's
native coupling.

CCF labels are grouped into broad anatomical divisions. Agreement is:

```text
transport mass linking equal annotated divisions
------------------------------------------------
transport mass linking annotated measurements on both sides
```

Higher is better. Unannotated matches are not counted as correct and are excluded
from the denominator. Inspect scored mass as well: conditional agreement alone
does not describe annotation coverage. Exact CCF-label agreement is also saved
and is stricter than broad division agreement. The saved `ccf_scored_mass` comes
from the exact-label scorer; division-level scored mass is calculated inside the
division scorer and should not automatically be assumed identical.

This is transport-weighted annotation agreement, not the percentage of visually
overlapping dots or tissue area. Different regional proportions between physical
sections can limit attainable agreement even after sensible alignment.

### Ground-truth positional error: controlled Allen and BARseq

Let `s` be the median nearest-neighbor distance among the evaluated reference
measurements. For each evaluated query measurement:

```text
normalized error_i = distance(mapped_query_i, true_reference_position_i) / s
```

Report the median and 90th-percentile error; lower is better. An error of 1 means
one typical reference-neighbor spacing, not one micrometer. For BARseq, `s` is
computed from the held-out shared-region reference subset and can vary by case.

The main figure uses the higher-is-better transformation:

```text
positional accuracy = 1 / (1 + median normalized error)
```

Examples: errors 0, 1, 3, and 9 become accuracies 1, 0.5, 0.25, and 0.1.
This is a bounded recovery score, **not a fraction of correctly registered
measurements**. Keep the raw errors available because this transformation
compresses differences among large failures. Average the per-case scores when
summarizing; transforming an average error is not mathematically equivalent.

### Observed-expression cosine: supplementary analysis

This metric compares observed profiles after coordinate registration, not the
splats' decoded profiles used in the fitting objective.

In each direction, transfer expression from up to five nearest source
measurements within radius `2*s`, weighting neighbors by inverse distance and
renormalizing the weights. Compare that transferred profile with the observed
target profile using cosine similarity. Repeat in the reverse direction, then
average the two directional scores equally.

- `profile_cosine`: mean cosine among supported targets in each direction.
- `coverage`: fraction with at least one source neighbor inside the radius,
  averaged over directions; directional fractions are also saved.
- `coverage_weighted_cosine`: assign zero to unsupported targets, average over
  all targets in each direction, then average directions.

Higher cosine is better, but supported-only cosine can hide unmatched tissue.
Coverage-weighted cosine penalizes such missing support; it is not generally
equal to the product of the two separately reported, direction-averaged metrics.
If a direction has no support, supported-only cosine is undefined, whereas its
coverage-weighted contribution is zero.

For the shuffle control, permute source profiles while keeping coordinates,
neighbors, weights, and support fixed. Repeat 99 times and report the null mean,
5th/95th percentiles, and observed-minus-null gain. This checks whether agreement
exceeds profile matching expected from the assay's background composition. It is
not a replacement for biological replication or an independent test set.

### Transformation diagnostics

The score tables also include local fold fraction, near-collapsed triangle
fraction, median absolute area ratio, and global-reflection diagnostics. These
are based on local coordinate geometry before and after mapping. Proper rigid
maps preserve area and distances; extreme area expansion or collapse can expose
failures that a single expression score misses.

Native-map replay checks verify that the saved transformation reproduces fitted
coordinates and can be applied directly to held-out coordinates. Non-affine
inverse-roundtrip checks assess numerical consistency. Neither check establishes
biological registration accuracy.

## 6. Publication figure and interpretation

### Final compact layout

The accepted main figure contains seven panels, A-G:

| Panel | Content |
|---|---|
| A | Natural Allen sections 29 to 28, side by side with inferred measurement correspondences |
| B | CCF division agreement for evaluation pairs 28/29 and 32/33 |
| C | Allen section 28 and its rotated/smoothly deformed copy, with inferred measurement correspondences |
| D | Positional accuracy for the controlled cases on sections 28 and 32 |
| E | BARseq section 23 unequal-coverage example before registration |
| F | The same BARseq example after registration |
| G | BARseq positional accuracy across shared fractions 100%, 60%, and 20% |

The BARseq example retains 80% per view, corresponding to nominal 60% shared
parent coverage, with crop direction 0 degrees. These example cases are fixed,
not chosen separately as each method's best outcome. `CORRESPONDENCE_EXAMPLE_METHOD`
defaults to Splat rigid and controls A, C, and F consistently. All eight methods
remain visible in B, D, and G.

### What the correspondence lines mean

Panels A and C connect **measurements**, not splat centers. Their construction is:

1. Apply the selected registration map to query coordinates.
2. Find mutual nearest neighbors between reference and mapped-query measurements.
3. Retain pairs no farther apart than twice the median reference-neighbor spacing.
4. If more than 60 pairs qualify, sample 60 uniformly without replacement using
   seed 1023. `MAX_MEASUREMENT_LINKS` controls this display cap.
5. Draw the two sections in their original input coordinates, shifting the query
   horizontally only for display. Connect the selected measurement identities.

Both sections use filled circles and the same CCF division color palette.
Unannotated measurements are grey. Matched endpoints have small black outlines;
all other measurements remain visible. Each panel reports shown and eligible
link counts. Neither CCF labels, expression similarity, nor known identity is
used to filter the displayed matches. There is no preference for high-scoring
regions or the smallest registration errors within the eligible set.

These lines illustrate **post-registration inferred correspondence**, not
verified biological matches, registration displacements, error vectors, or the
anchors used to train non-affine registration. The same inference rule is used
for the controlled copy even though its original measurement identities are
known. The horizontal display offset has no role in fitting or scoring.

Mutual matching and the distance cutoff intentionally omit unsupported matches;
the lines alone must not be treated as evidence of complete tissue recovery.
The quantitative panels use the full planned evaluation subsets, not the 60
displayed pairs. CCF agreement uses the spatial transport readout described
above, not these mutual-neighbor display links.

### BARseq panels and visual encodings

E and F are before/after overlays in a common reference frame, with filled grey
reference measurements and hollow teal query measurements. They use identical
padded coordinate limits and matched physical aspect ratios, preventing
independent zooming/recentering from changing the visual comparison. These
overlays show the supplied views; scoring remains restricted to held-out
measurements in the true shared region.

In G, colors identify methods. Section identities use both shapes and hatches:
section 15 is a diagonally hatched square, section 23 an oppositely hatched
triangle, and section 27 a crosshatched diamond. The legend uses the same symbols.
Dashed vertical separators distinguish the three shared-coverage groups. Boxes
summarize case-level scores, with individual cases shown; they are not confidence
intervals over independent animals. Shape meanings differ between the Allen and
BARseq quantitative panels and are explicitly separated in the legend.

Coverage-weighted cosine remains computed and saved, but is omitted from the
compact main figure. The optional expanded supplement restores the before/after
layout with cosine panels and A-L lettering. Set `MAKE_PUBLICATION_SUPPLEMENT=True`
to export it separately. Arial is requested, with Helvetica/DejaVu Sans fallback
when unavailable; the code alone does not guarantee that Arial is installed.

### Scope of the result

The intended claim is that continuous splat fields provide a useful medium for
expression-guided registration, especially when sampling differences weaken
geometry-only alignment. The experiments do not establish universal superiority,
general cross-animal robustness, or a benefit of three-dimensional splats.
Natural alignment, synthetic deformation recovery, and unequal-coverage recovery
answer complementary questions and should be reported separately. Failed cases
remain part of the results rather than being dropped or replaced by a different
splat variant after inspecting test performance.

## 7. Implementation references

- `paper/workflows/standardized_registration.py`: cases, common fitting runner,
  evaluation splits, and metric aggregation.
- `paper/workflows/standardized_rigid.py`: frozen-field restoration and rigid
  extension, reusing existing fit caches.
- `paper/barseq_registration_pilot.py`: symmetric multistart rigid objective.
- `paper/dlpfc_registration.py`: shared AE and independent field fitting.
- `paper/dlpfc_splat_engine.py`: field losses, affine/non-affine optimization.
- `paper/registration_perturbations.py`: model-derived splat anchors.
- `spatsplat/models.py`: amplitude splat field and pointwise reconstruction.
- `paper/dlpfc_registration_metrics.py`: common spatial transport readout.
- `paper/allen_ccf_diagnostics.py`: division labels and agreement calculation.
- `paper/allen_registration_cosine.py`: observed-expression and shuffle metrics.
- `paper/workflows/standardized_maps.py`: native transformation replay.
- `paper/workflows/standardized_publication.py`: main and supplementary figures.

## 8. Reproduction and recorded artifacts

The self-contained notebook uses the same fitting/evaluation helpers as the
development notebook. It needs the repository source, installed dependencies,
and downloaded data, but no other notebook execution or previously fitted model.

- `scripts/download_registration_allen.py` populates the Allen atlas cache with
  expression, section metadata, reconstructed CCF labels, and the CCF hierarchy.
- `scripts/download_barseq_pilot.py` downloads and checksum-verifies the fixed
  BARseq dataset and variable dictionary.
- `.env` supplies the user's absolute `SPATSPLAT_DATA_DIR`; optional
  `SPATSPLAT_ABC_ATLAS_DIR` overrides the default atlas cache location.
- `requirements-registration.txt` records the direct dependencies and pinned
  CAST/atlas-client revisions used for the reproduction setup.

Shared fit caches remain under `notebooks/checkpoints/registration_standardized/`.
The standalone notebook writes its reports under
`notebooks/checkpoints/registration_publication_reproducible/reports/<ID>/`
and figures under `figures/registration_publication_reproducible/<ID>/`.
It records `protocol.json`, `case_inventory.csv`, `scores.csv`,
`method_availability.csv`, `all_fit_status.csv`, `completion.json`, native-map
replay audits, and `data_environment_provenance.json`. The latter records package
versions, effective data paths, the selected atlas manifest when available, and
source fingerprints. The original development reports are not overwritten.

Exact reproducibility also depends on data release, package build, hardware, and
numerics, not just the random seed. The source-only clone/import checks and unit
tests passed when the notebook was prepared; a new-environment installation and
complete clean-data rerun were not executed as part of that packaging step.
See `docs/registration_publication_files.txt` for the explicit commit checklist.
