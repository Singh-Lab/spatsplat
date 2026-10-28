# Splat registration: methods, experiments, and metrics

This document describes the current two-dimensional standardized benchmark in
`notebooks/registration_standardized-allen_barseq.ipynb`. It does not describe
the separate joint volumetric registration experiments.

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

The main figure has three rows and three columns:

| Row | Before / after panels | Quantitative panel |
|---|---|---|
| Real Allen sections | A/B | C: CCF division agreement |
| Allen combined distortion | D/E | F: positional accuracy |
| BARseq unequal coverage | G/H | I: positional accuracy versus shared coverage |

Reference measurements are filled circles; query measurements are hollow.
The same configurable example method is shown in all three after panels, with
Splat rigid as the default. All eight methods appear in quantitative comparisons.
Examples are fixed, not chosen as each method's best result. The optional expanded
supplement adds the coverage-weighted cosine column and uses A-L lettering.

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

Per-run `protocol.json`, `scores.csv`, `method_availability.csv`, fit status,
and replay audits under `notebooks/checkpoints/registration_standardized/` provide
the recorded configuration and results. Figure exports go under root `figures/`.
