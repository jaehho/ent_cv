#set document(
  title: "ENT Surgical Instrument Annotation Conventions",
  author: "Jaeho Cho",
)

#set page(
  paper: "us-letter",
  margin: (x: 1in, y: 1in),
  numbering: "1",
)

#set text(font: "New Computer Modern", size: 11pt, lang: "en")
#set par(justify: true, leading: 0.62em)
#set heading(numbering: "1.")
#show link: set text(fill: blue.darken(20%))
#show heading.where(level: 1): set text(size: 18pt, weight: "bold")
#show heading.where(level: 2): set text(size: 13pt, weight: "bold")
#show heading.where(level: 2): h => block(above: 1.5em, below: 0.6em, h)
#show heading.where(level: 3): set text(size: 11.5pt, weight: "bold")
#show figure.caption: set text(size: 9.5pt, fill: luma(60))

#align(center)[
  #text(size: 20pt, weight: "bold")[
    Annotation Conventions for ENT Surgical Instrument Detection
  ]
  #v(0.4em)
  #text(size: 12pt, fill: luma(60))[
    Two bounding-box conventions: tool-only (Convention A) and
    grasp-coupled (Convention B).
  ]
  #v(0.2em)
  #text(size: 10pt, fill: luma(80))[2026-05-09]
]

#v(0.6em)

#align(center)[
  #block(
    width: 90%,
    inset: 0.8em,
    radius: 4pt,
    fill: luma(245),
  )[
    #set text(size: 10pt)
    #set par(justify: false)
    *Summary.* The dataset was re-annotated under a stricter convention.
    On #strong[1240] frames present in both label sets, the median bounding-box
    area increased by a factor of #strong[2.34×] (median width 1.95×;
    median height 1.12×). The asymmetric expansion reflects lateral inclusion
    of the operator's gloved hand, formalized here as a hand-object interaction
    (HOI) annotation in the sense of Shan et al. @shan2020understanding.
  ]
]

= Definitions <sec:definitions>

We distinguish two annotation conventions used over the lifetime of this
dataset.

== Convention A: tool-only bounding boxes

For each visible surgical instrument in a frame, the annotation is the
smallest axis-aligned rectangle enclosing the instrument's external
surface. The operator's gloved hand is *excluded*, even when grasping
the instrument. This convention follows the prevailing approach in earlier
surgical-tool benchmarks such as m2cai16-tool @twinanda2017endonet and
the EndoVis Robotic Instrument Segmentation challenge @allan2019robust.

== Convention B: grasp-coupled instrument bounding boxes

For each visible surgical instrument, the annotation is the smallest
axis-aligned rectangle enclosing both the instrument *and* the visible
portion of the operator's gloved hand grasping it. When the hand is
fully occluded or out of frame, Convention B degenerates to Convention A.
This convention follows the hand-object interaction (HOI) formulation of
Shan et al. @shan2020understanding.

#align(center)[
  #block(
    width: 92%,
    inset: 0.7em,
    radius: 4pt,
    stroke: 0.5pt + luma(180),
  )[
    #set text(size: 10pt)
    #set par(justify: false)
    *Operational rule (single sentence).*
    Convention A: enclose the metal/plastic tool only. Convention B: enclose
    the tool plus the visible glove that touches it. If no hand is visible,
    A and B produce the same box.
  ]
]

= Empirical Comparison <sec:empirical>

The previous annotation set was preserved as a backup before the swap, so
both conventions exist for the same frames. The two label sets share
#strong[1240 frames] (those with non-empty labels in both); a further
172 frames are annotated only under Convention B (newly covered).

== Per-box geometry

Statistics are computed per bounding box across all 1240 paired frames
($n_"OLD" approx n_"NEW" approx 1245$ boxes).

#figure(
  table(
    columns: (auto, 1fr, 1fr, 1fr),
    align: (left, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 or y == 1 { 0.7pt } else { 0pt },
      bottom: if y == 4 { 0.7pt } else { 0pt },
    ),
    table.header(
      [*Metric (normalized)*],
      [*Convention A*], [*Convention B*], [*Ratio (B / A)*],
    ),
    [Box width (median)],   [0.087], [0.170], [*1.95×*],
    [Box height (median)],  [0.184], [0.207], [1.12×],
    [Box area (median)],    [0.0151], [0.0354], [*2.34×*],
    [Boxes per frame (median)], [1.0], [1.0], [—],
  ),
  caption: [Per-box geometry on the 1240-frame overlap. Coordinates are
    normalized to image dimensions.],
) <tab:geometry>

The change is dominated by *width*, not height. Heights grew only 12%;
widths nearly doubled. Geometrically, this is consistent with the gripping
hand being situated at the proximal end of the elongated instrument shaft:
the box widens to admit the hand, while the long axis of the tool is
unchanged.

== Class distribution

On the overlap, per-class instance counts are nearly identical between the
two conventions (e.g., Forceps: 397 vs. 392; Microdebrider: 217 vs. 217).
This was a *box-extent* re-annotation, not a re-classification.

= Schema Extension <sec:schema>

Convention B is accompanied by an extension of the class schema from 10 to
13 classes. The added classes are:

- *Empty Hand* — a gloved hand visible in the frame, holding no instrument.
- *Not Sure* — annotator-flagged uncertainty (soft label).
- *Patient* — patient anatomy region.

These three classes currently have *zero instances* on the analyzed
1240-frame overlap and zero instances on the 172 new-only frames. They
are declared in the schema but unused in the present annotations.

= Visual Examples <sec:examples>

Each pair below shows the same frame rendered twice: Convention A
(red, left panel) and Convention B (green, right panel). The reported
ratio is the sum of all box areas under B divided by the corresponding
sum under A on that frame.

== Largest growth (most dramatic convention change)

#figure(
  image("figures/01_largest_growth_a.png", width: 100%),
  caption: [Frame with $approx 9.81 times$ area increase.
    Convention A boxes the tool only; Convention B boxes tool plus the
    gloved hand grasping it.],
) <fig:growth_a>

#figure(
  image("figures/02_largest_growth_b.png", width: 100%),
  caption: [$approx 8.53 times$ area increase.],
) <fig:growth_b>

#figure(
  image("figures/03_largest_growth_c.png", width: 100%),
  caption: [$approx 6.89 times$ area increase.],
) <fig:growth_c>

== Median growth (representative case)

#figure(
  image("figures/04_median_growth_a.png", width: 100%),
  caption: [$approx 2.26 times$ area increase. Representative of the
    median behavior across the 1240-frame overlap.],
) <fig:median_a>

#figure(
  image("figures/05_median_growth_b.png", width: 100%),
  caption: [$approx 2.25 times$ area increase.],
) <fig:median_b>

== Smallest growth (degenerate case where conventions converge)

#figure(
  image("figures/06_smallest_growth.png", width: 100%),
  caption: [$approx 0.95 times$ — Convention B reduces to A when little or
    no hand is visible alongside the instrument. These cases empirically
    validate the operational rule in @sec:definitions.],
) <fig:smallest>

= Drop-in Methods Paragraph <sec:methods>

The following paragraph can be used (or adapted) as a methods-section
description in a downstream publication.

#block(
  fill: luma(245),
  inset: 0.9em,
  radius: 4pt,
)[
  Initial annotations followed a tool-only convention: each bounding box
  was the smallest axis-aligned rectangle enclosing the instrument's
  external surface, excluding the operator's gloved hand (analogous to
  m2cai16-tool @twinanda2017endonet). We subsequently re-annotated the
  dataset under a grasp-coupled instrument convention, in which each box
  additionally encloses the visible portion of the gloved hand grasping
  the tool, following the hand-object interaction (HOI) formulation of
  Shan et al. @shan2020understanding. On the $N = 1240$ frames re-annotated under both
  conventions, the median box area increased $2.34 times$ (median width
  $1.95 times$; median height $1.12 times$); the asymmetric expansion
  reflects lateral inclusion of the gloved hand at the proximal end of
  the elongated instrument shaft. The schema was concurrently extended
  from 10 to 13 classes, adding #emph[Empty Hand] (gloved hand without
  instrument), #emph[Not Sure] (annotator-flagged uncertainty), and
  #emph[Patient] (anatomical context).
]

= Motivation <sec:motivation>

Possible motivations for adopting Convention B over Convention A. A
downstream publication should select the one that actually drove the
decision, rather than listing all three.

+ *Active vs. passive disambiguation.* A hand attached to a box marks
  who is operating the instrument. Convention A cannot distinguish a
  surgeon's actively-used forceps from an idle instrument set on the
  drape.
+ *Activity / phase recognition.* Hand-coupled boxes give a richer cue
  for downstream surgical-action and phase classification, in line with
  the instrument-verb-target triplet formulation of CholecT50
  @nwoye2022rendezvous.
+ *Robustness to tool-tip occlusion.* When the distal tip of an
  instrument is occluded by tissue, a grasp-coupled box still grounds the
  detection on the visible glove.

= Training Configurations <sec:training>

A focused experiment was run on the Convention B (grasp-coupled) dataset
to identify the training recipe that produces the most accurate
instrument detector. Eight configurations were tested, each trained from
the same starting point (Ultralytics' publicly released
#strong[YOLO26-small] weights, pretrained on COCO) for 100 passes through
the training data and then evaluated on a held-out set of three surgical
cases (513 frames) that the model never saw during training. The
held-out cases are entirely separate operations, so a score on this set
reflects how well the detector will work on a future, unseen case.

Three image-handling choices are held fixed across the sweep so the
comparison is apples-to-apples: input images are processed at
1024×1024 (`imgsz=1024`), images within a batch are padded to a common
aspect ratio for efficiency (`rect=True`), and the random-rescaling
augmentation is disabled (`scale=0.0`). Variants G and H deliberately
break two of these to test resolution and multi-scale; they are noted
as exceptions in their descriptions.

== A short primer on what is being trained <sec:training-primer>

The object detector is a neural network with two functional parts.
A #emph[backbone] reads the input image and extracts general visual
features — textures, edges, shapes — at multiple levels of abstraction.
A #emph[head] uses those features to produce the actual output:
where each instrument is (a bounding box) and what kind of instrument
it is (a class label). Both parts are initialized from weights that
were pretrained on a large generic-image dataset, then fine-tuned
on the surgical data.

Performance is reported as #emph[mean Average Precision] (mAP), the
standard object-detection metric. mAP\@50 counts a predicted box as
correct if it overlaps the human-drawn box by at least 50%, so it
measures whether the model #emph[found] the object at all. mAP\@50–95
averages this score across overlap thresholds from 50% to 95%, so it
additionally rewards tightly-localized boxes; it is the stricter and
more discriminating metric. Both range from 0 (worst) to 1 (perfect).

== Configurations tested <sec:training-configs>

Each configuration is described as a deviation from the previous one.
A is the baseline. B applies two schedule changes on top of A. C–H
each apply one or two further changes on top of B.

#strong[A — baseline.] Ultralytics' default training hyperparameters
applied unchanged: the default learning-rate schedule, the default
mosaic augmentation timing, the default classification-vs-box loss
balance, the default weight decay, and so on. This is #emph[not] a
previously published model — it is a fresh 100-epoch training run
from the same pretrained starting weights as every other variant,
and serves as the reference number that other changes are measured
against.

#strong[B — schedule changes] (`cos_lr=True`, `close_mosaic=50`). Two
changes to #emph[how] the model is trained over time. No change to the
data or to the network itself.

- #emph[Cosine learning rate.] The "learning rate" controls how large
  a step the model takes when it updates its parameters from each
  batch of training images. It starts high (to make fast initial
  progress) and shrinks to nearly zero (to fine-tune at the end). The
  default schedule shrinks linearly, like a straight line down. The
  cosine schedule shrinks along a smooth bell curve, producing a more
  graceful slowdown and usually a more stable final model.
- #emph[Delayed mosaic close.] YOLO uses an augmentation called
  #emph[mosaic]: four different training images are stitched into a
  single 2×2 collage and presented to the network as one input. This
  teaches the model to handle multiple objects at varied scales in
  cluttered scenes. Near the end of training, however, the model
  needs to readjust to seeing realistic single-image inputs, so
  mosaic is turned off for the final epochs. The default turns it
  off for only the last 10 epochs; this configuration turns it off
  for the last 50, giving the model more time to refine on
  un-stitched images before training ends.

#strong[C — class imbalance and image-level regularization]
(B + `cls=0.8`, `mixup=0.15`). Two changes intended to help the model
handle class imbalance and small-dataset overfitting.

- #emph[Higher classification weight.] Training judges the model on
  two things simultaneously: drawing the box accurately, and naming
  what is inside the box. The `cls` parameter sets how much to
  penalize classification errors relative to box-drawing errors. The
  default of 0.5 keeps them balanced; raising it to 0.8 tells
  training to care more about getting the instrument type right.
  This is motivated by the heavily imbalanced class distribution —
  Forceps appears 27 times more often than Energy-based hemostasis
  instruments — so the model could otherwise learn to predict
  "Forceps" too readily.
- #emph[MixUp.] A regularization technique: with 15% probability, two
  training images are blended together (one semi-transparent on top
  of the other) and the model has to learn from the blended image.
  The aim is to force the model to be robust to ambiguous inputs.
  MixUp is well studied for image classification, but it interacts
  unpredictably with object detection because the bounding boxes
  from both source images both apply to the blended result.

#strong[D — heavier image augmentation]
(B + `degrees=10`, `perspective=5e-4`, stronger HSV). Three
image-distortion knobs turned up to expose the model to more visual
variation during training.

- #emph[Rotation up to ±10°] applied randomly each iteration.
- #emph[A subtle 3D-perspective warp] applied each iteration, as if
  the image were viewed slightly off-axis.
- #emph[Stronger random color shifts] in hue, saturation, and
  brightness than the defaults. The default values were tuned for
  natural-scene datasets such as COCO, not surgical lighting.

The hope is that the model generalizes better when it sees more
variation. The risk is that endoscopic recordings do not actually
rotate or tilt much in practice, so heavy augmentation may train the
model on physically implausible images.

#strong[E — freeze the backbone] (B + `freeze=10`). Recall that the
detector has a pretrained backbone (general visual feature extractor)
and a task-specific head. Freezing the first 10 layers means
#emph[only the head learns] from the surgical data; the backbone stays
exactly as it was pretrained. This is a standard trick for very small
datasets, on the theory that the pretrained backbone is already useful
and there isn't enough data to update its weights without overfitting.
The downside is that if the pretrained features are not well suited to
surgical imagery, the frozen backbone caps the achievable performance.

#strong[F — stronger weight decay] (B + `weight_decay=1e-3`). "Weight
decay" is a regularization mechanism: at every training step, every
parameter in the network is gently pulled toward zero. This prevents
any single parameter from growing very large and causing the model to
memorize specific training images. The default value (0.0005) is a
light pull; doubling it to 0.001 applies a stronger pull, with the
goal of reducing overfitting on the relatively small surgical dataset.

#strong[G — higher image resolution] (B + `imgsz=1280`). Train and
evaluate on 1280×1280 images instead of the default 1024×1024. More
pixels per image means more detail is preserved, which usually helps
the model detect small or fine-grained features. The cost is that
each training step processes more data, takes longer, and uses more
GPU memory — which automatically reduces the batch size (the number
of images processed simultaneously). A smaller batch can make
training slightly less stable.

#strong[H — multi-scale training]
(B + `multi_scale=0.5`, `rect=False`). The winning configuration, with
two coupled changes.

- #emph[Multi-scale resizing.] Instead of training every batch at the
  same fixed size, the input size is randomized each step within
  ±50% of 1024. A given training step might process the batch at
  512×512, 768×768, 1024×1024, 1280×1280, or 1536×1536. The model
  sees the same kinds of objects at many different scales, and so
  learns to recognize an instrument whether it fills the frame or
  is far in the background. Surgical recordings vary dramatically
  in zoom level; this knob directly targets that variability.
- #emph[Rectangular batching off.] By default, YOLO pads all images
  in a batch to a common aspect ratio for efficiency, which requires
  the batch to share fixed dimensions. Multi-scale training needs
  variable dimensions, so this padding optimization has to be turned
  off. This is a prerequisite for multi-scale, not an independent
  experimental change.

== Phase 1 results

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    stroke: (x, y) => (
      top: if y == 0 or y == 1 { 0.7pt } else { 0pt },
      bottom: if y == 8 { 0.7pt } else { 0pt },
    ),
    table.header(
      [*ID*], [*Configuration (changes vs. Ultralytics defaults)*],
      [*mAP\@50*], [*mAP\@50–95*],
    ),
    [A], [baseline],                                                  [0.810], [0.716],
    [B], [`cos_lr=True`, `close_mosaic=50`],                          [0.852], [0.755],
    [C], [B + `cls=0.8`, `mixup=0.15`],                               [0.790], [0.693],
    [D], [B + `degrees=10`, `perspective=5e-4`, stronger HSV],        [0.835], [0.672],
    [E], [B + `freeze=10` (freeze backbone)],                         [0.788], [0.704],
    [F], [B + `weight_decay=1e-3`],                                   [0.815], [0.724],
    [G], [B + `imgsz=1280`],                                          [0.790], [0.716],
    [H], [B + `multi_scale=0.5`, `rect=False`],                       [*0.892*], [*0.804*],
  ),
  caption: [Phase 1 hyperparameter sweep on yolo26s, 100 epochs each,
    evaluated on a held-out set of three surgical cases (513 frames,
    409 labeled). H is the winner: +8.8 percentage points
    mAP\@50–95 over baseline. All scores range 0 (worst) to 1 (perfect).],
) <tab:training>

=== Observations from Phase 1

- The single biggest lift comes from the #strong[schedule changes]
  alone (B): switching to a cosine learning-rate schedule and giving
  the model longer to refine on un-stitched images is worth
  #strong[+3.9 percentage points] of mAP\@50–95 over the baseline.
- #strong[Multi-scale training] (H) adds a further #strong[+4.9 pts]
  on top of B — the largest single-knob improvement in the sweep.
  Surgical scenes vary substantially in instrument distance and
  zoom, which is exactly what multi-scale resizing targets.
- The configurations aimed at class imbalance (C) and heavier
  spatial augmentation (D) #strong[regressed] on this dataset.
  MixUp likely blurs the instrument silhouettes the detector relies
  on, and aggressive rotation/HSV introduces visual conditions
  surgical scenes never actually exhibit. Notably, the regression
  is much larger on mAP\@50–95 than on mAP\@50, consistent with
  these augmentations loosening box localization at strict overlap
  thresholds.
- Backbone freezing (E) #strong[hurt] despite the small dataset size,
  suggesting that surgical imagery is sufficiently different from
  the natural-image features the backbone was pretrained on that
  full fine-tuning is required to adapt.
- Doubling the weight decay (F) and raising the image resolution to
  1280 (G) produced no measurable improvement. The default
  regularization and resolution appear near-optimal for this dataset
  size and image scale; spending more memory on a larger resolution
  did not buy more accuracy here.

== Phase 2: architecture sanity check

Phase 1 isolated a strong recipe (H) on the smallest YOLO26 size. Phase 2
re-runs that same recipe on two alternative architectures to verify the
choice of model: a larger member of the same family (#strong[yolo26-medium],
roughly $2.7 times$ the parameters of yolo26-small) and a different
architecture family of similar size (#strong[yolo11-small]). All three
runs share H's hyperparameters; only the underlying network changes.

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    stroke: (x, y) => (
      top: if y == 0 or y == 1 { 0.7pt } else { 0pt },
      bottom: if y == 3 { 0.7pt } else { 0pt },
    ),
    table.header(
      [*Model*], [*Notes*], [*mAP\@50*], [*mAP\@50–95*],
    ),
    [yolo26-small],     [Phase 1 winner (H), repeated here for reference], [0.892], [0.804],
    [yolo26-medium],    [#strong[Phase 2 winner], larger same-family model], [*0.892*], [*0.814*],
    [yolo11-small],     [different architecture family, similar parameter count], [0.883], [0.795],
  ),
  caption: [Phase 2: H configuration applied to three architectures,
    100 epochs each. yolo26-medium gives a small but consistent gain
    over yolo26-small on the strict mAP\@50–95 metric; yolo11-small
    underperforms slightly, confirming yolo26 is the better family
    for this task.],
) <tab:training-p2>

== Phase 3: final training

The Phase 2 winner (yolo26-medium with the H recipe) was retrained
from scratch for #strong[300 epochs] with early stopping (`patience=50`),
keeping all other H hyperparameters fixed. The longer schedule lets the
cosine learning-rate decay run its full course and gives the post-mosaic
fine-tuning phase additional time to converge.

#figure(
  table(
    columns: (auto, auto, auto, auto, auto),
    align: (left, right, right, right, right),
    stroke: (x, y) => (
      top: if y == 0 or y == 1 { 0.7pt } else { 0pt },
      bottom: if y == 2 { 0.7pt } else { 0pt },
    ),
    table.header(
      [*Run*], [*mAP\@50*], [*mAP\@50–95*], [*Precision*], [*Recall*],
    ),
    [Phase 3 final (yolo26-medium, 300 ep)], [*0.923*], [*0.833*], [0.903], [0.843],
  ),
  caption: [Phase 3 final detector — yolo26-medium with the H recipe
    trained for 300 epochs. Precision (boxes predicted that are
    correct) and Recall (instruments present that are found) reported
    at the standard 50% overlap threshold.],
) <tab:training-p3>

#strong[End-to-end gain] over the Phase 1 baseline (A): mAP\@50 rose from
0.810 to #strong[0.923] ($+11.3$ percentage points); mAP\@50–95 rose from
0.716 to #strong[0.833] ($+11.7$ percentage points, a $16%$ relative
improvement on the stricter metric). At the 50% overlap threshold,
roughly $90%$ of the model's instrument predictions are correct and
$84%$ of the actual instruments in the held-out cases are detected.

== External reference: Ultralytics factory defaults

The Phase 1 baseline (A) holds three image-handling settings at the
project's standard values (`imgsz=1024`, `rect=True`, `scale=0.0`) rather
than at Ultralytics' true factory defaults (`imgsz=640`, `rect=False`,
`scale=0.5`). As a sanity check, a separate run was performed with the
complete factory defaults — same starting weights, same 100 epochs, same
held-out validation set — to provide an external reference that any
future publication can cite.

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    stroke: (x, y) => (
      top: if y == 0 or y == 1 { 0.7pt } else { 0pt },
      bottom: if y == 3 { 0.7pt } else { 0pt },
    ),
    table.header(
      [*Run*], [*Settings*], [*mAP\@50*], [*mAP\@50–95*],
    ),
    [Phase 1 A], [`imgsz=1024`, `rect=True`, `scale=0.0` (project defaults)], [0.810], [0.716],
    [Factory], [`imgsz=640`, `rect=False`, `scale=0.5` (Ultralytics defaults)], [*0.874*], [*0.795*],
    [Phase 3 final], [winning recipe on yolo26-medium, 300 ep],                  [*0.923*], [*0.833*],
  ),
  caption: [Two reference points and the final detector. The Ultralytics
    factory-defaults run noticeably outperforms the Phase 1 baseline
    despite using #strong[smaller] input images (640 vs 1024), revealing
    that the project's previous choice to disable random-scale
    augmentation (`scale=0.0`) was costing roughly 7–8 percentage points
    of mAP\@50–95.],
) <tab:training-external>

The finding is itself informative. Ultralytics' factory `scale=0.5`
applies random rescaling to each training image (between $0.5×$ and
$1.5×$) before it is letterboxed to the input size. This produces
scale-variability similar to (though weaker than) the explicit
multi-scale training that variant H uses, and the Phase 1 baseline's
`scale=0.0` was silently removing that variability. The Phase 1 sweep
was therefore measuring relative deltas against an unintentionally
weak reference. The final detector still beats the factory-defaults
run by roughly $4$ percentage points of mAP\@50–95, so the sweep's
ordering of which #emph[further] changes help (B, H) and which do not
(C, D, E, F, G) remains valid.

=== Do the two scale-augmentation knobs stack?

A natural follow-up question: if `scale=0.5` and explicit `multi_scale=0.5`
both add scale variability, does combining them produce an even better
detector? The H recipe was retrained on yolo26-medium for 100 epochs with
`scale=0.5` stacked on top of `multi_scale=0.5`, to be compared against
the existing Phase 2 yolo26-medium run (same recipe, `scale=0.0`).

#figure(
  table(
    columns: (auto, 1fr, auto, auto),
    align: (left, left, right, right),
    stroke: (x, y) => (
      top: if y == 0 or y == 1 { 0.7pt } else { 0pt },
      bottom: if y == 2 { 0.7pt } else { 0pt },
    ),
    table.header(
      [*Run*], [*Scale augmentation*], [*mAP\@50*], [*mAP\@50–95*],
    ),
    [Phase 2 yolo26-medium],   [`multi_scale=0.5`, `scale=0.0`],       [*0.892*], [*0.814*],
    [Stack test (yolo26-medium)], [`multi_scale=0.5`, `scale=0.5` (stacked)], [0.849], [0.767],
  ),
  caption: [Stacking `scale=0.5` on top of explicit `multi_scale=0.5`
    regresses by roughly $5$ percentage points of mAP\@50–95, with most
    of the loss falling on recall ($0.842$ → $0.766$). The two knobs are
    redundant — both randomly resize the input — and combining them
    over-augments.],
) <tab:training-stack>

Taken together: `scale=0.5` is helpful only when no explicit multi-scale
schedule is in use. The shipping recipe therefore keeps `multi_scale=0.5`
and leaves `scale=0.0`. The project's CLI default for `scale` is set to
$0.5$ so that ad-hoc training runs (which usually do not enable
`multi_scale`) inherit the factory-style scale augmentation by default.

#bibliography("refs.bib", style: "ieee", title: "References")

#v(0.5em)
#line(length: 100%, stroke: 0.4pt + luma(180))
#v(0.4em)

#set text(size: 9pt, fill: luma(80))
#align(center)[
  Source images and labels live under `/mnt/data/ent_cv/datasets/exports/`.
  Figures regenerated by `scripts/gen_conventions_figures.py`. Document
  source: `docs/main.typ`. Citations tracked in `docs/refs.bib`.
]
