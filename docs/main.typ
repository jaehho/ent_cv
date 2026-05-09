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
    hand-coupled (Convention B).
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

== Convention B: hand-coupled instrument bounding boxes

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
  dataset under a hand-coupled instrument convention, in which each box
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
  instrument is occluded by tissue, a hand-coupled box still grounds the
  detection on the visible glove.

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
