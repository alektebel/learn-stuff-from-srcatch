"""
Step 3 — quantizing in 8 dimensions, and where the gain actually comes from.

WHY THIS STEP EXISTS
--------------------
Step 2 gave an optimal SCALAR quantizer, and it still loses 4.35 dB to the Shannon
rate-distortion function for the same source at the same rate. The source was
memoryless, so there is no correlation left to exploit. Something else is being lost.

That something has two named parts, and separating them is the point of this step:

  - SHAPE gain: a scalar quantizer with equal-probability cells is not optimal for a
    Gaussian; the density varies. Lloyd-Max already captures this.
  - SPACE-FILLING gain: in d dimensions the optimal cell is not a cube. A cube is what
    you get from quantizing each coordinate independently, no matter how well you place
    the levels along each axis. This part is invisible in d = 1 and it is the entire
    reason QuIP# uses a lattice.

THE OBJECT
----------
For a lattice L with fundamental cell volume V, the normalised second moment is

        G(L) = ( E ||x - Q_L(x)||^2 / d ) / V^(2/d),        x uniform on a cell.

It is dimensionless and scale-invariant: G is a property of the SHAPE of the Voronoi
cell, nothing else. G(Z) = 1/12 exactly. The space-filling gain of L over the integer
lattice is 10*log10( (1/12) / G(L) ) dB.

    DESIGN DECISION — Z^8 and E8, both with determinant 1.
    Both lattices here are unimodular, so V = 1 and G reduces to the raw second moment.
    That removes a volume computation, which is not the lesson. If you add A2 (the
    hexagonal lattice) as an extra, you must scale it to unit determinant first or your
    G will be wrong by det^(2/d) and will look like a decoder bug.

WHAT TO WRITE
-------------
E8 = D8 union (D8 + 1/2), where D8 is the set of integer vectors with even coordinate
sum. You need a nearest-point decoder. Derive it:

  - Nearest point of Z^8 to x: trivial.
  - Nearest point of D8 to x: start from the Z^8 answer. If the parity is already even
    you are done. If not, you must move to a different integer vector; among all integer
    vectors with the opposite parity, which is closest, and how much does the move cost?
  - Nearest point of E8: two cosets, decode into each, keep the closer.

No searching. Each of these is O(d).
"""

import numpy as np


def z_nearest(x):
    """Nearest point of Z^d. x: (n, d) -> (n, d)."""
    raise NotImplementedError


def d8_nearest(x):
    """Nearest point of D8 (integer vectors with even coordinate sum). x: (n, 8)."""
    raise NotImplementedError


def e8_nearest(x):
    """Nearest point of E8 = D8 u (D8 + 1/2). x: (n, 8)."""
    raise NotImplementedError


def normalised_second_moment(decoder, d, n_samples, rng, det=1.0):
    """Monte-Carlo estimate of G for a lattice given its nearest-point decoder.

    Sample uniformly over a region large enough that edge effects are negligible, or
    (better, and worth thinking about) uniformly over one fundamental cell. State which
    you chose and why it is unbiased.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 3)
#
#   With 400,000 samples:
#     G(Z^8) = 0.08333   within 1%          (exact: 1/12)
#     G(E8)  = 0.07168   within 2%
#     gain   = 0.654 dB  within 0.05 dB
#
#   And structurally, with no tolerance at all: every vector returned by `e8_nearest`
#   must be a genuine E8 point. Either all its coordinates are integers with even sum,
#   or all are half-odd-integers with sum an integer. Test this on random input; it
#   catches decoder bugs that the value of G alone does not.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   The D8 parity fix. When the rounded vector has odd coordinate sum you must flip
#   exactly one coordinate to its next-nearest integer, and it must be the coordinate
#   whose rounding error was LARGEST in absolute value. Flip the smallest instead — the
#   obvious-looking "least damage" choice — and you get:
#
#       decoder                       G         gain vs Z
#       Z^8                        0.08325       +0.005 dB
#       D8 (scaled to det 1)       0.07585       +0.409 dB
#       E8, correct                0.07169       +0.653 dB
#       E8, wrong coordinate       0.10196       -0.876 dB     <- worse than scalar
#
#   The wrong version is loud: negative gain. But the D8-only version is quiet — 0.409 dB
#   is a perfectly plausible-looking number and you get it by forgetting the +1/2 coset
#   entirely. If your gain lands near 0.41 dB, you built D8, not E8.
#
#   Second failure: exact ties at x_i = k + 0.5, where numpy's round-half-to-even makes
#   the rounding error exactly 0.5 and `argmax` picks an arbitrary coordinate. It costs
#   nothing on continuous input and breaks reproducibility on any structured test vector.
#
# PROSE QUESTION — answer in writing before step 4
#
#   E8 is optimal for SPHERE PACKING in dimension 8 (Viazovska, 2016). Explain why that
#   is a different statement from "E8 minimises G in dimension 8", and say which of the
#   two a quantizer actually needs. Packing, covering and quantizing are three distinct
#   optimality criteria on lattices, and they do not have to have the same optimum.
#
#   [Confidence note: the packing result is Viazovska's theorem and I am confident in it.
#   Whether E8 is PROVEN optimal for the quantization criterion in dimension 8 I do not
#   know and have not checked. Do not take a claim either way from this file.]
#
#   Second: the total gap from scalar Lloyd-Max to the Shannon bound is 4.35 dB, of
#   which the space-filling part is at most 10*log10(2*pi*e/12) = 1.53 dB, approached
#   only as d -> infinity. E8 recovers 0.654 of that 1.53. Where does the other 2.8 dB
#   live, and which technique in the reading list goes after it?
# ---------------------------------------------------------------------------
