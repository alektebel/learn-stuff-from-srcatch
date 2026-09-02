"""
Step 6 - trellis-coded quantization. Where the codebook stops being stored.

THE WALL YOU HIT IN STEP 3
--------------------------
A d-dimensional vector quantizer at rate R bits/dimension needs 2^(R*d) codewords. At
R = 2, d = 8 that is 65,536 - fine. At d = 16 it is 4 billion. The space-filling gain
keeps improving with d (up to 1.53 dB), but the codebook size is exponential in d, and
that, not the theory, is why QuIP# stops at 8 dimensions.

TCQ breaks the coupling. The encoder is a finite state machine: at each step it consumes
k bits, moves to a new state, and emits a codeword that is a FUNCTION OF THE STATE. With
an L-bit state register there are 2^L reachable codewords, but you never store them -
you compute them. The rate is k bits per sample regardless of L, so codebook size and
bitrate are decoupled, which is exactly the sentence in QTIP's abstract.

THE OBJECT
----------
A bitshift trellis: state s is L bits, and the transition on input bits b is
s' = ((s << k) | b) mod 2^L. The codeword for state s is f(s) where f is a fixed
pseudorandom map onto (approximately) a standard Gaussian - a hash, then an inverse CDF.

Encoding a sequence x_1..x_T means choosing the state path minimising sum (x_t - f(s_t))^2.
That is a shortest path on a DAG with T*2^L nodes. Derive the recursion; it is the only
thing that makes this tractable, and it is not written here.

    DESIGN DECISION - the codebook is a hash, not a trained codebook.
    Nothing stops you from optimising the 2^L codeword values by Lloyd on the marginal
    distribution. QTIP does not, and the reason is arithmetic: f(s) must be computable
    inside the inner loop of a GPU kernel from the state alone, with no memory access.
    A trained codebook is a lookup table, and a lookup table is the thing being avoided.
    Cost: you give up shape optimality. You will measure it - at L = 2 the trellis is
    twice as bad as scalar Lloyd-Max at the same rate.
"""

import numpy as np


def state_codebook(L):
    """The value f(s) for each of the 2^L states.

    Requirements: deterministic, computable from s alone, and with an empirical
    distribution close to a standard Gaussian. A multiplicative hash to a uniform in
    (0,1) followed by the normal inverse CDF is one way; say what you chose.

    scipy is not assumed anywhere in this directory. If you go the inverse-CDF route,
    math.erf plus 50 steps of bisection is accurate to well past float64 noise and is
    six lines.
    """
    raise NotImplementedError


def viterbi(x, L, k):
    """Encode x with the bitshift trellis. Returns (reconstruction, distortion).

    x: (T,) array. Rate is k bits per sample. Free initial and final state.
    reconstruction: (T,) array of codeword values.
    distortion: mean squared error.
    """
    raise NotImplementedError


def brute_force(x, L, k):
    """Exhaustive minimum over all state paths. Only usable for tiny T; exists to
    certify `viterbi`, and for no other reason."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 6)
#
#   1. `viterbi` and `brute_force` return the SAME distortion for L = 3, k = 1, T = 8,
#      on 20 random inputs, to 1e-12. Not similar - the same. The recursion is exact.
#
#   2. On 3000 standard normal samples at k = 2 bits/sample:
#
#          L          2       4       6       8      10
#          D        0.243   0.133   0.111   0.087   0.079
#
#      Reference points at the same rate: scalar Lloyd-Max D = 0.1175, Shannon
#      D(R=2) = 2^-4 = 0.0625.
#
#      THOSE NUMBERS ARE HASH-SPECIFIC. They are what ONE choice of f produces. Across
#      three different hash families and two seeds the L=10 value ranged 0.0735-0.0832
#      and the L=2 value ranged 0.243-0.567, so do not chase my column. What survived
#      every variant, and what the checker actually asserts:
#
#        - D strictly decreasing in L;
#        - D(L=2)  >  D(scalar) = 0.1175;
#        - D(L=10) <  0.80 * D(scalar) = 0.0940;
#        - D > 0.0625 for every L, because nothing may beat the rate-distortion
#          function.
#      That last one is the only assertion in this directory that is a theorem rather
#      than a measurement, and it is the one worth watching - if your D drops below
#      0.0625 you have a bug, guaranteed, no matter how good the number looks.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   The traceback. Storing, for each (t, state), the best PREDECESSOR versus the best
#   INPUT BITS is a one-line difference and both compile. Get it wrong and you do not
#   crash: you emit a valid state path that is not the optimal one, and the distortion
#   comes out a few percent worse than scalar Lloyd-Max instead of better. You will
#   conclude that trellis coding does not work at this rate, which is a much more
#   comfortable conclusion than "my traceback is off by one". Test 1 exists to make that
#   conclusion unavailable to you.
#
#   Second: L = 2 being WORSE than scalar is correct and expected, not a bug. Do not fix
#   it. Understanding why it is correct is the prose question.
#
#   Third: forgetting that multiple predecessors map to the same successor state. The
#   forward pass must take a MINIMUM over predecessors, not a scatter-assign; numpy's
#   fancy-index assignment silently keeps whichever one it wrote last.
#
# PROSE QUESTION - answer in writing before the final checkpoint
#
#   (a) The codeword values are a hash - a random codebook, not an optimised one. At
#       L = 2 that random codebook is twice as bad as Lloyd-Max at the same rate. At
#       L = 10 it beats it by 33%. The codebook did not get better. What did?
#
#   (b) You now have three mechanisms that all buy distortion at fixed rate: better
#       level placement (step 2), better cell shape (step 3), and memory across samples
#       (step 6). Gersho & Gray call these shape, space-filling and memory gain. The
#       source here is i.i.d. Gaussian - it has NO memory to exploit. So where is the
#       trellis gain coming from, and which of the three names does it actually deserve?
#
#   (c) Put steps 4, 5 and 6 together on paper: what is the assumption each one makes
#       about its input, and which earlier step is responsible for making it true?
# ---------------------------------------------------------------------------
