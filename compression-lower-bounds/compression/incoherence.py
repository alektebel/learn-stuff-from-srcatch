"""
Step 4 — incoherence processing. The step the rest of the field is downstream of.

THE PROBLEM
-----------
Steps 2 and 3 quantize an i.i.d. Gaussian source and every number they produce assumes
it. A weight matrix from a trained transformer is not that. It has outlier rows, outlier
columns, and directions of much larger variance than the rest. Squared-error distortion
is dominated by exactly those, so a codebook designed for a Gaussian spends most of its
levels covering a tail that holds almost no mass.

You could design a codebook for the actual empirical distribution. QuIP does something
better: change the matrix so the Gaussian assumption becomes true.

THE OBJECT
----------
For W in R^(m x n), define the incoherence parameter

        mu(W) = max_ij |W_ij| * sqrt(m*n) / ||W||_F.

Convince yourself: mu >= 1 always, with equality exactly when every entry has the same
magnitude. Large mu means the Frobenius mass is concentrated in a few entries. mu is
invariant under scaling of W, which matters later.

The transform is W -> (H S1) W (S2 H^T) / normalisation, with H the Sylvester-Hadamard
matrix and S1, S2 independent diagonal +-1 (Rademacher) matrices. Two things to derive
before writing anything:

  (a) What normalisation makes H orthogonal? The answer decides whether the transform
      preserves ||W||_F, which decides whether the mu you compute afterwards is
      comparable to the one before.
  (b) Why is a two-sided transform needed rather than one-sided? What is preserved?

    DESIGN DECISION - Hadamard rather than a random orthogonal matrix.
    A uniformly random orthogonal Q also produces incoherence, with better constants.
    Hadamard is chosen because H*x costs O(n log n) with no multiplications, while Q*x
    costs O(n^2) and has to be stored. At inference time the transform runs on every
    forward pass, so the asymptotics are the entire argument.
    Cost: H exists only for n a power of 2 (and a few other sizes), so real
    implementations pad or block, and the sign flips are doing work the structure of H
    does not do by itself. That last part is the acceptance test.

WHAT TO WRITE
-------------
"""

import numpy as np


def hadamard(n):
    """The n x n Sylvester-Hadamard matrix, UNNORMALISED (entries +-1). n a power of 2."""
    raise NotImplementedError


def fast_hadamard(x):
    """H @ x in O(n log n), where H is the unnormalised Sylvester-Hadamard matrix.

    x: (n,) or (n, k). No explicit construction of H.
    """
    raise NotImplementedError


def incoherence(W):
    """mu(W) as defined above. Returns a float."""
    raise NotImplementedError


def random_hadamard_transform(W, rng):
    """Two-sided randomised Hadamard transform of W. Returns the transformed matrix.

    Must be exactly invertible given the same signs; the caller keeps the seed, not the
    matrix.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 4)
#
#   1. Orthogonality, to machine precision: for n in {64, 256, 1024}, with H the
#      normalised transform, ||H H^T - I||_max < 1e-12. (Exactly 0.0 for a correct
#      Sylvester construction in float64, since every entry is a dyadic rational.)
#   2. `fast_hadamard(x)` equals `hadamard(n) @ x` to 1e-10.
#   3. mu >= 1 for every matrix, and mu = 1 for an all-+-1 matrix.
#   3b. ||RHT(W)||_F / ||W||_F = 1 EXACTLY. Testing hadamard() on its own cannot see a
#      missing 1/sqrt(n) inside the transform, because the test can normalise for you.
#   3c. W = I, n = 256: mu(RHT(I)) must be ~3. If it is exactly sqrt(256) = 16.0, the
#      same sign vector is being used on both sides -- and S^2 = I makes the identity a
#      fixed point of that transform, so no randomisation happens at all.
#   4. Planted outlier: W random Gaussian with row 3 scaled by 60.
#           n = 1024:  mu(W) = 97.6  ->  mu(RHT(W)) = 4.7
#      Required: mu after < 8, and mu after grows slower than any power of n across
#      n in {64, 256, 1024} (measured: 3.20, 3.49, 4.72; compare sqrt(2 log n^2) =
#      4.08, 4.71, 5.27).
#   5. THE ONE THAT MATTERS. W = outer(h_7, h_11)/n, where h_i is a row of the
#      unnormalised Hadamard matrix. This W is PERFECTLY incoherent: mu(W) = 1.0.
#      Apply the DETERMINISTIC transform H W H^T. Measured at n = 256:
#
#           mu(W) = 1.00   ->   mu(H W H^T) = 256.0   ->   mu(RHT(W)) = 7.2
#
#      The deterministic Hadamard transform takes the most incoherent matrix that exists
#      and makes it the LEAST incoherent one possible (all the mass in a single entry,
#      mu = n). The random signs are not a theoretical nicety; they are the only thing
#      standing between you and an adversarial input. Your implementation must reproduce
#      all three numbers.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   Normalisation. If you build H with entries +-1 and forget the 1/sqrt(n), mu is
#   UNAFFECTED, because mu is scale-invariant. Every test in this file passes. The bug
#   surfaces in step 5, where the Hessian is no longer the Hessian of the same problem
#   and GPTQ's loss ratio drifts toward 1 for no visible reason. Test 1 exists solely to
#   catch this now.
#
#   Second: applying the same sign vector on both sides. It is one character of
#   difference and it destroys the independence the argument needs. Tests 4 and 5 both
#   still pass -- the adversarial matrix is not symmetric, so it does not expose the
#   coupling. Test 3c is the one that does, and it took a specifically constructed input
#   (the identity) to find. Note the shape of that: an invariant that holds for random
#   inputs and fails on a structured one is exactly the kind of bug randomised
#   preprocessing is supposed to prevent, and here it is in the preprocessing itself.
#
# PROSE QUESTION - answer in writing before step 5
#
#   (a) mu is scale-invariant. Name a quantity in step 5's objective that is not, and
#       say what breaks if the transform is orthogonal only up to a scalar.
#
#   (b) QuIP's guarantee is stated in terms of mu of the WEIGHTS and mu of the HESSIAN.
#       Why does the Hessian need its own incoherence condition - what goes wrong if the
#       weights are incoherent but the Hessian's eigenvectors are axis-aligned?
#
#   (c) You just measured that mu after RHT grows roughly like sqrt(log(mn)) rather than
#       staying at 1. Where does that log come from? [If you want the mechanism: it is a
#       maximum over mn approximately-Gaussian entries. Making that precise is the
#       exercise.]
# ---------------------------------------------------------------------------
