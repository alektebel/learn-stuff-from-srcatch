"""
Step 5 — GPTQ on one linear layer. Two afternoons, not one.

THE PROBLEM
-----------
Steps 1-4 minimise E||x - Q(x)||^2 on the weights themselves. That is the wrong
objective. A layer computes W X for calibration activations X; what you care about is
that the OUTPUT barely changes. So:

        minimise   || (W - W_hat) X ||_F^2      over W_hat with entries in a grid.

Expand it. You get a quadratic form in the error E = W - W_hat with a matrix that
depends only on X. Write that matrix down; call it H. Note what it is NOT: it is not the
Hessian of the network loss, it is the exact Hessian of this layer-local proxy, and the
distinction is the whole reason the method is cheap.

Now the key structural observation, and the one to derive rather than read:

  Suppose you have already fixed W_hat on columns 1..q and the remaining columns are
  free. Fix the quantized value of column q. The remaining columns can then be
  re-optimised in CLOSED FORM, because the objective is quadratic and unconstrained in
  them. Derive that closed form. It is one line of block matrix inverse.

That closed form is "error feedback". It is not a heuristic and it is not a gradient
step; it is the exact minimiser of the remaining problem.

    DESIGN DECISION - quantize columns in a fixed order, one at a time.
    The joint problem (choose all grid points simultaneously) is an integer least-squares
    problem: NP-hard in general, and exactly the problem lattice decoding solves in low
    dimension. Sequential-with-exact-reoptimisation is a greedy relaxation of it.
    Cost: named in the prose question. Do not skip it - "greedy but each step is optimal"
    is precisely the kind of sentence that sounds like a correctness proof and is not.

    DESIGN DECISION - Cholesky of H^-1, not of H.
    The update needs a row of H^-1 restricted to the not-yet-quantized columns, for every
    q. Computing H_FF^-1 afresh per column is O(n^4) overall. One Cholesky factor of
    H^-1 gives all of them. Deriving why the rows of that factor carry exactly the right
    quantity is the second half of the derivation above.
"""

import numpy as np


def hessian(X, damping=0.01):
    """The matrix of the quadratic form, plus damping.

    X: (d_in, n_samples) calibration activations.
    Returns (d_in, d_in).

    `damping` is a fraction of the mean diagonal, added to the diagonal. You will find
    out below why it is not optional.
    """
    raise NotImplementedError


def quantize_rtn(W, bits):
    """Round-to-nearest baseline. Symmetric grid, one scale per output row (per row of
    W), so that it is a fair comparison with `gptq` below. Returns W_hat."""
    raise NotImplementedError


def optimal_update(H, q, err):
    """The exact re-optimisation of columns q+1.. after column q has been fixed with
    error `err` = W_hat[:, q] - W[:, q].

    Returns the delta to ADD to W[:, q+1:], shape (d_out, d_in - q - 1).

    This is factored out of `gptq` on purpose: it is the one thing in this file that can
    be tested against a brute-force least-squares solve with NO tolerance games, and it
    is the thing everyone gets wrong by a sign.
    """
    raise NotImplementedError


def gptq(W, H, bits):
    """Quantize W column by column against H, applying `optimal_update` after each.

    W: (d_out, d_in), H: (d_in, d_in). Returns W_hat with the same shape.
    Use one scale per row of W, as in `quantize_rtn`, or the comparison is meaningless.
    """
    raise NotImplementedError


def proxy_loss(W, W_hat, X):
    """|| (W - W_hat) X ||_F^2."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 5)
#
#   1. THE HARD ONE, and the only test here with no tolerance to hide in. For a random
#      SPD H and a random column, `optimal_update` must agree, to 1e-9, with the
#      brute-force minimiser obtained by solving the (d-1)x(d-1) least-squares problem
#      directly. Measured agreement for a correct implementation: 0.0, exactly. If this
#      test does not pass, nothing below means anything - the rest of the file will
#      still produce plausible numbers.
#
#   2. On the checker's synthetic layer (d_out=64, d_in=128, 512 correlated samples),
#      ratio = proxy_loss(GPTQ) / proxy_loss(RTN):
#
#          bits     2       3       4       8
#          ratio   0.34    0.25    0.28    0.30
#
#      Required: ratio < 0.6 at every bit width. A ratio near 1.0 means the update is
#      not being applied; a ratio above 1.0 means it is being applied with the wrong
#      sign, which is the single most common bug in this file.
#
#   3. Sanity: at 16 bits both losses are near zero and the ratio is meaningless. The
#      checker does not test it. Know why it does not.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   Damping, and it will not look like a damping problem. With n_samples < d_in, H is
#   singular by construction (rank at most n_samples) and the Cholesky raises. The
#   obvious fix is to raise the damping until it stops raising. Do that and GPTQ
#   degenerates smoothly toward RTN - the ratio climbs from 0.28 through 0.5, 0.8, to
#   1.0 - with no error at any point. You will read the ratio as "GPTQ does not help on
#   my problem".
#
#   Instrument it: sweep the damping over three orders of magnitude and plot the ratio.
#   The curve should be flat and then rise. If it is rising already at your chosen
#   value, your calibration set is too small, and that is a data problem, not an
#   algorithm problem.
#
#   Second failure: recomputing the grid scale from the UPDATED columns as you go.
#   Tempting, and it improves the loss - but then `quantize_rtn` and `gptq` are no longer
#   using the same grid, and the ratio you report is measuring two changes at once.
#
# PROSE QUESTION - answer in writing before step 6
#
#   (a) Every step of GPTQ is the exact minimiser of the remaining problem given the
#       past. Explain, precisely, why the result is nonetheless not the global minimiser
#       of the original objective. What would the optimal procedure cost, and what
#       classical problem is it?
#
#   (b) GPTQ processes columns left to right. The paper argues the order is not very
#       important for large layers; other work orders columns by decreasing diagonal of
#       H. Give the argument for why the order should matter, then give the argument for
#       why it might not, and say which one your own measurement supports. [I am not
#       supplying either argument, and I am not confident enough about the exact claim
#       in the GPTQ paper's text to paraphrase it - go read section 3 rather than trust
#       a summary.]
#
#   (c) H is the Hessian of the LAYER-LOCAL proxy, not of the network loss. Name a
#       failure mode of the whole method that this substitution creates, and say which
#       paper in READING.md attacks it.
# ---------------------------------------------------------------------------
