"""
Step 2 - Strassen from the rank of a tensor, and the wall immediately behind it.

THE OBJECT
----------
Bilinear maps are tensors. Multiplication of n x n matrices is the tensor

        T[(i,j), (j',k), (k',i')] = 1  iff  j = j', k = k', i = i',

an n^2 x n^2 x n^2 array. Convince yourself this encodes C = A B: contract T with the
flattened A in the first slot and the flattened B in the second, and read C out of the
third. Note the index order in the third slot; getting it wrong gives you C transposed
and everything still "works" on symmetric test inputs.

A rank-R decomposition T = sum_{r=1..R} u_r (x) v_r (x) w_r is exactly an algorithm that
computes the product with R MULTIPLICATIONS: form the R scalars (u_r . a)(v_r . b), then
take linear combinations. Additions are free in this model, and understanding why that
is a defensible accounting is half the point.

The rank of the 2x2 tensor is 7, not 8. Recursively applied that gives
O(n^log2(7)) = O(n^2.807).

WHAT TO DERIVE FIRST
--------------------
Before you write any decomposition: why does a rank-R decomposition of the FIXED 2x2
tensor give an algorithm for every n, and where does log2(7) come from? Write that down.
It is a two-line argument about block matrices and a recursion, and it is the reason
anyone cares about the rank of one small tensor.

    DESIGN DECISION - count multiplications only.
    In the bilinear model additions are not counted. The justification is the recursion:
    at the top level each "multiplication" is a multiplication of n/2 x n/2 blocks, and
    each "addition" is an addition of n/2 x n/2 blocks -- O(n^2) against O(n^2.807).
    The additions are asymptotically free because they never recurse.
    Cost: for any n a real machine will run, the constant hidden in those additions
    dominates, which is why Strassen wins in practice only above a crossover in the
    hundreds. A model that makes an operation free is a model that will mislead you
    about that operation.
"""

import numpy as np


def matmul_tensor(n=2):
    """The <n,n,n> matrix multiplication tensor as an (n^2, n^2, n^2) numpy array."""
    raise NotImplementedError


def strassen_factors():
    """Return (U, V, W) with shapes (7, 4), (7, 4), (7, 4) such that

        sum_r  U[r] (x) V[r] (x) W[r]  ==  matmul_tensor(2)

    exactly, in integer arithmetic. Derive them, or reconstruct them from the seven
    products; either way you must be able to say which product each row is.
    """
    raise NotImplementedError


def multiply_from_factors(A, B, factors):
    """Compute A @ B for 2x2 matrices using ONLY the seven scalar products implied by
    `factors`. Returns (C, n_multiplications).

    n_multiplications must be 7. If your implementation cannot report that honestly,
    it is not using the decomposition.
    """
    raise NotImplementedError


def als_fit(T, rank, rng, iters=1500, restarts=8):
    """Best rank-`rank` fit to T by alternating least squares. Returns the smallest
    Frobenius residual found over `restarts` random restarts.

    This is a heuristic and it can get stuck; that is why there are restarts, and it is
    why the result is EVIDENCE and not a proof. Keep that distinction in the front of
    your mind when you read the acceptance test.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 2)
#
#   1. || sum_r U[r] (x) V[r] (x) W[r] - matmul_tensor(2) ||_max = 0.0. Exactly zero:
#      the entries are small integers and there is nothing to round.
#   2. `multiply_from_factors` reproduces A @ B to 1e-12 on random A, B, and reports
#      exactly 7 multiplications.
#   3. ALS residuals on the 2x2 tensor (||T|| = 2*sqrt(2) = 2.8284):
#
#          rank      5        6        7        8
#          residual 1.4142   1.0000   0.0000   0.0000
#
#      Required: rank 7 and 8 below 1e-6; rank 6 above 0.5. The rank-6 plateau at
#      exactly 1.000 is the observation to sit with.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   The third index. T's third slot is indexed by (k,i), not (i,k) -- the output comes
#   out transposed relative to the naive guess. Every symmetric test matrix passes,
#   A @ B on random matrices fails, and you will look for the bug in your seven products
#   for an hour. Test with a deliberately non-symmetric A and B on your very first run.
#
#   Second: ALS with rank > 4 on a 4x4x4 tensor is rank-deficient in the least-squares
#   subproblems, so `np.linalg.solve` raises or silently returns garbage. Use `lstsq`.
#   You will see this as rank-7 residuals that fail to reach zero, and conclude the rank
#   is 8.
#
# PROSE QUESTION - answer in writing before step 3
#
#   (a) ALS found 1.000 at rank 6 across 8 restarts. That is EVIDENCE that the best
#       rank-6 approximation has residual 1, not a proof: ALS is a local method and 8
#       restarts is 8 restarts. Say precisely what would have to be true for the
#       measurement to be misleading.
#
#   (b) Now the part that is a theorem and that the measurement is consistent with: the
#       BORDER rank of the 2x2 matmul tensor is also 7. Border rank is the smallest R
#       such that T is a LIMIT of rank-R tensors. Explain why a plateau strictly above
#       zero at rank 6 is what border rank 7 predicts, and why -- for a general tensor --
#       rank and border rank differing would make your measurement uninformative about
#       rank.
#       [Confidence: I am confident the rank of <2,2,2> is exactly 7 and that its border
#       rank is also 7. I have not re-derived either. Both are in Buergisser-Clausen-
#       Shokrollahi; go there rather than trusting this comment.]
#
#   (c) log2(7) = 2.807. The current record for omega is around 2.37, and it does NOT
#       come from finding the rank of a bigger matmul tensor by search. Say in one
#       sentence what changed, and why that makes your ALS experiment a dead end as a
#       research programme even though it is the right exercise here.
# ---------------------------------------------------------------------------
