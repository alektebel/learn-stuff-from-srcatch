"""
Step 1 - the comparison bound, and how much slack it hides.

THE ARGUMENT
------------
An algorithm that learns about its input only through pairwise comparisons is a binary
decision tree: each internal node is a comparison, each branch an outcome, each leaf an
output permutation. Two facts, both one line:

  - the tree must have at least n! leaves;
  - a binary tree of depth d has at most 2^d leaves.

Therefore any comparison sort makes at least ceil(log2(n!)) comparisons in the worst
case, which is Omega(n log n) by Stirling.

That is the entire proof, and it is the cleanest lower bound you will meet. Everything
after this step is an attempt to do the same thing for a model that is not this weak,
and the attempts get progressively less successful.

WHAT TO WRITE
-------------
Two things: an instrumented sort so you can count real comparisons, and an exhaustive
search over decision trees so you can see how much room the bound leaves.

    DESIGN DECISION - exhaustive search over decision trees, not over algorithms.
    `min_comparisons(n)` searches for the shallowest decision tree that separates all n!
    permutations. It is a search over TREES, which is finite, not over programs, which is
    not. That substitution is the only reason the question is decidable, and it is worth
    noticing how much the substitution costs you: the tree does not have to correspond
    to any algorithm you could write down.
    Cost: it is doubly exponential. n = 5 is fine, n = 6 is a project, n = 12 is a
    published result (Wells, 1965) and not something you will reproduce here.
"""


def counting_sort_key(seq, counter):
    """Sort `seq` using ONLY comparisons, incrementing counter['n'] once per comparison.

    Any correct comparison sort. Merge sort is the natural choice because its worst case
    is provably close to the bound; insertion sort will pass the correctness test and
    lose badly on the count, which is itself informative.
    """
    raise NotImplementedError


def worst_case_comparisons(sort_fn, n):
    """Maximum number of comparisons `sort_fn` makes over ALL n! permutations of
    range(n). Exhaustive; n stays small."""
    raise NotImplementedError


def information_bound(n):
    """ceil(log2(n!)). Compute it exactly -- with integers and bit_length, not with
    math.log, which loses the ceiling at n = 12 and beyond for reasons worth knowing."""
    raise NotImplementedError


def min_comparisons(n):
    """The minimum worst-case comparison count over ALL decision trees for n elements.

    Exhaustive search. Feasible to n = 5, painful at n = 6. Returns an int.
    """
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 1)
#
#   information_bound(n) for n = 1..12:  0, 1, 3, 5, 7, 10, 13, 16, 19, 22, 26, 29
#
#   min_comparisons(n) for n = 1..5:     0, 1, 3, 5, 7
#     -- equal to the information bound. The bound is TIGHT here.
#
#   worst_case_comparisons(your sort, n) >= information_bound(n) for n = 1..7.
#   No comparison sort may go below it, ever, for any input.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   `min_comparisons` will not finish. Measured, n = 5:
#
#       memoised + cut off when the subtree already meets the bound:  737 calls, 0.01 s
#       memoised only:                                              4,231 calls, 0.10 s
#       cut off only:                                              10,331 calls, 0.04 s
#       neither:                                        still running after 2 minutes
#
#   Two independent things are needed and it is worth seeing why each helps: the
#   recursion revisits the same surviving SET of permutations by many different paths
#   (so memoise on it, which means it has to be hashable and canonically ordered), and
#   a branch whose subtree already achieves ceil(log2 |S|) cannot be improved (so stop
#   looking). Neither is an optimisation of the algorithm; both are consequences of the
#   same counting argument you just proved.
#
#   Second: `min_comparisons` that searches over which PAIR to compare next but forgets
#   that the answer may depend on the outcomes so far. A decision tree is ADAPTIVE -- the
#   two subtrees below a comparison may test different pairs. Fix the comparison
#   sequence in advance instead and you are computing the minimum size of a SORTING
#   NETWORK, which is a different and strictly larger quantity. If your n = 5 answer
#   comes out above 7, check this before anything else.
#   [I believe the minimum-size sorting network for n = 5 has 9 comparators, so that is
#   the number to expect from the non-adaptive version; I have not verified it here.]
#
#   Third, and not a failure so much as a thing to notice: computing
#   `information_bound` as math.ceil(math.log2(math.factorial(n))) is fine. I expected
#   float rounding to break it near an integer boundary and checked every n up to 3000
#   before writing this: it never diverges from the exact integer computation. The
#   integer version is still the one to write, because you should not have to check.
#
# PROSE QUESTION - answer in writing before step 2
#
#   (a) Radix sort runs in O(n) on fixed-width integers. It is not a counterexample.
#       State precisely which hypothesis of the argument it violates, and note that
#       this is the FIRST time in either track that a lower bound turned out to be a
#       statement about a model rather than about a problem. It will not be the last.
#
#   (b) The bound is tight for n <= 5 (both equal 7 at n = 5). It is NOT tight in
#       general: the minimum for n = 12 is 30 while ceil(log2 12!) = 29.
#       [Confidence: I am confident the n=12 result is 30 and that it is due to Wells;
#       I have not re-derived it and you should not quote it from this file.]
#       What does a one-comparison gap tell you about counting arguments in general?
# ---------------------------------------------------------------------------
