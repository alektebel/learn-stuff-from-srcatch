"""
Step 4 - the red-blue pebble game. The only bound here that predicts a real number.

THE MODEL
---------
Two levels of memory: fast (M words, "red pebbles") and slow (unbounded, "blue"). A value
must be in fast memory to be used. Moving a word between the levels costs 1. Computation
inside fast memory is free.

Hong and Kung (1981) proved that any schedule of the standard n^3-operation matrix
multiplication algorithm moves Omega(n^3 / sqrt(M)) words. Note what the bound is over:
all SCHEDULES of one algorithm, not all algorithms. That restriction is what makes the
proof possible and is the subject of the prose question.

The shape of the argument, which you should reconstruct before implementing anything:
partition the computation into phases of M transfers each; bound how many multiplications
a single phase can perform given that at most 2M values are ever available to it; conclude
that the number of phases is at least n^3 divided by that bound. The per-phase bound is
where a geometric inequality on the three projections of a set of lattice points enters.
You do not have to prove that inequality; you do have to know it is the load-bearing step.

WHAT TO WRITE
-------------
A cache simulator and two schedules. Then measure, and compare against the bound.

    DESIGN DECISION - fully associative LRU, counting reads only.
    Real caches are set-associative and write-back, and both change the constant. LRU
    fully associative is the closest simple thing to the pebble game's "you may keep any
    M values you like", so it measures the model rather than a particular chip.
    Cost: your measured constant will not match a hardware performance counter, and you
    should not expect it to. What must match is the SCALING.
"""


class Cache:
    """Fully associative LRU over `M` words. `touch(key)` records an access and counts
    a miss when the key is not resident."""

    def __init__(self, M):
        raise NotImplementedError

    def touch(self, key):
        raise NotImplementedError


def naive_traffic(n, M):
    """Misses for the textbook i-j-k triple loop. Keys: ('A',i,k), ('B',k,j), ('C',i,j)."""
    raise NotImplementedError


def tiled_traffic(n, M, b):
    """Misses for the b x b blocked schedule: the same n^3 multiply-adds, reordered."""
    raise NotImplementedError


def best_tile(M):
    """The tile size the bound suggests. Derive it: three b x b tiles must be resident
    simultaneously. One line, and it is the only place the sqrt(M) enters."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 4)
#
#   Over n in {32, 48, 64} and M in {48, 108, 192, 300} -- note M << n^2, which matters --
#   with b = best_tile(M), form
#
#          q = traffic * sqrt(M) / n^3.
#
#   TILED:  q must lie in [4, 8] for every (n, M), and max(q)/min(q) < 1.5 across the
#           whole grid. Measured: 5.09 to 5.42. Constant q IS the n^3/sqrt(M) scaling;
#           that is the whole measurement.
#
#   NAIVE:  q must vary by more than a factor of 1.5 across the same grid. Measured:
#           10.8 to 20.9. Its traffic is ~n^3 regardless of M, so q drifts like sqrt(M).
#           The naive schedule does not obey the law; it merely fails to violate it.
#
#   Also: both schedules must perform the same n^3 multiply-adds. If `tiled_traffic`
#   changes the arithmetic, it is not the same algorithm and the bound does not apply
#   to it.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   Choosing M >= n^2. Then a whole matrix fits in fast memory, the asymptotic regime
#   has not started, and the naive loop BEATS the tiled one. At n = 48, M = 3072 the
#   measured traffic was naive 6,912 versus tiled 12,176. Nothing is wrong; you are
#   simply not in the regime the theorem describes. Every acceptance number above has
#   M << n^2 for this reason, and noticing that this is a precondition rather than a
#   convenience is most of the lesson.
#
#   Second: an LRU that does not move a key to the front on a HIT. It is still a valid
#   eviction policy, but it is FIFO, not LRU, and the tiled constant comes out wrong
#   while the naive one barely moves -- so it looks like a bug in the tiling.
#
#   Third: forgetting that C[i][j] is both read and written. Count it once per access,
#   consistently, in both schedules, or the comparison measures your bookkeeping.
#
# PROSE QUESTION - answer in writing before step 5
#
#   (a) The bound is over all schedules of the standard algorithm. Strassen performs
#       O(n^2.807) operations and therefore is NOT covered. Does Strassen beat
#       Omega(n^3/sqrt(M))? [Ballard-Demmel-Holtz-Schwartz answer this; the answer is
#       not "no". Look it up rather than guessing, and note that I am not stating the
#       exponent here because I do not remember it with confidence.]
#
#   (b) This is the only bound in this directory that a hardware engineer uses. Say what
#       property of it makes it useful where Baur-Strassen's Omega(n log n) is not. The
#       answer is not that one is bigger.
#
#   (c) Your measured constant is about 5. The bound's constant is not 5. Explain why
#       the two need not agree and what your measurement does and does not establish.
# ---------------------------------------------------------------------------
