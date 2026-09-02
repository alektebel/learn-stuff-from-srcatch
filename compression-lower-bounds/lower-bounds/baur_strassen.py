"""
Step 3 - Baur-Strassen: all n partial derivatives for the price of a constant.

THE THEOREM
-----------
Let f be computed by an arithmetic circuit with L operations. Then f TOGETHER WITH all
n of its first partial derivatives can be computed with O(L) operations -- a constant
factor, independent of n.

Read that again. There are n derivatives. Computing them costs a constant times the cost
of computing the single value f. That is not obvious and it is the reason backpropagation
exists; the machine learning name for this theorem is "the backward pass costs the same
as the forward pass".

Its other use is what puts it in this directory. Run it backwards: if you can prove that
computing all partials of some explicit f requires many operations, you get a lower bound
on computing f itself, up to the constant. That is the source of the Omega(n log n) bound
for explicit polynomials -- and it is STILL essentially the best known, forty years on.
Step 5 asks you to take that sentence seriously.

WHAT TO DERIVE
--------------
Represent a circuit as a DAG whose nodes are inputs, constants, additions and
multiplications, in topological order. Then:

  - evaluate forward, keeping every intermediate value;
  - define the adjoint of node i as the partial derivative of the output with respect
    to that node's value;
  - derive the recursion the adjoints satisfy, by the chain rule, traversing the DAG
    in REVERSE topological order.

Do not write "backpropagation" and move on. Derive the recursion for the two gate types
and count the operations each contributes to the backward pass. That count is the
constant in the theorem.

    DESIGN DECISION - a DAG, not an expression tree.
    An expression tree would make the traversal trivial and the theorem false. The whole
    content is that a node feeding K consumers is evaluated ONCE forward and receives K
    adjoint contributions backward -- so the backward cost is proportional to the number
    of EDGES, which is what makes it O(L) and not O(nL). If your representation cannot
    share a subexpression, you have not built the object the theorem is about.
"""


def evaluate(nodes, x):
    """Evaluate the circuit at x.

    nodes: list, in topological order, of
        ('var', i)        - input variable i
        ('const', c)      - constant
        ('+', a, b)       - sum of nodes[a] and nodes[b]  (a, b < current index)
        ('*', a, b)       - product

    The output is the LAST node.
    Returns (values, n_ops) where n_ops counts arithmetic operations only.
    """
    raise NotImplementedError


def gradient(nodes, x):
    """Return (grad, n_ops_total) where grad[i] = d(output)/d(x_i) and n_ops_total
    counts the arithmetic in the forward pass AND the backward pass together.

    State your op-counting convention in a comment. The constant in the theorem depends
    on it, which is exactly why the acceptance test does not assert a constant.
    """
    raise NotImplementedError


def random_circuit(n, extra, rng):
    """A random DAG with n input variables and `extra` gates, each combining two
    uniformly chosen earlier nodes. Deterministic given rng."""
    raise NotImplementedError


# ---------------------------------------------------------------------------
# ACCEPTANCE TEST (check.py step 3)
#
#   1. `gradient` agrees with central finite differences to 1e-5 relative, on random
#      circuits, for a randomly chosen coordinate. Correctness first.
#   2. THE ACTUAL CONTENT: for n = 4, 16, 64, 256, 1024 with `extra = 4n` gates,
#
#          ratio = n_ops_total / n_ops_forward
#
#      must be bounded AND MUST NOT GROW WITH n. Required: ratio < 8 for every n, and
#      ratio(1024) < 1.25 * ratio(4). Measured with one particular convention (2 ops per
#      addition node and 4 per multiplication node in the backward pass): 4.00, 3.84,
#      4.05, 3.98, 4.00.
#
#      The checker does NOT assert 4. The constant is an artefact of how you count, and
#      the theorem is about the ratio being CONSTANT, not about its value. The reading
#      list quotes "<= 3x"; that is a different accounting, not a different theorem, and
#      you should not spend an afternoon trying to hit 3.
#
#   3. Sharing: a circuit built so that one node feeds many consumers must still give a
#      constant ratio. This is the test that distinguishes a DAG from a tree.
#
# PREDICTED FAILURE MODE (read before you implement)
#
#   Overwriting adjoints instead of accumulating them. `adj[a] = ...` rather than
#   `adj[a] += ...`. On a tree it is correct. On a DAG it silently drops every
#   contribution but the last, and the gradient is wrong only in the coordinates that
#   feed more than one gate -- so a finite-difference spot check on ONE random coordinate
#   passes most of the time. Check every coordinate, at least once.
#
#   Second: counting the forward pass twice, once inside `gradient` and once by calling
#   `evaluate` separately. The ratio comes out near 5 instead of 4 and you will not know
#   which of the two numbers is wrong.
#
# PROSE QUESTION - answer in writing before step 4
#
#   (a) Forward-mode differentiation computes ONE directional derivative per pass, so
#       all n partials cost n passes. Reverse mode computes all n in one. Both are exact
#       and both are the chain rule. Where does the asymmetry come from -- what is
#       different about the two ends of the DAG? Answer in terms of the shape of the
#       circuit, not in terms of which is "more efficient".
#
#   (b) A function with n inputs and n outputs. Which mode wins, and what does that tell
#       you about the scope of the theorem?
#
#   (c) The bound this theorem yields, Omega(n log n) for explicit polynomials, has stood
#       since 1983. Before you read anything about barriers: say what you would try, and
#       then say why you expect it not to work.
# ---------------------------------------------------------------------------
