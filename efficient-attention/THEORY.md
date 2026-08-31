# The theory, in the order you will implement it

Read a section, implement it, come back. Reading all of this before writing any code
is the failure mode this repository exists to prevent — the arguments below only bite
once you have a working thing to test them against.

Notation throughout: `N` queries, `M` keys and values (`N = M` for self-attention),
head dimension `d`, value dimension `dv` (usually `= d`). One head; multi-head is this
`H` times.

---

## 0. What standard attention costs, and which cost each paper attacks

```
S = Q K^T / sqrt(d)        (N, M)      2 N M d flops
P = softmax(S, axis=-1)    (N, M)      ~5 N M flops
O = P V                    (N, dv)     2 N M dv flops
```

Time is `O(N M d)`. Memory is `O(N M)` — and the memory is the problem, for three
reasons that are worth separating because the three papers separate them:

1. **Footprint.** `S` and `P` are `N x M`. At `N = 8192`, fp16, that is 128 MB per
   head per layer, and during training it must stay alive until the backward pass.
   32 heads and 32 layers do not fit on any GPU that exists.
2. **Traffic.** Even when it fits, `S` is written to HBM by the matmul kernel, read
   back by the softmax kernel, written again, and read again by the second matmul.
   Attention at these shapes is **memory bound**: an A100 does ~312 TFLOP/s of fp16
   matmul but only ~2 TB/s of HBM, a ratio of about 150 flops per byte. Attention has
   an arithmetic intensity far below that, so the GPU spends most of its time waiting.
3. **Asymptotics.** `N^2` growth caps the context length outright.

The three methods attack different items on that list, and this is the single most
useful thing to keep straight:

| | attacks | exact? | causal? |
|---|---|---|---|
| Clustered attention | flops **and** footprint (1, 3) | no — bounded approximation | no, encoder only |
| Linear attention | flops **and** footprint (1, 3) | no — a *different* function | yes, and it is the point |
| FlashAttention | traffic **and** footprint (1, 2) | **yes**, bit-for-bit in exact arithmetic | yes |

FlashAttention does not reduce flops at all. Clustered and linear attention do not
reduce traffic per flop. Reaching for the wrong one is the standard mistake.

---

## 1. Clustered attention

> Vyas, Katharopoulos, Fleuret. *Fast Transformers with Clustered Attention.*
> NeurIPS 2020. arXiv:2007.04825

### The core idea in plain language

Look at the attention matrix of a trained transformer and many rows are near-duplicates
of each other. Queries that are close in space attend to the same keys with nearly the
same weights, so computing each of those rows separately is doing the same work many
times.

So: **group the queries, compute one attention row per group, and give every member of
the group the group's answer.** With `C` groups you have replaced `N` rows of attention
with `C`, and the cost stops depending on `N` except through the clustering itself.

### The mathematics

Partition the queries into `C` clusters `S_1..S_C` and let `Q^c_j` be the mean of the
queries in cluster `j`. Then

```
A^c = softmax(Q^c K^T / sqrt(d))      (C, M)     C rows, not N
V^c = A^c V                           (C, dv)
V̂_i = V^c_{cluster(i)}                (N, dv)    a gather, no arithmetic
```

Clustering is done in Hamming space on LSH bits, not in `R^d`: hash each query with
`B` random hyperplanes (`bit_b = [Q_i . r_b > 0]`), then run K-means with Hamming
distance and majority-bit centroids. For unit-norm vectors `P[bit differs] = theta/pi`,
so Hamming distance is a monotone estimator of angular distance, and each distance is a
popcount rather than `d` multiply-adds. The centroid used *in the attention* is still
the mean of the cluster's queries in `R^d` — two different spaces, and conflating them
is the usual bug.

### Complexity

```
clustering       O(N d B  +  N C B T)      hashing + T K-means iterations
attention        O(C M d  +  C M dv)
gather           O(N dv)
memory           O(C M)                    the centroid attention matrix
```

With `C` fixed this is **linear in sequence length**, in both time and memory.

### The guarantee

The paper's argument, which you re-derive in step 5, is that the error is controlled by
how far each query is from its centroid. Writing `a = softmax(u)` for the true row and
`b = softmax(v)` for the centroid's, with `delta = ||u - v||_inf`:

```
|log a_n - log b_n| <= |u_n - v_n| + |log Z_u - log Z_v| <= 2 delta
  =>  ||a - b||_1 <= exp(2 delta) - 1

||V_i - V̂_i||_2  <=  ||a - b||_1 * max_n ||V_n||_2
                  <=  ( exp( 2 ||Q_i - c||_2 max_n||K_n||_2 / sqrt(d) ) - 1 ) * max_n||V_n||_2
```

Two honest observations, both of which you will measure:

* The bound is **exponential** in the query-to-centroid distance, so numerically it is
  often worse than the trivial `2 max||V||` that holds for any two distributions. At
  cluster spread 0.3 it says nothing; at spread 0.01 it is ~15x tighter than trivial.
* What it *says* is nonetheless the right thing, and it is why the method is usable:
  **error is a continuous function of clustering quality and vanishes with it.** You can
  buy accuracy with clusters. A method whose failure mode is unbounded would not let you.

### Improved clustered attention

The plain version's weakness is that a peaked attention distribution is exactly the case
where two nearby queries can disagree — they may pick different top keys. The fix is to
use the centroid pass only to *find* the important keys, then be exact about them.

For query `i` in cluster `j`, with `T` the cluster's top-`k` keys by `A^c_j`:

```
w        = softmax over T of (Q_i . K_n / sqrt(d))       exact, k terms
m_j      = sum_{n in T} A^c_{jn}                         centroid's mass on T
V̂_i      = m_j * sum_{n in T} w_n V_n   +   ( V^c_j - sum_{n in T} A^c_{jn} V_n )
           \_____ this query's own preferences _____/     \_ the rest, as the centroid saw it _/
```

This works because a softmax restricted to a subset and renormalised is exactly the
conditional distribution of the full softmax on that subset. Cost becomes
`O(C M d + N k d)` — still linear in `N`.

The guarantee improves qualitatively:

```
||V_i - V̂_i||_2  <=  2 [ (1 - alpha_i) + (1 - m_j) ] * max_n ||V_n||_2
```

where `alpha_i` is the query's own true mass on `T`. **Missing mass is the only way to
be wrong**, and at `k = M` the error is exactly zero for every query *regardless of how
bad the clustering was*. That is a much stronger statement than the exponential bound,
and the checker verifies it as an exactness property rather than a limit.

Measured in this directory (256 queries in 8 real groups, 32 clusters, `d = 32`):

```
                    relative error vs exact attention
clustered                        0.084
+ top-8 exact keys               0.015
+ top-32 exact keys              0.0026
```

### Compared with standard attention

Same output shape, same interface, `C/N` of the attention flops, and an error you can
trade against `C` and `k`. What you give up:

* **Causal masking.** Members of a cluster sit at different positions, so no single mask
  is right for the group, and recomputing per member is the cost you were avoiding.
  This is an encoder method — which is why the paper evaluates on speech recognition and
  why linear attention, not this, is what people reach for in decoders.
* **Data dependence.** On iid Gaussian queries — where nothing clusters — the relative
  error in this directory is ~0.99, i.e. total garbage. The method's accuracy is a
  property of your data, and it is one you must measure rather than assume.
* **The clustering is not free**, and in a naive implementation it dominates: our numpy
  K-means loop makes clustered attention *slower* in wall clock than plain attention up
  to `N = 1024`, even though it does 10x fewer flops.

---

## 2. Linear attention

> Katharopoulos, Vyas, Pappas, Fleuret. *Transformers are RNNs: Fast Autoregressive
> Transformers with Linear Attention.* ICML 2020. arXiv:2006.16236

### The core idea in plain language

The `N x M` matrix exists only because softmax has to normalise across it. Attention is
"for each query, take a weighted average of values, with weights given by a similarity
to each key" — and if the similarity can be written as a dot product of *features*, the
sum over keys can be done **once, first, for all queries**, instead of once per query.

Matrix products are associative. Softmax is not. That is the entire paper.

### The mathematics

Generalise attention to any non-negative similarity `sim(q, k)`:

```
O_i = sum_j sim(Q_i, K_j) V_j / sum_j sim(Q_i, K_j)
```

Softmax attention is `sim(q,k) = exp(q.k / sqrt(d))`. Choose instead a feature map
`phi` with `sim(q,k) = phi(q) . phi(k)`, and the sums factorise:

```
O_i = phi(Q_i)^T ( sum_j phi(K_j) V_j^T )  /  phi(Q_i)^T ( sum_j phi(K_j) )
                   \______ (m, dv) ______/                \____ (m,) ____/
```

Both bracketed quantities are independent of `i`: compute them once in `O(M m dv)`, then
each output is two small matrix-vector products. The paper uses `phi(x) = elu(x) + 1`.
The requirement is **positivity**, and it is not cosmetic: it keeps the denominator from
vanishing or changing sign, and keeps the implied weights a genuine convex combination.

**Causal masking becomes a prefix sum.** The state at position `i` is just the sum
restricted to `j <= i`:

```
S_i = sum_{j<=i} phi(K_j) V_j^T     (m, dv)
z_i = sum_{j<=i} phi(K_j)           (m,)
O_i = phi(Q_i)^T S_i / (phi(Q_i) . z_i)
```

No mask is built and nothing is computed and then discarded — compare with standard
attention, which computes the full `N x N` matrix and throws half of it away.

**And that recurrence is an RNN**, which is the title:

```
s <- s + phi(k) v^T
z <- z + phi(k)
o  = phi(q)^T s / (phi(q) . z)
```

### Complexity

```                     time                      memory at inference
softmax attention       O(N^2 d)                  KV cache: O(N d), grows with every token
linear attention        O(N m dv)                 state: O(m dv), CONSTANT
generation of L tokens  O(L^2 d)  ->  O(L m dv)
```

At inference this is the difference between "cost per token grows with the conversation"
and "cost per token is flat forever". Peak intermediate memory in this directory's
measurements is 8 KB at every sequence length, against 8 MB at `N = 1024` for softmax
attention — and 8 GB at `N = 65536`.

**Training** needs care: the obvious vectorised implementation (`cumsum` over all `N`
outer products) materialises an `(N, m, dv)` tensor — 134 MB per head at `N = 4096`,
`m = dv = 64`, *worse* than the `N^2` matrix it replaced. The gradients must be written
as a forward scan and a reverse scan carrying `(m, dv+1)` state:

```
dphi(Q_i) = S'_i dN'_i          forward scan     (S'_i = sum_{j<=i} phi(K_j) V'^T_j)
dphi(K_j) = G'_j V'_j           reverse scan     (G'_j = sum_{i>=j} phi(Q_i) dN'^T_i)
dV_j      = (G'_j)^T phi(K_j)   reverse scan
```

with `V' = [V | 1]` — appending a column of ones makes the denominator one more channel
of the numerator, so there is a single rule to differentiate rather than two. Query `i`
sees keys up to `i`; key `j` is seen by queries from `j` on: that asymmetry is why one
scan runs forwards and the other backwards. This is why the method ships as a custom
CUDA kernel: autograd would store all `N` intermediate states and lose the whole point.

### Approximation guarantee

**There isn't one, and that is the honest framing.** Linear attention is not an
approximation of softmax attention. It is a different attention with a different
similarity function. There is no `epsilon` and no bound relating the two outputs, and
the relative difference on random inputs is ~0.7 — a meaningless number, because the
question "how close is it to softmax attention" is the wrong question. The right one is
"does a model trained with it reach the same loss", and the answer is: close on many
tasks, measurably worse on recall-heavy ones.

The structural reason is **rank**. `sum_j phi(K_j) V_j^T` has rank at most `m`, so the
entire past is compressed into an `m x dv` summary however long the sequence is. Softmax
attention can retrieve one specific token out of a million; this cannot. And because the
state is a plain sum of outer products, nothing is ever forgotten — old associations are
superimposed on new ones until the state saturates. Every later linear RNN (gating,
decay, delta rules, and the modern state-space models) is an answer to that one sentence.

If you specifically want an *approximation of softmax attention* with linear cost, that
is Performer's positive random features, a different construction with a different
(variance-based) guarantee and a different cost.

### Compared with standard attention

Cheaper asymptotically in both time and memory, and uniquely good at autoregressive
decoding. Not exact, not a drop-in replacement for a trained softmax model's weights,
and structurally limited in retrieval.

---

## 3. FlashAttention

> Dao, Fu, Ermon, Rudra, Ré. *FlashAttention: Fast and Memory-Efficient Exact Attention
> with IO-Awareness.* NeurIPS 2022. arXiv:2205.14135
> Dao. *FlashAttention-2: Faster Attention with Better Parallelism and Work
> Partitioning.* 2023. arXiv:2307.08691

### The core idea in plain language

The previous two methods make attention cheaper by changing what it computes.
FlashAttention changes **nothing** about what is computed — the output is exact — and
instead observes that the kernel is memory bound. So: never write the `N x M` matrix to
HBM at all. Compute it one tile at a time in SRAM, consume the tile immediately, throw
it away.

The obstacle is softmax. It needs the row maximum and the row sum before it can
normalise anything, and both are only known after the whole row has been seen. Tiling
looks impossible. The resolution is the **online softmax**: keep a provisional maximum
and correct the accumulation when it changes.

### The mathematics: online softmax

State per row: running max `m`, running sum `l`, running unnormalised output `acc`.
On seeing a new block of scores `s` with values `v`:

```
m_new = max(m, max(s))
c     = exp(m - m_new)                    one scalar per row
l_new = c * l + sum(exp(s - m_new))
acc   = c * acc + exp(s - m_new) V_block
```

and at the very end `O = acc / l`. The correction `c` is exact — an exponential sum
written relative to one offset is rewritten relative to another by one multiply — so the
result is identical to computing the row in one go, and every exponent is `<= 0`
throughout, so nothing overflows. Two properties follow that are worth noticing because
step 17 cashes them in: the result does not depend on **block size** or on the **order**
blocks arrive in. Attention is associative over key ranges.

### The algorithm

```
B_c = ceil(M_sram / 4d),  B_r = min(B_c, d)         block sizes DERIVED from SRAM

for each K/V block j:                    <- FlashAttention-1 loop order
    load K_j, V_j to SRAM
    for each Q block i:
        load Q_i, O_i, l_i, m_i
        S_ij = Q_i K_j^T / sqrt(d)       in SRAM, never written out
        update (m_i, l_i, O_i) by the online rule
        store O_i, l_i, m_i
store L = m + log(l)                     one number per query
```

`L` is the only extra thing kept, and it is `O(N)`.

**Backward by recomputation.** Standard backward reads the saved `N x N` matrix.
FlashAttention rebuilds each tile exactly, from `L`:

```
P_ij = exp(Q_i K_j^T / sqrt(d) - L_i)
```

and uses the identity `sum_k P_ik dP_ik = sum_k dO_ik O_ik` (proved in step 2) so that
the row-coupling term is a length-`N` vector computed from `dO` and `O`, not a second
`N x N` intermediate. Trading arithmetic for memory is a *good* trade on a memory-bound
kernel: the recomputed tile never leaves SRAM, while the stored one would have cost two
`N^2` round trips.

### Complexity

```
                 flops          HBM accesses         extra memory
standard         O(N^2 d)       Theta(N^2 + N d)     O(N^2)
FlashAttention   O(N^2 d)       Theta(N^2 d^2 / M)   O(N)
```

Same flops. Same quadratic in `N` for traffic — the improvement is the factor `d^2/M`,
where `M` is SRAM capacity in elements. With `d = 64` and 100 KB of SRAM that is a
several-fold reduction, and the paper proves it is optimal up to constants: no exact
attention algorithm can do asymptotically better.

The memory column is the one that changed practice. `O(N)` instead of `O(N^2)` is what
made long-context training possible at all, independently of speed.

Measured in this directory (`d = 32`, 64 KB SRAM budget, byte counts from `io_model.py`):

```
    N     standard      flash-1      flash-2    ratio
  128     965.0 KB     368.0 KB     321.0 KB    2.62x
  256       3.6 MB       1.1 MB       1.0 MB    3.28x
  512      14.3 MB       4.2 MB       3.5 MB    3.42x

largest single HBM tensor at N=256:  65536 elements (standard, = N^2)
                                      8192 elements (flash,    = N*d)
```

**Causality is a bonus here.** A tile of keys entirely in the future of a tile of queries
is fully masked, so it is skipped outright: 24 blocks computed and 16 skipped at
`N = 256` in this directory. Standard attention computes the whole matrix and discards
half of it.

### Exactness

Bit-for-bit identical in exact arithmetic; in float64 our implementations agree with
plain numpy attention to ~4e-16, which is float reassociation, not approximation. This
is worth stating loudly because it is what separates FlashAttention from every other
entry in the "efficient attention" literature: **there is no accuracy question to
evaluate.** You do not have to check whether your task tolerates it.

### FlashAttention-2: the same algorithm, arranged better

FA-1 was already IO-optimal to within constants and still reached only 25-40% of peak
matmul throughput. FA-2 changes three things about the *arrangement*:

1. **Fewer non-matmul operations.** Tensor cores do matmul roughly 16x faster than the
   ALUs do anything else, so a rescale of the accumulator costs far more than its flop
   count suggests. FA-1 keeps `O_i` normalised in HBM, so each update un-normalises,
   rescales and re-normalises. FA-2 keeps the accumulator unnormalised and divides by
   `l` **once**, at the end. Measured here: 122,880 elementwise accumulator ops down to
   65,536, at `N = 256`.

2. **Parallelism over the sequence.** FA-1's outer loop is over K/V blocks, so every row
   block shares the same `O` accumulator and the outer loop cannot be parallelised; FA-1
   gets its parallelism from `batch x heads`, which is not enough blocks to fill a GPU
   when the batch is small and the sequence is long. FA-2 swaps the loops: outer over Q
   blocks, and **every iteration is independent**. One long sequence at batch size 1
   fills the machine. Step 16 verifies the independence directly by computing row blocks
   separately, in reverse order, and getting identical numbers.

   Splitting the *key* axis (step 17) is the same idea taken further, and it is what
   makes decoding fast: at generation time `N = 1`, there is only one row block, and
   without a key split the whole GPU sits idle behind one SM. Each split returns
   `(O_p, m_p, l_p)` and they merge exactly:

   ```
   m = max_p m_p;   w_p = exp(m_p - m) l_p;   O = sum_p w_p O_p / sum_p w_p
   ```

   This merge is the reusable idea of the whole directory: it is flash-decoding, it is
   ring attention across devices, it is chunked/paged KV caches. All of them are
   "compute attention over a slice, keep `(m, l)`, combine later".

3. **Better work partitioning inside a block.** FA-1 splits the K/V block across warps
   and shares `Q`, so all four warps must write partial results to shared memory and
   synchronise to combine them. FA-2 splits `Q` across warps and shares K/V, so each warp
   owns a slice of the output and needs no cross-warp reduction at all. This is below
   the level of abstraction modelled here — `io_model.py` has no warps — and it is why
   step 18's two-pass backward is presented as the *representative* of this idea rather
   than the thing itself: it removes the read-modify-write on `dQ` (an `atomicAdd` on
   real hardware) by running one pass with the loops in each order. Measured here: `dQ`
   written 144 times under FA-1, 12 times under FA-2.

Note what FA-2 does **not** do: it barely changes HBM traffic (1.1 MB to 1.0 MB in our
measurement). Its ~2x speedup on real hardware comes from the ALU work and the
occupancy, both of which are invisible in a byte count. If you only ever look at the
metric the previous paper optimised, you will conclude that FA-2 does nothing.

---

## 4. Choosing between them

```
Do you need the exact softmax attention function?
├── yes -> FlashAttention. There is no trade-off to evaluate; it is strictly better
│          than the naive implementation, and it is what PyTorch's SDPA now does.
│          If context length is still the binding constraint, you are asking a
│          different question -- go to "no".
└── no  -> is the model autoregressive?
          ├── yes -> linear attention (or a modern gated variant). Constant state per
          │          token is a property nothing else on this list has. Verify on a
          │          recall-heavy eval, because that is where its rank limit shows.
          └── no  -> clustered attention, if your queries actually cluster. Measure
                     that first: on data that does not cluster it is worthless, and
                     the measurement costs one afternoon rather than one training run.
```

Three questions worth being able to answer after implementing all of this:

1. Why can FlashAttention be exact when the other two cannot? *Because it changes the
   schedule, not the function. The other two change the function.*
2. Why does clustered attention have a bound and linear attention does not? *Clustered
   attention approximates the same quantity, so "how far off" is meaningful. Linear
   attention computes a different quantity, so it is not.*
3. Why is FlashAttention still `O(N^2)` and still worth it? *Because the constant, the
   memory footprint, and the arithmetic intensity all improve by large factors, and
   because exactness has a value that no asymptotic captures.*

And one that has no clean answer, which is where the field is: the three methods compose
awkwardly. FlashAttention's tiling assumes dense blocks; clustered attention's gather is
irregular; linear attention has no score matrix to tile. Combining sparsity with IO
awareness is what the block-sparse work, and everything after it, is about.
