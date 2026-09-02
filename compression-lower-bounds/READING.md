# Reading list

Curated, not exhaustive. Every item is here because a step in one of the two tracks
depends on it.

**Verification status is marked per entry.** `[v]` means the arXiv ID and title were
checked against arXiv while writing this file. Books and pre-arXiv papers are given as
full citations rather than links, because a guessed URL is worse than no URL.

---

## 1. The EXL3 lineage — read strictly in this order

| # | Paper | Status |
|---|---|---|
| 1 | LeCun, Denker, Solla, *Optimal Brain Damage*, NeurIPS 1989 | pre-arXiv |
| 2 | Hassibi & Stork, *Optimal Brain Surgeon*, NeurIPS 1992 | pre-arXiv |
| 3 | Nagel, Amjad, van Baalen, Louizos, Blankevoort, *Up or Down? Adaptive Rounding for Post-Training Quantization*, ICML 2020 — [2004.10568](https://arxiv.org/abs/2004.10568) | `[v]` |
| 4 | Frantar, Ashkboos, Hoefler, Alistarh, **GPTQ**, ICLR 2023 — [2210.17323](https://arxiv.org/abs/2210.17323) | `[v]` |
| 5 | Chee, Cai, Kuleshov, De Sa, **QuIP: 2-Bit Quantization of LLMs With Guarantees** — [2307.13304](https://arxiv.org/abs/2307.13304) | `[v]` |
| 6 | Tseng, Chee, Sun, Kuleshov, De Sa, **QuIP#: Even Better LLM Quantization with Hadamard Incoherence and Lattice Codebooks**, ICML 2024 — [2402.04396](https://arxiv.org/abs/2402.04396) | `[v]` |
| 7 | Tseng, Sun, Hou, De Sa, **QTIP: Quantization with Trellises and Incoherence Processing**, NeurIPS 2024 — [2406.11235](https://arxiv.org/abs/2406.11235) | `[v]` |
| 8 | **EXL3 format doc** — <https://github.com/turboderp-org/exllamav3/blob/master/doc/exl3.md> | url given by author, not fetched |
| 9 | **EXL3 quantizer source** — `exllamav3/modules/quant/exl3_lib/quantize.py` | " |
| 10 | **QTIP reference implementation** — <https://github.com/Cornell-RelaxML/qtip> | " |

The conceptual step is 5 → 6: incoherence processing via the random Hadamard transform.
**Step 4 of `compression/` is where you find out why the transform must be random** —
the deterministic Hadamard transform has an input on which it makes incoherence
maximally *worse*, and you will construct it.

Everything after step 6 is codebook engineering: QuIP# replaces the scalar codebook with
the E8 lattice (8 dimensions, 0.654 dB of space-filling gain), and QTIP replaces the
lattice with a trellis whose codebook is *computed from the state* rather than stored,
which decouples codebook size from bitrate.

## 2. Orthogonal branches

| Paper | Status |
|---|---|
| Lin et al., **AWQ** (activation-aware scaling) — [2306.00978](https://arxiv.org/abs/2306.00978) | url as given |
| Egiazarian et al., **AQLM** (additive / multi-codebook VQ) — [2401.06118](https://arxiv.org/abs/2401.06118) | url as given |
| Malinovskii et al., **PV-Tuning** (beyond straight-through) — [2405.14852](https://arxiv.org/abs/2405.14852) | `[v]` |
| Dettmers et al., **SpQR** (outlier isolation) — [2306.03078](https://arxiv.org/abs/2306.03078) | url as given |
| Xiao, Lin, Seznec, Wu, Demouth, Han, **SmoothQuant** — [2211.10438](https://arxiv.org/abs/2211.10438) | `[v]` |

## 3. Sparsity

| Paper | Status |
|---|---|
| Frantar & Alistarh, **SparseGPT** — [2301.00774](https://arxiv.org/abs/2301.00774) | url as given |
| Sun, Liu, Bair, Kolter, **Wanda** — [2306.11695](https://arxiv.org/abs/2306.11695) | `[v]` |
| Frankle & Carbin, **The Lottery Ticket Hypothesis** — [1803.03635](https://arxiv.org/abs/1803.03635) | `[v]` |

> The v1 arXiv title is *"The Lottery Ticket Hypothesis: Training Pruned Neural
> Networks"*; the ICLR 2019 title is *"…: Finding Sparse, Trainable Neural Networks"*.
> Both refer to the same ID.

## 4. The theory the whole field is a special case of

Sections 1–3 are applied rate–distortion theory with a language model as the source.
Read at least Gersho & Gray ch. 5–6 before `compression/` step 3, or you will re-derive
the space-filling/shape decomposition by accident and not recognise it.

- Cover & Thomas, *Elements of Information Theory*, 2nd ed., ch. 10 — rate–distortion.
- Gersho & Gray, *Vector Quantization and Signal Compression*, Springer 1992. Lloyd–Max,
  Zador's bound, and the decomposition of the quantization gap into space-filling,
  shape and memory gains.
- Zamir, *Lattice Coding for Signals and Networks*, CUP 2014. Why lattices, and why E8.
- Ungerboeck, "Channel coding with multilevel/phase signals", IEEE Trans. IT, 1982.
- Marcellin & Fischer, "Trellis coded quantization of memoryless and Gauss–Markov
  sources", IEEE Trans. Comm., 1990. The direct ancestor of QTIP.
- Zador, "Asymptotic quantization error of continuous signals", IEEE Trans. IT, 1982.
- Panter & Dite, "Quantization distortion in pulse-count modulation with nonuniform
  spacing of levels", Proc. IRE, 1951. Source of the constant `√3π/2` measured in step 2.

## 5. Empirical limits

- Kumar, Ankner, Spector, Bordelon, Muennighoff, Paul, Pehlevan, Ré, Raghunathan,
  *Scaling Laws for Precision*, ICLR 2025 — [2411.04330](https://arxiv.org/abs/2411.04330) `[v]`

Its sharpest claim, and the one worth checking against your own step-6 numbers: post-
training-quantization damage *grows* with the amount of pretraining data, so past some
token budget more pretraining makes the quantized model worse.

## 6. Computational lower bounds

Pre-arXiv or books. Full citations so you can find them yourself.

- Arora & Barak, *Computational Complexity: A Modern Approach*, CUP 2009.
- Bürgisser, Clausen, Shokrollahi, *Algebraic Complexity Theory*, Springer 1997. Tensor
  rank, border rank, ω. The source for track step 2.
- Strassen, "Gaussian elimination is not optimal", Numer. Math. 13 (1969). The rank-7
  decomposition itself.
- Strassen, "Vermeidung von Divisionen", J. reine angew. Math. 264 (1973). Degree bound.
- Baur & Strassen, "The complexity of partial derivatives", Theor. Comp. Sci. 22 (1983).
- Hong & Kung, "I/O complexity: the red-blue pebble game", STOC 1981. **The practically
  useful one** — it is the only bound in this list that predicts a number you can
  measure on hardware.
- Ballard, Demmel, Holtz, Schwartz, "Minimizing communication in numerical linear
  algebra", SIAM J. Matrix Anal. Appl. 32 (2011).
- Razborov & Rudich, "Natural proofs", JCSS 55 (1997).
- Williams, "Nonuniform ACC circuit lower bounds", JACM 2014.

## 7. Physical limits — read only after writing `lower-bounds/05_the_gap.md`

- Landauer, IBM J. Res. Dev. 5 (1961); Bennett, Int. J. Theor. Phys. 21 (1982);
  Margolus & Levitin, Physica D 120 (1998); Lloyd, Nature 406 (2000).

> **These bound nothing you can act on.** `kT ln 2` at room temperature is many orders of
> magnitude below the energy a real GPU spends per bit erased, so the Landauer limit
> never binds. They are in the list for one reason: they are the only limits in it that
> are *provably* tight and *provably* irrelevant, which is a useful calibration for how
> to read the rest. Do not start here.
