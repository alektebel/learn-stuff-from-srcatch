-- Limits and Continuity in Lean - Template
-- This file guides you through proving theorems about limits, sequences, and continuity
-- Essential foundation for real analysis and calculus

/-
LEARNING OBJECTIVES:
1. Understand epsilon-delta definition of limits
2. Prove properties of convergent sequences
3. Master continuity proofs
4. Work with limit theorems
5. Understand squeeze theorem and algebraic limit laws
6. Develop intuition for epsilon-delta arguments needed in analysis
-/

-- ============================================================================
-- PART 1: SEQUENCES AND LIMITS
-- ============================================================================

namespace Limits

-- Definition: A sequence converges to a limit
-- A sequence (aₙ) converges to L if for all ε > 0, there exists N such that
-- for all n ≥ N, |aₙ - L| < ε
def sequence_converges (a : ℕ → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ N, ∀ n ≥ N, |a n - L| < ε

notation a " ⟶ " L => sequence_converges a L

-- TODO 1.1: Prove that constant sequences converge to the constant
-- If aₙ = c for all n, then aₙ → c
theorem constant_sequence_converges (c : ℝ) : 
  (fun _ => c) ⟶ c :=
  sorry  -- TODO: Show |c - c| = 0 < ε for any ε > 0

-- TODO 1.2: Prove uniqueness of limits
-- If a sequence converges, its limit is unique
theorem limit_unique {a : ℕ → ℝ} {L₁ L₂ : ℝ} 
  (h1 : a ⟶ L₁) (h2 : a ⟶ L₂) : L₁ = L₂ :=
  sorry  -- TODO: Use contradiction - assume L₁ ≠ L₂, choose ε = |L₁ - L₂|/2

-- TODO 1.3: Prove convergent sequences are bounded
theorem convergent_implies_bounded {a : ℕ → ℝ} {L : ℝ} 
  (h : a ⟶ L) : ∃ M, ∀ n, |a n| ≤ M :=
  sorry  -- TODO: Choose ε = 1, find N, then M = max(|a 0|, ..., |a (N-1)|, |L| + 1)

-- ============================================================================
-- PART 2: ALGEBRAIC LIMIT LAWS
-- ============================================================================

-- TODO 2.1: Sum of limits equals limit of sums
-- If aₙ → L and bₙ → M, then (aₙ + bₙ) → L + M
theorem limit_add {a b : ℕ → ℝ} {L M : ℝ}
  (ha : a ⟶ L) (hb : b ⟶ M) : 
  (fun n => a n + b n) ⟶ (L + M) :=
  sorry  -- TODO: Given ε > 0, choose ε/2 for each sequence

-- TODO 2.2: Scalar multiplication of limits
-- If aₙ → L, then (c · aₙ) → c · L
theorem limit_scalar_mul (c : ℝ) {a : ℕ → ℝ} {L : ℝ}
  (ha : a ⟶ L) :
  (fun n => c * a n) ⟶ (c * L) :=
  sorry  -- TODO: Handle c = 0 separately, otherwise use ε/|c|

-- TODO 2.3: Product of limits equals limit of products
-- If aₙ → L and bₙ → M, then (aₙ · bₙ) → L · M
theorem limit_mul {a b : ℕ → ℝ} {L M : ℝ}
  (ha : a ⟶ L) (hb : b ⟶ M) :
  (fun n => a n * b n) ⟶ (L * M) :=
  sorry  -- TODO: Use |aₙbₙ - LM| = |aₙbₙ - aₙM + aₙM - LM| ≤ |aₙ||bₙ - M| + |M||aₙ - L|

-- ============================================================================
-- PART 3: SQUEEZE THEOREM (SANDWICH THEOREM)
-- ============================================================================

-- TODO 3.1: Prove the Squeeze Theorem
-- If aₙ ≤ bₙ ≤ cₙ for all n, and aₙ → L and cₙ → L, then bₙ → L
theorem squeeze_theorem {a b c : ℕ → ℝ} {L : ℝ}
  (hab : ∀ n, a n ≤ b n) (hbc : ∀ n, b n ≤ c n)
  (ha : a ⟶ L) (hc : c ⟶ L) :
  b ⟶ L :=
  sorry  -- TODO: Given ε > 0, find N₁, N₂ for a and c, take N = max(N₁, N₂)

-- ============================================================================
-- PART 4: SUBSEQUENCES
-- ============================================================================

-- Definition: A subsequence is obtained by selecting elements at increasing indices
def is_subsequence (a : ℕ → ℝ) (b : ℕ → ℝ) : Prop :=
  ∃ (n : ℕ → ℕ), (∀ k, n k < n (k + 1)) ∧ (∀ k, b k = a (n k))

-- TODO 4.1: Prove that subsequences of convergent sequences converge to the same limit
theorem subsequence_converges {a b : ℕ → ℝ} {L : ℝ}
  (hsub : is_subsequence a b) (ha : a ⟶ L) :
  b ⟶ L :=
  sorry  -- TODO: Use the fact that indices are increasing, so eventually pass N

-- ============================================================================
-- PART 5: EPSILON-DELTA LIMITS OF FUNCTIONS
-- ============================================================================

-- Definition: Limit of a function at a point
-- lim_{x → a} f(x) = L means: for all ε > 0, exists δ > 0 such that
-- 0 < |x - a| < δ implies |f(x) - L| < ε
def function_limit (f : ℝ → ℝ) (a L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, 0 < |x - a| ∧ |x - a| < δ → |f x - L| < ε

notation "lim[" a "] " f " = " L => function_limit f a L

-- TODO 5.1: Prove that the limit of identity function is the point itself
-- lim_{x → a} x = a
theorem limit_identity (a : ℝ) :
  lim[a] (fun x => x) = a :=
  sorry  -- TODO: Choose δ = ε

-- TODO 5.2: Prove that the limit of constant function is the constant
-- lim_{x → a} c = c
theorem limit_constant (a c : ℝ) :
  lim[a] (fun _ => c) = c :=
  sorry  -- TODO: |c - c| = 0 < ε for any δ > 0

-- TODO 5.3: Sum rule for limits
-- If lim_{x → a} f(x) = L and lim_{x → a} g(x) = M, then lim_{x → a} (f + g)(x) = L + M
theorem function_limit_add (f g : ℝ → ℝ) (a L M : ℝ)
  (hf : lim[a] f = L) (hg : lim[a] g = M) :
  lim[a] (fun x => f x + g x) = (L + M) :=
  sorry  -- TODO: Choose δ to work for both f and g with ε/2

-- TODO 5.4: Product rule for limits
-- If lim_{x → a} f(x) = L and lim_{x → a} g(x) = M, then lim_{x → a} (f · g)(x) = L · M
theorem function_limit_mul (f g : ℝ → ℝ) (a L M : ℝ)
  (hf : lim[a] f = L) (hg : lim[a] g = M) :
  lim[a] (fun x => f x * g x) = (L * M) :=
  sorry  -- TODO: Similar to sequence case, bound |f| and use triangle inequality

-- ============================================================================
-- PART 6: CONTINUITY
-- ============================================================================

-- Definition: A function is continuous at a point
-- f is continuous at a if lim_{x → a} f(x) = f(a)
def continuous_at (f : ℝ → ℝ) (a : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ x, |x - a| < δ → |f x - f a| < ε

-- TODO 6.1: Prove that identity function is continuous
theorem identity_continuous (a : ℝ) :
  continuous_at (fun x => x) a :=
  sorry  -- TODO: Choose δ = ε

-- TODO 6.2: Prove that constant functions are continuous
theorem constant_continuous (c a : ℝ) :
  continuous_at (fun _ => c) a :=
  sorry  -- TODO: Any δ works since |c - c| = 0

-- TODO 6.3: Prove sum of continuous functions is continuous
theorem continuous_add (f g : ℝ → ℝ) (a : ℝ)
  (hf : continuous_at f a) (hg : continuous_at g a) :
  continuous_at (fun x => f x + g x) a :=
  sorry  -- TODO: Use ε/2 for each function

-- TODO 6.4: Prove product of continuous functions is continuous
theorem continuous_mul (f g : ℝ → ℝ) (a : ℝ)
  (hf : continuous_at f a) (hg : continuous_at g a) :
  continuous_at (fun x => f x * g x) a :=
  sorry  -- TODO: Use boundedness near a and triangle inequality

-- TODO 6.5: Prove composition of continuous functions is continuous
theorem continuous_comp (f g : ℝ → ℝ) (a : ℝ)
  (hf : continuous_at f a) (hg : continuous_at g (f a)) :
  continuous_at (fun x => g (f x)) a :=
  sorry  -- TODO: Chain the δ's: given ε, get δ₁ for g, then δ₂ for f

-- ============================================================================
-- PART 7: SPECIAL LIMITS
-- ============================================================================

-- TODO 7.1: Prove that 1/n → 0
theorem limit_one_over_n :
  (fun n => (1 : ℝ) / (n + 1)) ⟶ 0 :=
  sorry  -- TODO: Given ε > 0, choose N > 1/ε (use Archimedean property)

-- TODO 7.2: Prove that if |r| < 1, then rⁿ → 0
theorem limit_geometric_lt_one (r : ℝ) (hr : |r| < 1) :
  (fun n => r ^ n) ⟶ 0 :=
  sorry  -- TODO: Use |rⁿ| = |r|ⁿ and solve |r|ⁿ < ε

-- TODO 7.3: Prove monotone convergence theorem (bounded monotone sequences converge)
-- This requires the completeness of reals (least upper bound property)
axiom monotone_convergence : 
  ∀ (a : ℕ → ℝ), 
  (∀ n, a n ≤ a (n + 1)) →  -- monotone increasing
  (∃ M, ∀ n, a n ≤ M) →      -- bounded above
  ∃ L, a ⟶ L

end Limits

/-
IMPLEMENTATION GUIDE:

Phase 1: Basic Sequences (1-2 days)
- Start with constant_sequence_converges
- Prove limit_unique using proof by contradiction
- Practice epsilon-delta arguments

Phase 2: Algebraic Laws (2-3 days)
- Implement limit_add, limit_scalar_mul, limit_mul
- Key technique: split epsilon appropriately
- Practice using triangle inequality

Phase 3: Squeeze Theorem (1 day)
- Classic theorem with beautiful proof
- Great practice combining multiple limit arguments

Phase 4: Function Limits (2-3 days)
- Similar to sequences but with delta
- Prove basic limits (identity, constant)
- Implement sum and product rules

Phase 5: Continuity (2-3 days)
- Connect limits and continuity
- Prove operations preserve continuity
- Composition is most challenging

Phase 6: Special Limits (2-3 days)
- Concrete examples: 1/n, geometric series
- Use Archimedean property of reals
- Requires understanding of real number completeness

Common Tactics:
- intro/intros : Introduce hypotheses
- apply : Apply a theorem
- exact : Provide exact term
- obtain : Extract existential witnesses
- have : State intermediate results
- calc : Chain of equalities/inequalities

Key Techniques:
- Triangle inequality: |a + b| ≤ |a| + |b|
- Splitting epsilon: ε/2 for sum of two things
- Choosing N or δ appropriately
- Proof by contradiction for uniqueness

Learning Resources:
- Real Analysis textbooks (Rudin, Abbott, Tao)
- Mathematics in Lean: Analysis chapter
- Mathlib4: Topology.MetricSpace.Basic
- Natural Number Game for warm-up

Difficulty: ⭐⭐⭐ Medium-Hard
Estimated Time: 10-15 days
Prerequisites: BasicLogic.lean, SetTheory.lean, NaturalNumbers.lean
Next Steps: BanachSpaces.lean, PartialDerivatives.lean
-/
