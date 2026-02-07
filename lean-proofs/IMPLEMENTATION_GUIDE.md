# Lean Proofs Implementation Guide

This guide provides verbose, step-by-step implementation guidelines for proving your way to Galois Theory in Lean 4. Each section includes theoretical background, proof strategies, common tactics, and detailed guidance without providing complete implementations.

## Table of Contents

1. [Getting Started with Lean 4](#getting-started-with-lean-4)
2. [BasicLogic.lean - Propositional Logic](#basiclogiclean---propositional-logic)
3. [SetTheory.lean - Sets and Relations](#settheorylean---sets-and-relations)
4. [NaturalNumbers.lean - Peano Axioms](#naturalnumberslean---peano-axioms)
5. [Groups.lean - Group Theory](#groupslean---group-theory)
6. [Rings.lean - Ring Theory](#ringslean---ring-theory)
7. [Fields.lean - Field Theory](#fieldslean---field-theory)
8. [Polynomials.lean - Polynomial Rings](#polynomialslean---polynomial-rings)
9. [FieldExtensions.lean - Field Extensions](#fieldextensionslean---field-extensions)
10. [SplittingFields.lean - Splitting Fields](#splittingfieldslean---splitting-fields)
11. [GaloisTheory.lean - The Fundamental Theorem](#galoistheorylean---the-fundamental-theorem)
12. [Common Lean Tactics Reference](#common-lean-tactics-reference)
13. [Debugging and Troubleshooting](#debugging-and-troubleshooting)

---

## Getting Started with Lean 4

### Installation

```bash
# Install elan (Lean version manager)
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Create a new Lean project
lake new galois_project
cd galois_project

# Add mathlib dependency in lakefile.lean
# Then run:
lake update
lake build
```

### VS Code Setup

1. Install VS Code
2. Install the "lean4" extension
3. Open your .lean files
4. Use the Lean Infoview panel (Ctrl+Shift+Enter) to see proof states

### Basic Syntax

```lean
-- Comments start with --
/- 
  Multi-line comments 
  go between /- and -/
-/

-- Defining a theorem
theorem my_theorem (P Q : Prop) : P → Q → P ∧ Q := by
  intro hp hq      -- Introduce hypotheses
  constructor      -- Use And.intro
  · exact hp       -- First goal
  · exact hq       -- Second goal

-- Alternative: direct proof term
theorem my_theorem' (P Q : Prop) : P → Q → P ∧ Q :=
  fun hp hq => ⟨hp, hq⟩
```

---

## BasicLogic.lean - Propositional Logic

### Theoretical Background

Propositional logic is the foundation of all mathematical reasoning. In Lean, propositions are types (`Prop`), and proofs are terms of those types. This is called the Curry-Howard correspondence.

**Key Concepts:**
- **Conjunction (∧)**: "P and Q" - both must be true
- **Disjunction (∨)**: "P or Q" - at least one must be true  
- **Implication (→)**: "if P then Q" - functions from proofs of P to proofs of Q
- **Negation (¬)**: "not P" - defined as `P → False`
- **Biconditional (↔)**: "P if and only if Q" - equivalence

### Implementation Strategy

#### Phase 1: Conjunction (AND)

**Theorems to Prove:**
1. `and_intro`: Given P and Q, construct P ∧ Q
2. `and_elim_left`: From P ∧ Q, extract P
3. `and_elim_right`: From P ∧ Q, extract Q
4. `and_comm`: Prove P ∧ Q ↔ Q ∧ P (commutativity)
5. `and_assoc`: Prove (P ∧ Q) ∧ R ↔ P ∧ (Q ∧ R) (associativity)

**Tactics to Use:**
- `constructor` or `And.intro`: Build a conjunction
- `.left` or `And.left`: Extract left component
- `.right` or `And.right`: Extract right component
- Anonymous constructor syntax: `⟨proof1, proof2⟩`

**Example Approach:**
```lean
-- To prove: P → Q → P ∧ Q
-- Strategy:
--   1. Introduce hypothesis hp : P
--   2. Introduce hypothesis hq : Q
--   3. Use And.intro (or ⟨hp, hq⟩) to construct the conjunction
-- 
-- Tactic mode:
--   intro hp hq
--   constructor
--   · exact hp
--   · exact hq
--
-- Term mode:
--   fun hp hq => ⟨hp, hq⟩
```

#### Phase 2: Disjunction (OR)

**Theorems to Prove:**
1. `or_intro_left`: From P, construct P ∨ Q
2. `or_intro_right`: From Q, construct P ∨ Q
3. `or_comm`: Prove P ∨ Q → Q ∨ P
4. `or_assoc`: Prove (P ∨ Q) ∨ R ↔ P ∨ (Q ∨ R)

**Tactics to Use:**
- `Or.inl`: Construct disjunction from left
- `Or.inr`: Construct disjunction from right
- `cases h with | inl hp => ... | inr hq => ...`: Case analysis on disjunction
- `match h with | Or.inl hp => ... | Or.inr hq => ...`: Pattern matching

**Example Approach:**
```lean
-- To prove: P ∨ Q → Q ∨ P
-- Strategy:
--   1. Introduce hypothesis h : P ∨ Q
--   2. Case analysis on h:
--      - Case h = Or.inl hp: We have P, construct Q ∨ P with Or.inr hp
--      - Case h = Or.inr hq: We have Q, construct Q ∨ P with Or.inl hq
--
-- Using cases tactic:
--   intro h
--   cases h with
--   | inl hp => exact Or.inr hp
--   | inr hq => exact Or.inl hq
--
-- Using match expression:
--   fun h => match h with
--     | Or.inl hp => Or.inr hp
--     | Or.inr hq => Or.inl hq
```

#### Phase 3: Implication

**Theorems to Prove:**
1. `imp_self`: Prove P → P (identity)
2. `imp_trans`: Prove (P → Q) → (Q → R) → (P → R) (transitivity)
3. `modus_ponens`: Prove (P → Q) → P → Q

**Tactics to Use:**
- `intro`: Introduce implication hypothesis (lambda abstraction)
- `apply`: Apply a function/implication
- Function application: `h1 h2` applies h1 to h2

**Example Approach:**
```lean
-- To prove: (P → Q) → (Q → R) → (P → R)
-- Strategy:
--   1. Introduce h1 : P → Q
--   2. Introduce h2 : Q → R
--   3. Need to show P → R, so introduce hp : P
--   4. Apply h1 to hp to get Q
--   5. Apply h2 to the result to get R
--
-- Tactic mode:
--   intro h1 h2 hp
--   apply h2
--   apply h1
--   exact hp
--
-- Term mode (function composition):
--   fun h1 h2 hp => h2 (h1 hp)
```

#### Phase 4: Negation

**Theorems to Prove:**
1. `not_not_intro`: Prove P → ¬¬P
2. `ex_falso`: From P and ¬P, prove anything (explosion principle)
3. `contrapositive`: Prove (P → Q) → (¬Q → ¬P)

**Key Insight:**
- Negation ¬P is defined as P → False
- False has no constructor, so you can't prove it (unless there's a contradiction)
- From False, you can prove anything using `False.elim`

**Tactics to Use:**
- `absurd`: Given P and ¬P, derive False
- `False.elim`: From False, prove anything
- Remember ¬P means P → False

**Example Approach:**
```lean
-- To prove: P → ¬¬P
-- Remember: ¬¬P means (P → False) → False
-- Strategy:
--   1. Introduce hp : P
--   2. Need to show (P → False) → False
--   3. Introduce hnp : P → False
--   4. Apply hnp to hp to get False
--
-- Tactic mode:
--   intro hp hnp
--   apply hnp
--   exact hp
--
-- Term mode:
--   fun hp hnp => hnp hp
```

#### Phase 5: Biconditional (IFF)

**Theorems to Prove:**
1. `iff_intro`: Prove (P → Q) → (Q → P) → (P ↔ Q)
2. `iff_elim_left`: From P ↔ Q, extract P → Q
3. `iff_elim_right`: From P ↔ Q, extract Q → P
4. `iff_refl`: Prove P ↔ P
5. `iff_trans`: Prove (P ↔ Q) → (Q ↔ R) → (P ↔ R)

**Tactics to Use:**
- `Iff.intro` or `⟨forward, backward⟩`: Construct biconditional
- `.mp` (modus ponens): Extract forward direction from ↔
- `.mpr` (modus ponens reverse): Extract backward direction from ↔
- `constructor`: Split ↔ into two goals

**Example Approach:**
```lean
-- To prove: P ∧ Q ↔ Q ∧ P
-- Strategy:
--   1. Need to prove both directions
--   2. Forward (→): Assume P ∧ Q, destruct to get P and Q, construct Q ∧ P
--   3. Backward (←): Assume Q ∧ P, destruct to get Q and P, construct P ∧ Q
--
-- Tactic mode:
--   constructor
--   · intro h
--     constructor
--     · exact h.right
--     · exact h.left
--   · intro h
--     constructor
--     · exact h.right
--     · exact h.left
--
-- Term mode with anonymous functions:
--   ⟨fun ⟨hp, hq⟩ => ⟨hq, hp⟩, fun ⟨hq, hp⟩ => ⟨hp, hq⟩⟩
```

#### Phase 6: Advanced Propositional Logic

**Theorems to Prove:**
1. De Morgan's Laws:
   - `de_morgan_and`: ¬(P ∧ Q) ↔ ¬P ∨ ¬Q (requires classical logic!)
   - `de_morgan_or`: ¬(P ∨ Q) ↔ ¬P ∧ ¬Q (constructive!)
2. Distributive Laws:
   - `and_or_distrib_left`: P ∧ (Q ∨ R) ↔ (P ∧ Q) ∨ (P ∧ R)
   - `or_and_distrib_left`: P ∨ (Q ∧ R) ↔ (P ∨ Q) ∧ (P ∨ R)

**Classical vs Constructive Logic:**
- Some theorems require classical logic (law of excluded middle)
- Use `open Classical` and `em P` (excluded middle) when needed
- Try to prove constructively first when possible

---

## SetTheory.lean - Sets and Relations

### Theoretical Background

In Lean, sets are predicates (functions to `Prop`). A set `S : Set α` is represented as `α → Prop`, where `x ∈ S` means `S x` is true.

**Key Concepts:**
- **Membership**: x ∈ S
- **Subset**: S ⊆ T means ∀ x, x ∈ S → x ∈ T
- **Operations**: ∪ (union), ∩ (intersection), \ (difference), ᶜ (complement)
- **Relations**: Binary relations are sets of pairs (or functions α → α → Prop)
- **Functions**: Can be injective, surjective, or bijective

### Implementation Strategy

#### Phase 1: Basic Set Operations

**Theorems to Prove:**
1. Subset reflexivity: `S ⊆ S`
2. Subset transitivity: `S ⊆ T → T ⊆ U → S ⊆ U`
3. Set extensionality: `(∀ x, x ∈ S ↔ x ∈ T) → S = T`
4. Empty set properties: `∀ x, x ∉ ∅`
5. Universal set properties: `∀ x, x ∈ univ`

**Tactics to Use:**
- `intro`: Introduce element and membership hypothesis
- `apply`: Apply subset hypothesis
- `ext`: Prove set equality by extensionality
- `simp`: Simplify set membership expressions

**Example Approach:**
```lean
-- To prove: S ⊆ S
-- Recall: S ⊆ T is defined as ∀ x, x ∈ S → x ∈ T
-- Strategy:
--   1. Unfold definition: need ∀ x, x ∈ S → x ∈ S
--   2. Introduce x and hypothesis h : x ∈ S
--   3. The goal is x ∈ S, which we have as h
--
-- Implementation:
--   intro x h
--   exact h
```

#### Phase 2: Union and Intersection

**Theorems to Prove:**
1. Union membership: `x ∈ S ∪ T ↔ x ∈ S ∨ x ∈ T`
2. Intersection membership: `x ∈ S ∩ T ↔ x ∈ S ∧ x ∈ T`
3. Union commutativity: `S ∪ T = T ∪ S`
4. Intersection commutativity: `S ∩ T = T ∩ S`
5. Union associativity: `(S ∪ T) ∪ U = S ∪ (T ∪ U)`
6. Intersection associativity: `(S ∩ T) ∩ U = S ∩ (T ∩ U)`
7. Distributive laws:
   - `S ∩ (T ∪ U) = (S ∩ T) ∪ (S ∩ U)`
   - `S ∪ (T ∩ U) = (S ∪ T) ∩ (S ∪ U)`

**Tactics to Use:**
- `ext`: Prove set equality
- `constructor`: Split ↔ into both directions
- `cases`: Case analysis on disjunctions
- `simp [membership_iff]`: Simplify membership

**Example Approach:**
```lean
-- To prove: S ∪ T = T ∪ S
-- Strategy:
--   1. Use extensionality: show ∀ x, x ∈ S ∪ T ↔ x ∈ T ∪ S
--   2. Unfold union: x ∈ S ∨ x ∈ T ↔ x ∈ T ∨ x ∈ S
--   3. This reduces to commutativity of ∨ (already proved!)
--
-- Implementation:
--   ext x
--   constructor
--   · intro h
--     cases h with
--     | inl hs => exact Or.inr hs
--     | inr ht => exact Or.inl ht
--   · intro h
--     cases h with
--     | inl ht => exact Or.inr ht
--     | inr hs => exact Or.inl hs
```

#### Phase 3: Relations

**Theorems to Prove:**
1. Reflexive relation properties
2. Symmetric relation properties
3. Transitive relation properties
4. Equivalence relation properties (reflexive + symmetric + transitive)
5. Equivalence classes partition a set
6. Composition of relations

**Key Definitions:**
```lean
-- A relation R is reflexive if ∀ x, R x x
def Reflexive (R : α → α → Prop) : Prop := ∀ x, R x x

-- A relation R is symmetric if ∀ x y, R x y → R y x
def Symmetric (R : α → α → Prop) : Prop := ∀ x y, R x y → R y x

-- A relation R is transitive if ∀ x y z, R x y → R y z → R x z
def Transitive (R : α → α → Prop) : Prop := ∀ x y z, R x y → R y z → R x z

-- A relation is an equivalence relation if it's reflexive, symmetric, and transitive
```

**Example Approach:**
```lean
-- To prove: Equivalence relation properties
-- Strategy:
--   1. Define what it means to be an equivalence relation
--   2. Prove that equality is an equivalence relation
--   3. Prove that intersection of equivalence relations is an equivalence relation
--   4. Define equivalence classes
--   5. Prove that equivalence classes partition the set
--
-- For equivalence classes:
--   - Define: [x] = {y | R x y}
--   - Prove: x ∈ [x] (using reflexivity)
--   - Prove: [x] = [y] ↔ R x y (key theorem)
--   - Prove: [x] ∩ [y] = ∅ ∨ [x] = [y] (partition property)
```

#### Phase 4: Functions

**Theorems to Prove:**
1. Function composition properties
2. Identity function properties
3. Injective function properties:
   - `f a = f b → a = b`
   - Composition of injections is injective
4. Surjective function properties:
   - `∀ b, ∃ a, f a = b`
   - Composition of surjections is surjective
5. Bijective function properties:
   - Bijection has an inverse
   - Composition of bijections is bijective

**Example Approach:**
```lean
-- To prove: Composition of injective functions is injective
-- Given: f : α → β, g : β → γ
-- Given: f is injective, g is injective
-- To show: g ∘ f is injective
--
-- Strategy:
--   1. Unfold injective: (g ∘ f) a = (g ∘ f) b → a = b
--   2. Unfold composition: g (f a) = g (f b) → a = b
--   3. Use injectivity of g: g (f a) = g (f b) → f a = f b
--   4. Use injectivity of f: f a = f b → a = b
--   5. Chain these together
--
-- Implementation uses functional composition and injectivity hypotheses
```

---

## NaturalNumbers.lean - Peano Axioms

### Theoretical Background

Natural numbers can be defined inductively using the Peano axioms. In Lean, this is done with an inductive type.

**Peano Axioms:**
1. 0 is a natural number
2. Every natural number n has a successor S(n)
3. 0 is not the successor of any natural number
4. Different natural numbers have different successors (S is injective)
5. **Induction**: If P(0) and ∀n, P(n) → P(S(n)), then ∀n, P(n)

### Implementation Strategy

#### Phase 1: Define Natural Numbers

```lean
-- Inductive definition (already in Lean as Nat)
inductive MyNat : Type
  | zero : MyNat
  | succ : MyNat → MyNat

-- Notation
notation "0" => MyNat.zero
notation "S" => MyNat.succ
```

#### Phase 2: Define Addition

**Recursive Definition:**
```lean
-- Addition is defined recursively
-- 0 + n = n
-- (S m) + n = S (m + n)

def add : MyNat → MyNat → MyNat
  | 0, n => n
  | S m, n => S (add m n)
```

**Theorems to Prove:**
1. `add_zero`: `n + 0 = n` (requires induction!)
2. `add_succ`: `n + S m = S (n + m)`
3. `add_comm`: `n + m = m + n` (commutativity)
4. `add_assoc`: `(n + m) + p = n + (m + p)` (associativity)

**Example Approach:**
```lean
-- To prove: n + 0 = n
-- Strategy: Induction on n
-- Base case: 0 + 0 = 0
--   By definition of add, this is true
-- Inductive step: Assume n + 0 = n, prove S n + 0 = S n
--   S n + 0 = S (n + 0)    by definition of add
--           = S n          by inductive hypothesis
--
-- Tactic mode:
--   intro n
--   induction n with
--   | zero => rfl
--   | succ n ih =>
--     simp [add]
--     exact ih
```

#### Phase 3: Define Multiplication

**Recursive Definition:**
```lean
-- Multiplication is defined recursively
-- 0 * n = 0
-- (S m) * n = n + (m * n)

def mul : MyNat → MyNat → MyNat
  | 0, n => 0
  | S m, n => add n (mul m n)
```

**Theorems to Prove:**
1. `mul_zero`: `n * 0 = 0`
2. `mul_one`: `n * 1 = n`
3. `mul_comm`: `n * m = m * n`
4. `mul_assoc`: `(n * m) * p = n * (m * p)`
5. `left_distrib`: `n * (m + p) = n * m + n * p`
6. `right_distrib`: `(n + m) * p = n * p + m * p`

**Each proof requires induction and uses previously proved lemmas.**

#### Phase 4: Order Relations

**Theorems to Prove:**
1. Define ≤ relation
2. Prove ≤ is reflexive
3. Prove ≤ is transitive
4. Prove ≤ is antisymmetric
5. Prove trichotomy: `∀ n m, n < m ∨ n = m ∨ m < n`
6. Prove well-ordering: every non-empty set has a minimum

**Induction Principle:**
```lean
-- Mathematical induction principle
axiom induction :
  ∀ (P : MyNat → Prop),
    P 0 →
    (∀ n, P n → P (S n)) →
    (∀ n, P n)

-- Strong induction (also provable)
axiom strong_induction :
  ∀ (P : MyNat → Prop),
    (∀ n, (∀ m, m < n → P m) → P n) →
    (∀ n, P n)
```

---

## Groups.lean - Group Theory

### Theoretical Background

A group is a set G with a binary operation · that satisfies:
1. **Closure**: ∀ a b ∈ G, a · b ∈ G
2. **Associativity**: ∀ a b c ∈ G, (a · b) · c = a · (b · c)
3. **Identity**: ∃ e ∈ G, ∀ a ∈ G, e · a = a · e = a
4. **Inverses**: ∀ a ∈ G, ∃ a⁻¹ ∈ G, a · a⁻¹ = a⁻¹ · a = e

### Implementation Strategy

#### Phase 1: Define Group Structure

```lean
-- Group class definition
class Group (G : Type) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  mul_left_inv : ∀ a, mul (inv a) a = one

-- Notation
instance : Mul G := ⟨Group.mul⟩
instance : One G := ⟨Group.one⟩
notation a "⁻¹" => Group.inv a
```

#### Phase 2: Basic Group Properties

**Theorems to Prove:**
1. `mul_right_inv`: `a · a⁻¹ = e` (from left inverse)
2. `inv_mul_cancel_left`: `a⁻¹ · (a · b) = b`
3. `mul_inv_cancel_left`: `a · (a⁻¹ · b) = b`
4. `inv_inv`: `(a⁻¹)⁻¹ = a`
5. `one_unique`: If `e' · a = a` for all a, then `e' = e`
6. `inv_unique`: If `b · a = e`, then `b = a⁻¹`
7. `mul_eq_one_iff`: `a · b = e ↔ a = b⁻¹`

**Example Approach:**
```lean
-- To prove: a · a⁻¹ = e
-- Given: a⁻¹ · a = e (mul_left_inv axiom)
-- Strategy:
--   1. Multiply both sides on the left by (a⁻¹)⁻¹
--   2. Use associativity and inverse property
--   3. Simplify to get the result
--
-- Detailed steps:
--   a · a⁻¹ = a · a⁻¹                         (identity)
--         = e · (a · a⁻¹)                     (left identity)
--         = ((a⁻¹)⁻¹ · a⁻¹) · (a · a⁻¹)       (inverse property)
--         = (a⁻¹)⁻¹ · (a⁻¹ · (a · a⁻¹))       (associativity)
--         = (a⁻¹)⁻¹ · ((a⁻¹ · a) · a⁻¹)       (associativity)
--         = (a⁻¹)⁻¹ · (e · a⁻¹)               (inverse property)
--         = (a⁻¹)⁻¹ · a⁻¹                     (left identity)
--         = e                                 (inverse property)
```

#### Phase 3: Subgroups

**Definition:**
A subset H ⊆ G is a subgroup if:
1. H is non-empty (or: e ∈ H)
2. H is closed under multiplication
3. H is closed under inverses

**Theorems to Prove:**
1. Subgroup criterion: H is a subgroup ↔ (e ∈ H ∧ ∀ a b ∈ H, a · b⁻¹ ∈ H)
2. Intersection of subgroups is a subgroup
3. Trivial subgroup {e} is a subgroup
4. Whole group G is a subgroup

**Example Approach:**
```lean
-- To prove: Subgroup criterion
-- Forward direction: If H is a subgroup, then e ∈ H and a · b⁻¹ ∈ H
--   1. H non-empty, so ∃ a ∈ H
--   2. H closed under inverses, so a⁻¹ ∈ H
--   3. H closed under multiplication, so a · a⁻¹ = e ∈ H
--   4. For a, b ∈ H: b⁻¹ ∈ H by closure under inverses
--   5. Then a · b⁻¹ ∈ H by closure under multiplication
--
-- Backward direction: If e ∈ H and a · b⁻¹ ∈ H, then H is a subgroup
--   1. Non-empty: e ∈ H given
--   2. Closed under inverses: For a ∈ H, need a⁻¹ ∈ H
--      Use e · a⁻¹ = a⁻¹ and criterion with a=e, b=a
--   3. Closed under multiplication: For a, b ∈ H, need a · b ∈ H
--      Note: a · b = a · (b⁻¹)⁻¹
--      First get b⁻¹ ∈ H, then a · (b⁻¹)⁻¹ ∈ H by criterion
```

#### Phase 4: Cosets and Lagrange's Theorem

**Definitions:**
- Left coset: `a H = {a · h | h ∈ H}`
- Right coset: `H a = {h · a | h ∈ H}`
- Index: [G : H] = number of left cosets

**Theorems to Prove:**
1. Cosets partition the group
2. All cosets have the same size as H
3. **Lagrange's Theorem**: |G| = [G : H] · |H|
4. Corollary: Order of element divides order of group

**Example Approach (Lagrange):**
```lean
-- Lagrange's Theorem outline:
-- Strategy:
--   1. Define left cosets gH for all g ∈ G
--   2. Prove cosets partition G:
--      a. Every element is in some coset
--      b. Cosets are either equal or disjoint
--   3. Prove all cosets have same size as H:
--      a. Define bijection f : H → gH by f(h) = g·h
--      b. Prove f is injective: g·h₁ = g·h₂ → h₁ = h₂
--      c. Prove f is surjective: every g·h is in range
--   4. Count: |G| = (number of cosets) × |H|
```

#### Phase 5: Group Homomorphisms

**Definition:**
A function φ : G → H is a group homomorphism if:
`φ(a · b) = φ(a) · φ(b)` for all a, b ∈ G

**Theorems to Prove:**
1. `hom_one`: φ(e_G) = e_H
2. `hom_inv`: φ(a⁻¹) = φ(a)⁻¹
3. Kernel is a subgroup: ker(φ) = {g ∈ G | φ(g) = e_H}
4. Image is a subgroup: im(φ) = {φ(g) | g ∈ G}
5. φ is injective ↔ ker(φ) = {e}
6. Composition of homomorphisms is a homomorphism

#### Phase 6: Normal Subgroups and Quotient Groups

**Definition:**
N is a normal subgroup of G (N ⊴ G) if:
`∀ g ∈ G, g N g⁻¹ = N`
Equivalently: Left and right cosets coincide

**Theorems to Prove:**
1. Kernel of homomorphism is normal
2. Normal subgroups allow quotient group construction
3. **First Isomorphism Theorem**: G/ker(φ) ≅ im(φ)
4. **Second Isomorphism Theorem**
5. **Third Isomorphism Theorem**

---

## Rings.lean - Ring Theory

### Theoretical Background

A ring is a set R with two binary operations + and · such that:
1. (R, +) is an abelian group
2. Multiplication is associative
3. Distributive laws hold

### Implementation Strategy

#### Phase 1: Define Ring Structure

```lean
class Ring (R : Type) extends AddCommGroup R, Monoid R where
  left_distrib : ∀ a b c, a * (b + c) = a * b + a * c
  right_distrib : ∀ a b c, (a + b) * c = a * c + b * c
```

#### Phase 2: Basic Ring Properties

**Theorems to Prove:**
1. `zero_mul`: 0 · a = 0
2. `mul_zero`: a · 0 = 0
3. `neg_mul`: (-a) · b = -(a · b)
4. `mul_neg`: a · (-b) = -(a · b)
5. `neg_mul_neg`: (-a) · (-b) = a · b

**Example Approach:**
```lean
-- To prove: 0 · a = 0
-- Strategy:
--   0 · a = (0 + 0) · a          (additive identity)
--         = 0 · a + 0 · a        (distributivity)
--   Therefore: 0 · a = 0 · a + 0 · a
--   Subtract 0 · a from both sides: 0 = 0 · a
--
-- In Lean, use:
--   1. Rewrite with zero_add
--   2. Rewrite with right_distrib
--   3. Use additive cancellation
```

#### Phase 3: Ideals

**Definition:**
I is an ideal of R if:
1. I is an additive subgroup
2. ∀ r ∈ R, ∀ a ∈ I, r · a ∈ I and a · r ∈ I (absorption)

**Theorems to Prove:**
1. Ideal criterion
2. Intersection of ideals is an ideal
3. Sum of ideals is an ideal
4. Product of ideals is an ideal
5. Principal ideal generation

#### Phase 4: Quotient Rings

**Theorems to Prove:**
1. Quotient by ideal is a ring
2. Canonical projection is a homomorphism
3. **First Isomorphism Theorem for Rings**

#### Phase 5: Integral Domains and Fields

**Definitions:**
- Integral domain: Commutative ring with 1 and no zero divisors
- Field: Integral domain where every non-zero element has an inverse

**Theorems to Prove:**
1. Finite integral domain is a field
2. Field has no proper ideals
3. Prime ideal ↔ quotient is integral domain
4. Maximal ideal ↔ quotient is field

---

## Fields.lean - Field Theory

### Theoretical Background

A field is a commutative ring where every non-zero element has a multiplicative inverse.

### Implementation Strategy

#### Phase 1: Define Field Structure

```lean
class Field (F : Type) extends Ring F where
  mul_comm : ∀ a b, a * b = b * a
  inv : F → F
  mul_inv_cancel : ∀ a ≠ 0, a * inv a = 1
  inv_zero : inv 0 = 0
```

#### Phase 2: Basic Field Properties

**Theorems to Prove:**
1. `inv_mul_cancel`: a⁻¹ · a = 1 for a ≠ 0
2. `inv_inv`: (a⁻¹)⁻¹ = a
3. `mul_inv`: (a · b)⁻¹ = a⁻¹ · b⁻¹
4. `div_def`: a / b = a · b⁻¹
5. Field homomorphism properties

#### Phase 3: Subfields

**Theorems to Prove:**
1. Subfield criterion
2. Intersection of subfields is a subfield
3. Prime subfield (smallest subfield)

#### Phase 4: Field Extensions

**Definition:**
If F ⊆ E are fields, then E is a field extension of F (denoted E/F).

**Theorems to Prove:**
1. E is a vector space over F
2. Dimension [E:F] (degree of extension)
3. Finite vs infinite extensions

---

## Polynomials.lean - Polynomial Rings

### Theoretical Background

F[X] is the ring of polynomials with coefficients in field F.

### Implementation Strategy

#### Phase 1: Define Polynomial Structure

```lean
-- Polynomial as finitely supported function ℕ → F
structure Polynomial (F : Type) [Field F] where
  coeff : ℕ → F
  support_finite : ∃ n, ∀ m ≥ n, coeff m = 0
```

#### Phase 2: Basic Polynomial Operations

**Define and prove properties:**
1. Addition of polynomials
2. Multiplication of polynomials
3. Degree function
4. Leading coefficient
5. F[X] is a ring

#### Phase 3: Division Algorithm

**Theorem to Prove:**
For f, g ∈ F[X] with g ≠ 0, ∃! q, r such that:
- f = q · g + r
- deg(r) < deg(g)

**Implementation Strategy:**
Use strong induction on degree of f.

#### Phase 4: Irreducibility

**Theorems to Prove:**
1. Define irreducible polynomials
2. F[X] is a unique factorization domain
3. Eisenstein's criterion for irreducibility
4. Irreducibility tests

#### Phase 5: Roots and Factors

**Theorems to Prove:**
1. **Remainder Theorem**: f(a) = r where f = q(X-a) + r
2. **Factor Theorem**: (X-a) | f ↔ f(a) = 0
3. Number of roots ≤ degree
4. Polynomial interpolation

---

## FieldExtensions.lean - Field Extensions

### Theoretical Background

Study of how fields relate to each other through containment.

### Implementation Strategy

#### Phase 1: Simple Extensions

**Define:** F(α) = smallest field containing F and α

**Theorems to Prove:**
1. F(α) is well-defined
2. If α is algebraic over F, then F(α) = F[α]
3. [F(α):F] = deg(min_poly(α))

#### Phase 2: Algebraic Elements

**Definition:**
α is algebraic over F if ∃ non-zero f ∈ F[X] with f(α) = 0.

**Theorems to Prove:**
1. Minimal polynomial exists and is unique
2. Minimal polynomial is irreducible
3. Minimal polynomial is the polynomial of smallest degree with α as root

#### Phase 3: Tower Law

**Theorem to Prove:**
If F ⊆ K ⊆ E, then:
`[E:F] = [E:K] · [K:F]`

**This is crucial for Galois theory!**

#### Phase 4: Algebraic Closure

**Definition:**
F̄ is an algebraic closure of F if:
1. F ⊆ F̄
2. F̄ is algebraically closed (every polynomial splits)
3. F̄ is algebraic over F

**Theorems to Prove:**
1. Algebraic closure exists
2. Algebraic closure is unique up to isomorphism

---

## SplittingFields.lean - Splitting Fields

### Theoretical Background

The splitting field of a polynomial is the smallest field where it factors completely.

### Implementation Strategy

#### Phase 1: Define Splitting Field

**Definition:**
E is a splitting field of f ∈ F[X] over F if:
1. f splits completely in E (factors into linear factors)
2. E is generated over F by the roots of f

#### Phase 2: Existence and Uniqueness

**Theorems to Prove:**
1. Every polynomial has a splitting field
2. Splitting field is unique up to F-isomorphism
3. Splitting field is a finite extension

#### Phase 3: Normal Extensions

**Definition:**
E/F is normal if every irreducible f ∈ F[X] that has one root in E splits completely in E.

**Theorems to Prove:**
1. E/F is normal ↔ E is splitting field of some polynomial
2. Compositum of normal extensions is normal
3. Normal extension properties

#### Phase 4: Separable Extensions

**Definition:**
E/F is separable if every α ∈ E has a separable minimal polynomial.

**Theorems to Prove:**
1. Characteristic 0 ⇒ all extensions are separable
2. Perfect field ⇒ all extensions are separable
3. Separable extensions are preserved by composition

---

## GaloisTheory.lean - The Fundamental Theorem

### Theoretical Background

Galois theory establishes a correspondence between intermediate field extensions and subgroups of the Galois group.

### Implementation Strategy

#### Phase 1: Galois Extensions

**Definition:**
E/F is a Galois extension if it is:
1. Normal
2. Separable  
3. Algebraic (often implicit from being splitting field)

#### Phase 2: Galois Group

**Definition:**
Gal(E/F) = {σ : E → E | σ is field automorphism and σ|_F = id}

**Theorems to Prove:**
1. Gal(E/F) is a group
2. |Gal(E/F)| ≤ [E:F]
3. |Gal(E/F)| = [E:F] for Galois extensions

#### Phase 3: Fixed Fields

**Definition:**
For subgroup H ≤ Gal(E/F), the fixed field is:
`E^H = {a ∈ E | σ(a) = a for all σ ∈ H}`

**Theorems to Prove:**
1. E^H is a subfield
2. F ⊆ E^H ⊆ E
3. H ≤ K ⇒ E^K ⊆ E^H (order reversal)

#### Phase 4: Fundamental Theorem of Galois Theory

**The Big Theorem:**

For Galois extension E/F:

1. **Correspondence**: There is a bijection between:
   - Subgroups H of Gal(E/F)
   - Intermediate fields F ⊆ K ⊆ E
   
   Given by: H ↦ E^H and K ↦ Gal(E/K)

2. **Properties preserved:**
   - [E:K] = |Gal(E/K)|
   - [K:F] = [Gal(E/F) : Gal(E/K)]

3. **Normal subgroups:**
   H ⊴ Gal(E/F) ↔ E^H/F is Galois
   
   When this holds: Gal(E^H/F) ≅ Gal(E/F)/H

**Implementation Strategy:**

This is a major undertaking requiring all previous work!

```lean
-- Step 1: Prove the correspondence is well-defined
-- Show: If H ≤ Gal(E/F), then E^H is an intermediate field
-- Show: If F ⊆ K ⊆ E, then Gal(E/K) ≤ Gal(E/F)

-- Step 2: Prove the correspondence is bijective
-- Show: Gal(E/E^H) = H for all H
-- Show: E^(Gal(E/K)) = K for all K

-- Step 3: Prove order reversal
-- H ≤ K ⇒ E^K ⊆ E^H
-- K ⊆ L ⇒ Gal(E/L) ≤ Gal(E/K)

-- Step 4: Prove the degree formulas
-- [E:E^H] = |H|
-- [E^H:F] = |Gal(E/F)|/|H|

-- Step 5: Prove the normal subgroup correspondence
-- H ⊴ Gal(E/F) ↔ E^H/F is normal (and hence Galois)
-- Prove the isomorphism Gal(E^H/F) ≅ Gal(E/F)/H
```

#### Phase 5: Applications

**Theorems to Prove:**
1. **Solvable by radicals**: 
   - f solvable by radicals ↔ Gal(f) is solvable group
   - Prove S_5 is not solvable ⇒ general quintic not solvable

2. **Ruler and compass**:
   - Constructible ↔ degree is power of 2
   - Doubling cube impossible
   - Trisecting angle impossible
   - Squaring circle impossible (requires π transcendental)

3. **Finite fields**:
   - 𝔽_{p^n} exists and is unique
   - Gal(𝔽_{p^n}/𝔽_p) ≅ ℤ/nℤ generated by Frobenius

---

## Common Lean Tactics Reference

### Basic Tactics

- `intro h`: Introduce hypothesis or lambda binding
- `intros`: Introduce all hypotheses
- `exact h`: Provide exact proof term
- `apply h`: Apply theorem/hypothesis
- `rw [h]`: Rewrite using equality h
- `simp`: Simplify using simp lemmas
- `rfl`: Prove by reflexivity

### Logical Tactics

- `constructor`: Build conjunction or structure
- `cases h`: Case analysis on hypothesis
- `left` / `right`: Choose side of disjunction
- `split`: Split biconditional into two goals
- `exfalso`: Change goal to False (proof by contradiction)
- `by_contra h`: Proof by contradiction

### Induction

- `induction n`: Induction on n
- `induction n with | zero => ... | succ n ih => ...`: Pattern matching style

### Set and Type Tactics

- `ext`: Prove equality by extensionality
- `funext`: Functional extensionality
- `simp [mem_def]`: Simplify membership

### Advanced Tactics

- `calc`: Chain of equalities/inequalities
- `have : statement := proof`: Introduce intermediate result
- `show goal`: State goal explicitly
- `suffices : statement by proof`: Suffices to show

---

## Debugging and Troubleshooting

### Reading Error Messages

Lean error messages typically tell you:
1. What the current goal is
2. What hypotheses you have available
3. Why your tactic failed

### Common Errors

1. **Type mismatch**: The proof term you provided has wrong type
   - Solution: Check goal carefully, use correct constructor

2. **Unknown identifier**: Variable/theorem not in scope
   - Solution: Import module, or define earlier

3. **Invalid apply**: Function/theorem doesn't match goal
   - Solution: Check if you need `apply` or `exact`, or if types align

4. **Tactic failed**: The tactic couldn't complete
   - Solution: Try simpler tactics, break into steps, check hypotheses

### Debugging Workflow

1. Check tactic state (goal + hypotheses)
2. Try `sorry` to see if rest of proof works
3. Break complex proofs into lemmas
4. Use `#check` to verify types
5. Use `#print` to see definitions
6. Compare with similar proofs in Mathlib

### Getting Help

- Lean Zulip: Very active and friendly community
- Look at Mathlib: Many examples to learn from
- Use VS Code hover: Shows type information
- Use `#check` and `#print`: Inspect definitions

---

## Final Notes

This guide provides a roadmap, but the actual proving is up to you! Some tips:

1. **Start small**: Don't jump to Galois theory immediately
2. **Understand deeply**: Don't just copy proofs, understand why they work
3. **Use Mathlib**: Learn from existing proofs
4. **Be patient**: Galois theory is a months-long journey
5. **Have fun**: Proving theorems in Lean is deeply satisfying!

Remember: The goal is not just to prove Galois theorem, but to understand it deeply through the process of formalization. Good luck! 🎉
