-- Basic Logic Proofs in Lean - Template
-- This file guides you through proving basic logical statements in Lean

-- TODO 1: Prove basic propositional logic theorems
-- Guidelines: Use intro, apply, exact tactics

-- Modus Ponens: If P implies Q, and P is true, then Q is true
theorem modus_ponens (P Q : Prop) (h1 : P → Q) (h2 : P) : Q :=
  sorry  -- TODO: Prove this using apply or direct function application

-- Conjunction introduction: If P and Q are true, then P ∧ Q is true
theorem and_intro (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q :=
  sorry  -- TODO: Use And.intro or ⟨hp, hq⟩

-- Conjunction elimination (left): If P ∧ Q is true, then P is true
theorem and_elim_left (P Q : Prop) (h : P ∧ Q) : P :=
  sorry  -- TODO: Use h.left or And.left h

-- Conjunction elimination (right): If P ∧ Q is true, then Q is true
theorem and_elim_right (P Q : Prop) (h : P ∧ Q) : Q :=
  sorry  -- TODO: Use h.right or And.right h

-- TODO 2: Prove theorems about disjunction (OR)
-- Guidelines: Use Or.inl, Or.inr, and cases for elimination

-- Disjunction introduction (left): If P is true, then P ∨ Q is true
theorem or_intro_left (P Q : Prop) (hp : P) : P ∨ Q :=
  sorry  -- TODO: Use Or.inl

-- Disjunction introduction (right): If Q is true, then P ∨ Q is true
theorem or_intro_right (P Q : Prop) (hq : Q) : P ∨ Q :=
  sorry  -- TODO: Use Or.inr

-- OR is commutative
theorem or_comm (P Q : Prop) : P ∨ Q → Q ∨ P :=
  sorry  -- TODO: Use intro, cases, and Or constructors

-- TODO 3: Prove theorems about implication
-- Guidelines: Use intro (λ), apply

-- Implication is transitive
theorem imp_trans (P Q R : Prop) (h1 : P → Q) (h2 : Q → R) : P → R :=
  sorry  -- TODO: Chain implications

-- TODO 4: Prove theorems about negation
-- Guidelines: Negation is defined as ¬P := P → False

-- Double negation introduction
theorem not_not_intro (P : Prop) (hp : P) : ¬¬P :=
  sorry  -- TODO: Prove ¬¬P (which means (P → False) → False)

-- Contradiction: From P and ¬P, prove anything
theorem ex_falso (P Q : Prop) (hp : P) (hnp : ¬P) : Q :=
  sorry  -- TODO: Use absurd or False.elim

-- TODO 5: Prove basic equivalence theorems
-- Guidelines: Use Iff.intro, split into two directions

-- AND is commutative
theorem and_comm (P Q : Prop) : P ∧ Q ↔ Q ∧ P :=
  sorry  -- TODO: Prove both directions

-- De Morgan's Law: ¬(P ∨ Q) ↔ ¬P ∧ ¬Q
theorem de_morgan_or (P Q : Prop) : ¬(P ∨ Q) ↔ ¬P ∧ ¬Q :=
  sorry  -- TODO: Prove both directions (challenging!)

/-
IMPLEMENTATION GUIDE:

Step 1: Start with simple theorems (modus_ponens, and_intro, and_elim)
        Learn basic tactics: intro, apply, exact

Step 2: Work on disjunction theorems
        Learn: cases, Or.inl, Or.inr

Step 3: Practice implication chains
        Understand function composition in logic

Step 4: Tackle negation
        Remember: ¬P is defined as P → False

Step 5: Prove equivalences
        Use Iff.intro or ⟨forward, backward⟩

Common Tactics:
- intro : Introduce a hypothesis or lambda
- apply : Apply a theorem or hypothesis
- exact : Provide exact proof term
- cases : Case analysis on a hypothesis
- constructor : Use a constructor of an inductive type
- sorry : Placeholder (admits the theorem without proof)

Learning Resources:
- Theorem Proving in Lean: https://leanprover.github.io/theorem_proving_in_lean/
- Lean Community: https://leanprover-community.github.io/
-/
