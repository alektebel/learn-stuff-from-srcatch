-- Basic Logic Proofs in Lean - Complete Solution

-- Modus Ponens
theorem modus_ponens (P Q : Prop) (h1 : P → Q) (h2 : P) : Q :=
  h1 h2

-- Conjunction introduction
theorem and_intro (P Q : Prop) (hp : P) (hq : Q) : P ∧ Q :=
  ⟨hp, hq⟩

-- Conjunction elimination (left)
theorem and_elim_left (P Q : Prop) (h : P ∧ Q) : P :=
  h.left

-- Conjunction elimination (right)
theorem and_elim_right (P Q : Prop) (h : P ∧ Q) : Q :=
  h.right

-- Disjunction introduction (left)
theorem or_intro_left (P Q : Prop) (hp : P) : P ∨ Q :=
  Or.inl hp

-- Disjunction introduction (right)
theorem or_intro_right (P Q : Prop) (hq : Q) : P ∨ Q :=
  Or.inr hq

-- OR is commutative
theorem or_comm (P Q : Prop) : P ∨ Q → Q ∨ P :=
  fun h => match h with
    | Or.inl hp => Or.inr hp
    | Or.inr hq => Or.inl hq

-- Implication is transitive
theorem imp_trans (P Q R : Prop) (h1 : P → Q) (h2 : Q → R) : P → R :=
  fun hp => h2 (h1 hp)

-- Double negation introduction
theorem not_not_intro (P : Prop) (hp : P) : ¬¬P :=
  fun hnp => hnp hp

-- Contradiction
theorem ex_falso (P Q : Prop) (hp : P) (hnp : ¬P) : Q :=
  absurd hp hnp

-- AND is commutative
theorem and_comm (P Q : Prop) : P ∧ Q ↔ Q ∧ P :=
  ⟨fun ⟨hp, hq⟩ => ⟨hq, hp⟩, fun ⟨hq, hp⟩ => ⟨hp, hq⟩⟩

-- De Morgan's Law
theorem de_morgan_or (P Q : Prop) : ¬(P ∨ Q) ↔ ¬P ∧ ¬Q :=
  ⟨fun h => ⟨fun hp => h (Or.inl hp), fun hq => h (Or.inr hq)⟩,
   fun ⟨hnp, hnq⟩ h => match h with
     | Or.inl hp => hnp hp
     | Or.inr hq => hnq hq⟩

-- Additional useful theorems

-- AND is associative
theorem and_assoc (P Q R : Prop) : (P ∧ Q) ∧ R ↔ P ∧ (Q ∧ R) :=
  ⟨fun ⟨⟨hp, hq⟩, hr⟩ => ⟨hp, ⟨hq, hr⟩⟩,
   fun ⟨hp, ⟨hq, hr⟩⟩ => ⟨⟨hp, hq⟩, hr⟩⟩

-- OR is associative
theorem or_assoc (P Q R : Prop) : (P ∨ Q) ∨ R ↔ P ∨ (Q ∨ R) := by
  constructor
  · intro h
    cases h with
    | inl h' =>
      cases h' with
      | inl hp => exact Or.inl hp
      | inr hq => exact Or.inr (Or.inl hq)
    | inr hr => exact Or.inr (Or.inr hr)
  · intro h
    cases h with
    | inl hp => exact Or.inl (Or.inl hp)
    | inr h' =>
      cases h' with
      | inl hq => exact Or.inl (Or.inr hq)
      | inr hr => exact Or.inr hr

-- Distributive law: P ∧ (Q ∨ R) ↔ (P ∧ Q) ∨ (P ∧ R)
theorem and_or_distrib (P Q R : Prop) : P ∧ (Q ∨ R) ↔ (P ∧ Q) ∨ (P ∧ R) :=
  ⟨fun ⟨hp, hqr⟩ => match hqr with
     | Or.inl hq => Or.inl ⟨hp, hq⟩
     | Or.inr hr => Or.inr ⟨hp, hr⟩,
   fun h => match h with
     | Or.inl ⟨hp, hq⟩ => ⟨hp, Or.inl hq⟩
     | Or.inr ⟨hp, hr⟩ => ⟨hp, Or.inr hr⟩⟩
