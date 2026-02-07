-- Set Theory in Lean - Template
-- This file guides you through proving fundamental set theory theorems
-- progressing toward understanding relations and functions needed for Galois theory

-- In Lean, a set is represented as a predicate (α → Prop)
-- x ∈ S means S x evaluates to true

/-
LEARNING OBJECTIVES:
1. Understand how sets work in Lean (sets as predicates)
2. Prove basic set operations and their properties
3. Work with relations (reflexive, symmetric, transitive)
4. Understand functions (injective, surjective, bijective)
5. Prove properties about function composition
-/

-- ============================================================================
-- PART 1: BASIC SET OPERATIONS
-- ============================================================================

section BasicSetOps

variable {α : Type} (S T U : Set α)

-- TODO 1.1: Prove subset is reflexive
-- Recall: S ⊆ T means ∀ x, x ∈ S → x ∈ T
theorem subset_refl : S ⊆ S :=
  sorry  -- TODO: Use intro tactics to introduce x and hypothesis

-- TODO 1.2: Prove subset is transitive  
theorem subset_trans : S ⊆ T → T ⊆ U → S ⊆ U :=
  sorry  -- TODO: Chain the two subset hypotheses

-- TODO 1.3: Prove subset is antisymmetric (needs extensionality)
theorem subset_antisymm : S ⊆ T → T ⊆ S → S = T :=
  sorry  -- TODO: Use Set.ext (extensionality) to prove set equality

-- TODO 1.4: Prove empty set has no elements
theorem mem_empty (x : α) : x ∉ (∅ : Set α) :=
  sorry  -- TODO: Empty set is defined so this should be immediate

-- TODO 1.5: Prove universal set contains all elements
theorem mem_univ (x : α) : x ∈ (Set.univ : Set α) :=
  sorry  -- TODO: Universal set is defined to contain everything

end BasicSetOps

-- ============================================================================
-- PART 2: UNION AND INTERSECTION
-- ============================================================================

section UnionIntersection

variable {α : Type} (S T U : Set α)

-- TODO 2.1: Characterize union membership
theorem mem_union (x : α) : x ∈ S ∪ T ↔ x ∈ S ∨ x ∈ T :=
  sorry  -- TODO: This should be definitional, but prove both directions

-- TODO 2.2: Characterize intersection membership
theorem mem_inter (x : α) : x ∈ S ∩ T ↔ x ∈ S ∧ x ∈ T :=
  sorry  -- TODO: Similar to union, prove both directions

-- TODO 2.3: Prove union is commutative
theorem union_comm : S ∪ T = T ∪ S :=
  sorry  -- TODO: Use ext, then unfold union, use or_comm from BasicLogic

-- TODO 2.4: Prove intersection is commutative  
theorem inter_comm : S ∩ T = T ∩ S :=
  sorry  -- TODO: Similar to union_comm, use and_comm

-- TODO 2.5: Prove union is associative
theorem union_assoc : (S ∪ T) ∪ U = S ∪ (T ∪ U) :=
  sorry  -- TODO: Use ext and or_assoc from BasicLogic

-- TODO 2.6: Prove intersection is associative
theorem inter_assoc : (S ∩ T) ∩ U = S ∩ (T ∩ U) :=
  sorry  -- TODO: Use ext and and_assoc from BasicLogic

-- TODO 2.7: Prove intersection distributes over union
theorem inter_distrib_union : S ∩ (T ∪ U) = (S ∩ T) ∪ (S ∩ U) :=
  sorry  -- TODO: Use ext, unfold operations, use and_or_distrib from BasicLogic

-- TODO 2.8: Prove union distributes over intersection
theorem union_distrib_inter : S ∪ (T ∩ U) = (S ∪ T) ∩ (S ∪ U) :=
  sorry  -- TODO: Similar to above, use or_and_distrib

end UnionIntersection

-- ============================================================================
-- PART 3: COMPLEMENT AND DIFFERENCE
-- ============================================================================

section Complement

variable {α : Type} (S T : Set α)

-- TODO 3.1: Characterize complement membership
theorem mem_compl (x : α) : x ∈ Sᶜ ↔ x ∉ S :=
  sorry  -- TODO: Complement is defined this way

-- TODO 3.2: Prove De Morgan's law for union
theorem compl_union : (S ∪ T)ᶜ = Sᶜ ∩ Tᶜ :=
  sorry  -- TODO: Use ext, unfold, use de_morgan_or from BasicLogic

-- TODO 3.3: Prove De Morgan's law for intersection (requires classical logic)
theorem compl_inter : (S ∩ T)ᶜ = Sᶜ ∪ Tᶜ :=
  sorry  -- TODO: Use ext, unfold, may need Classical.em (excluded middle)

-- TODO 3.4: Prove double complement
theorem compl_compl : Sᶜᶜ = S :=
  sorry  -- TODO: Use ext, may need classical logic for one direction

end Complement

-- ============================================================================
-- PART 4: RELATIONS
-- ============================================================================

section Relations

variable {α : Type}

-- Define properties of relations
def Reflexive (R : α → α → Prop) : Prop := ∀ x, R x x

def Symmetric (R : α → α → Prop) : Prop := ∀ x y, R x y → R y x

def Transitive (R : α → α → Prop) : Prop := ∀ x y z, R x y → R y z → R x z

def Equivalence (R : α → α → Prop) : Prop :=
  Reflexive R ∧ Symmetric R ∧ Transitive R

-- TODO 4.1: Prove equality is reflexive
theorem eq_refl : Reflexive (@Eq α) :=
  sorry  -- TODO: Unfold Reflexive, use rfl

-- TODO 4.2: Prove equality is symmetric
theorem eq_symm : Symmetric (@Eq α) :=
  sorry  -- TODO: Unfold Symmetric, use Eq.symm

-- TODO 4.3: Prove equality is transitive
theorem eq_trans : Transitive (@Eq α) :=
  sorry  -- TODO: Unfold Transitive, use Eq.trans

-- TODO 4.4: Prove equality is an equivalence relation
theorem eq_equiv : Equivalence (@Eq α) :=
  sorry  -- TODO: Combine the three previous theorems

-- TODO 4.5: Prove intersection of equivalence relations is an equivalence relation
theorem inter_equiv {R S : α → α → Prop} 
    (hR : Equivalence R) (hS : Equivalence S) : 
    Equivalence (fun x y => R x y ∧ S x y) :=
  sorry  -- TODO: Prove each property separately using hypotheses

-- Define equivalence class
def EquivClass (R : α → α → Prop) (a : α) : Set α :=
  {x | R a x}

-- TODO 4.6: Prove element is in its own equivalence class (if R is reflexive)
theorem mem_equiv_class_self {R : α → α → Prop} (hrefl : Reflexive R) (a : α) :
    a ∈ EquivClass R a :=
  sorry  -- TODO: Unfold EquivClass, use reflexivity

-- TODO 4.7: Prove equivalence classes are equal iff elements are related
theorem equiv_class_eq_iff {R : α → α → Prop} (heq : Equivalence R) (a b : α) :
    EquivClass R a = EquivClass R b ↔ R a b :=
  sorry  -- TODO: This is a key theorem! Use extensionality and symmetry/transitivity

end Relations

-- ============================================================================
-- PART 5: FUNCTIONS
-- ============================================================================

section Functions

variable {α β γ : Type}

-- Define function properties
def Injective (f : α → β) : Prop := ∀ a₁ a₂, f a₁ = f a₂ → a₁ = a₂

def Surjective (f : α → β) : Prop := ∀ b, ∃ a, f a = b

def Bijective (f : α → β) : Prop := Injective f ∧ Surjective f

-- TODO 5.1: Prove identity function is injective
theorem id_injective : Injective (@id α) :=
  sorry  -- TODO: Unfold Injective, id x = x so this is immediate

-- TODO 5.2: Prove identity function is surjective
theorem id_surjective : Surjective (@id α) :=
  sorry  -- TODO: Unfold Surjective, for any b, choose a = b

-- TODO 5.3: Prove identity function is bijective
theorem id_bijective : Bijective (@id α) :=
  sorry  -- TODO: Combine the two previous theorems

-- TODO 5.4: Prove composition of injections is injective
theorem comp_injective {f : α → β} {g : β → γ} 
    (hf : Injective f) (hg : Injective g) : Injective (g ∘ f) :=
  sorry  -- TODO: If g(f(a₁)) = g(f(a₂)), use injectivity of g then f

-- TODO 5.5: Prove composition of surjections is surjective
theorem comp_surjective {f : α → β} {g : β → γ}
    (hf : Surjective f) (hg : Surjective g) : Surjective (g ∘ f) :=
  sorry  -- TODO: For any c, get b from surj of g, then a from surj of f

-- TODO 5.6: Prove composition of bijections is bijective
theorem comp_bijective {f : α → β} {g : β → γ}
    (hf : Bijective f) (hg : Bijective g) : Bijective (g ∘ f) :=
  sorry  -- TODO: Combine previous two theorems

-- TODO 5.7: Prove bijection has a left inverse (requires choice)
-- This is more advanced and may require axiom of choice
theorem bijective_has_inverse {f : α → β} (hf : Bijective f) :
    ∃ g : β → α, (∀ a, g (f a) = a) ∧ (∀ b, f (g b) = b) :=
  sorry  -- TODO: Constructing inverse requires choice axiom, use Classical.choice

end Functions

/-
IMPLEMENTATION GUIDE SUMMARY:

Part 1 - Basic Set Operations:
  Start with simple subset proofs using intro/apply
  Learn set extensionality for equality proofs
  
Part 2 - Union and Intersection:
  Use ext tactic to prove set equality
  Reduce to logical equivalences from BasicLogic
  Pattern: unfold operations, use logical lemmas
  
Part 3 - Complement:
  Similar to union/intersection but with negation
  Some proofs require classical logic (open Classical)
  De Morgan's laws connect to propositional logic
  
Part 4 - Relations:
  Define properties as predicates on relations
  Prove equality satisfies all properties (good practice)
  Equivalence classes are fundamental for quotient structures
  Key theorem: equiv_class_eq_iff (needs careful proof)
  
Part 5 - Functions:
  Composition proofs follow clear patterns
  Injective: chain implications backward
  Surjective: chain existentials forward  
  Inverse requires axiom of choice (advanced)

TACTICS FREQUENTLY USED:
- intro/intros: Introduce hypotheses
- ext: Set extensionality (to prove S = T)
- apply: Apply subset or implication
- exact: Provide exact term
- constructor: Build ∧ or structure
- cases: Case analysis on ∨
- use: Provide witness for ∃
- funext: Function extensionality

TIPS:
- Draw Venn diagrams for set operations
- Think about which direction is harder for ↔ proofs
- Use previously proved lemmas from BasicLogic
- Start with easier proofs to build confidence
- Check Mathlib for similar proofs as examples

NEXT STEPS:
After completing this file, move to NaturalNumbers.lean
to learn induction, which is crucial for algebraic proofs.
-/
