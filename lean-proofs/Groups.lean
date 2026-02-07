-- Group Theory in Lean - Template
-- This file guides you through the fundamentals of group theory
-- Groups are the key algebraic structure in Galois theory (Galois groups!)

/-
LEARNING OBJECTIVES:
1. Understand group axioms and structure
2. Prove basic group properties from axioms
3. Work with subgroups and cosets
4. Understand and prove Lagrange's theorem
5. Study group homomorphisms and isomorphisms
6. Learn about normal subgroups and quotient groups
7. Prove the isomorphism theorems
-/

-- ============================================================================
-- PART 1: GROUP DEFINITION AND BASIC PROPERTIES
-- ============================================================================

-- A group is a set G with operation · satisfying:
-- 1. Associativity: (a · b) · c = a · (b · c)
-- 2. Identity: ∃ e, e · a = a · e = a
-- 3. Inverses: ∀ a, ∃ a⁻¹, a · a⁻¹ = a⁻¹ · a = e

class Group (G : Type) where
  mul : G → G → G
  one : G
  inv : G → G
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  mul_left_inv : ∀ a, mul (inv a) a = one

-- Notation
infixl:70 " * " => Group.mul
notation "e" => Group.one
postfix:max "⁻¹" => Group.inv

section BasicGroupProperties

variable {G : Type} [Group G]

-- TODO 1.1: Prove right inverse from left inverse
-- We have a⁻¹ · a = e as an axiom, need to prove a · a⁻¹ = e
theorem mul_right_inv (a : G) : a * a⁻¹ = e :=
  sorry
  -- Hint: Multiply (a * a⁻¹) on left by (a⁻¹)⁻¹
  -- Strategy:
  --   a * a⁻¹ = e * (a * a⁻¹)
  --         = ((a⁻¹)⁻¹ * a⁻¹) * (a * a⁻¹)
  --         = (a⁻¹)⁻¹ * (a⁻¹ * (a * a⁻¹))
  --         = (a⁻¹)⁻¹ * ((a⁻¹ * a) * a⁻¹)
  --         = (a⁻¹)⁻¹ * (e * a⁻¹)
  --         = (a⁻¹)⁻¹ * a⁻¹
  --         = e

-- TODO 1.2: Prove inverse of inverse
theorem inv_inv (a : G) : (a⁻¹)⁻¹ = a :=
  sorry
  -- Hint: Show that a acts as inverse of a⁻¹
  -- Use uniqueness of inverse

-- TODO 1.3: Prove identity is unique
theorem one_unique {e' : G} (h : ∀ a, e' * a = a) : e' = e :=
  sorry
  -- Hint: Apply h to e, use one_mul

-- TODO 1.4: Prove inverse is unique
theorem inv_unique {a b : G} (h : b * a = e) : b = a⁻¹ :=
  sorry
  -- Hint: Multiply both sides by a⁻¹ on the right

-- TODO 1.5: Prove left cancellation
theorem mul_left_cancel {a b c : G} : a * b = a * c → b = c :=
  sorry
  -- Hint: Multiply both sides by a⁻¹ on the left

-- TODO 1.6: Prove right cancellation
theorem mul_right_cancel {a b c : G} : a * c = b * c → a = b :=
  sorry
  -- Hint: Multiply both sides by c⁻¹ on the right

-- TODO 1.7: Prove inverse of product
theorem mul_inv (a b : G) : (a * b)⁻¹ = b⁻¹ * a⁻¹ :=
  sorry
  -- Hint: Show that b⁻¹ * a⁻¹ is the inverse of a * b
  -- Verify: (a * b) * (b⁻¹ * a⁻¹) = e

end BasicGroupProperties

-- ============================================================================
-- PART 2: SUBGROUPS
-- ============================================================================

section Subgroups

variable {G : Type} [Group G]

-- A subgroup is a subset H ⊆ G that forms a group under the same operation
structure Subgroup (G : Type) [Group G] where
  carrier : Set G
  one_mem : e ∈ carrier
  mul_mem : ∀ {a b}, a ∈ carrier → b ∈ carrier → a * b ∈ carrier
  inv_mem : ∀ {a}, a ∈ carrier → a⁻¹ ∈ carrier

-- Notation
instance : SetLike (Subgroup G) G where
  coe := Subgroup.carrier
  coe_injective' := sorry  -- Prove subgroups equal if carriers equal

-- TODO 2.1: Prove subgroup criterion (one-step subgroup test)
-- H is a subgroup iff e ∈ H and ∀ a,b ∈ H, a * b⁻¹ ∈ H
theorem subgroup_criterion (H : Set G) :
    (e ∈ H ∧ ∀ a b, a ∈ H → b ∈ H → a * b⁻¹ ∈ H) ↔
    ∃ (HG : Subgroup G), H = HG.carrier :=
  sorry
  -- Forward: Given criterion, construct subgroup
  -- Backward: Given subgroup, verify criterion

-- TODO 2.2: Prove intersection of subgroups is a subgroup
def subgroup_inter (H K : Subgroup G) : Subgroup G where
  carrier := H.carrier ∩ K.carrier
  one_mem := sorry  -- TODO: e ∈ H and e ∈ K
  mul_mem := sorry  -- TODO: Use mul_mem of both H and K
  inv_mem := sorry  -- TODO: Use inv_mem of both H and K

-- TODO 2.3: Prove {e} is a subgroup (trivial subgroup)
def trivial_subgroup : Subgroup G where
  carrier := {e}
  one_mem := sorry
  mul_mem := sorry  -- TODO: e * e = e
  inv_mem := sorry  -- TODO: e⁻¹ = e

-- TODO 2.4: Prove G is a subgroup of itself
def top_subgroup : Subgroup G where
  carrier := Set.univ
  one_mem := sorry
  mul_mem := sorry
  inv_mem := sorry

end Subgroups

-- ============================================================================
-- PART 3: COSETS AND LAGRANGE'S THEOREM
-- ============================================================================

section Cosets

variable {G : Type} [Group G] (H : Subgroup G)

-- Left coset: a * H = {a * h | h ∈ H}
def leftCoset (a : G) : Set G := {g | ∃ h ∈ H, g = a * h}

-- Right coset: H * a = {h * a | h ∈ H}
def rightCoset (a : G) : Set G := {g | ∃ h ∈ H, g = h * a}

-- TODO 3.1: Prove a ∈ a * H
theorem mem_leftCoset (a : G) : a ∈ leftCoset H a :=
  sorry
  -- Hint: a = a * e and e ∈ H

-- TODO 3.2: Prove cosets are equal or disjoint
theorem coset_disjoint_or_eq (a b : G) :
    leftCoset H a = leftCoset H b ∨ leftCoset H a ∩ leftCoset H b = ∅ :=
  sorry
  -- Hint: If intersection non-empty, show cosets equal
  -- Use group axioms to relate elements

-- TODO 3.3: Prove a * H = b * H iff a⁻¹ * b ∈ H
theorem leftCoset_eq_iff (a b : G) :
    leftCoset H a = leftCoset H b ↔ a⁻¹ * b ∈ H :=
  sorry
  -- Hint: If a * H = b * H, then b ∈ a * H
  -- So b = a * h for some h ∈ H
  -- Thus a⁻¹ * b = h ∈ H

-- TODO 3.4: Define bijection between H and a * H
def leftCosetBijection (a : G) : H → leftCoset H a :=
  fun ⟨h, hH⟩ => ⟨a * h, sorry⟩  -- TODO: Prove a * h ∈ leftCoset H a

-- TODO 3.5: Prove all cosets have the same cardinality as H
-- This is implied by the bijection above in finite case

-- TODO 3.6: LAGRANGE'S THEOREM
-- |G| = |G/H| * |H| where |G/H| is the number of left cosets
-- For finite groups: order of subgroup divides order of group
axiom lagrange_theorem {G : Type} [Group G] [Fintype G] (H : Subgroup G) [Fintype H] :
  ∃ k : ℕ, Fintype.card G = k * Fintype.card H
  -- Proof sketch:
  -- 1. Cosets partition G
  -- 2. All cosets have same size as H
  -- 3. Count: |G| = (number of cosets) * |H|

-- TODO 3.7: Corollary - Order of element divides order of group
-- If a ∈ G has order n, then n divides |G|

end Cosets

-- ============================================================================
-- PART 4: GROUP HOMOMORPHISMS
-- ============================================================================

section Homomorphisms

variable {G H : Type} [Group G] [Group H]

-- A group homomorphism preserves the group operation
structure GroupHom (G H : Type) [Group G] [Group H] where
  toFun : G → H
  map_mul : ∀ a b, toFun (a * b) = toFun a * toFun b

-- Notation
infixr:25 " →* " => GroupHom
instance : CoeFun (G →* H) (fun _ => G → H) := ⟨GroupHom.toFun⟩

variable (φ : G →* H)

-- TODO 4.1: Prove homomorphism preserves identity
theorem map_one : φ e = e :=
  sorry
  -- Hint: φ(e) = φ(e * e) = φ(e) * φ(e)
  -- Cancel φ(e) from both sides

-- TODO 4.2: Prove homomorphism preserves inverses
theorem map_inv (a : G) : φ a⁻¹ = (φ a)⁻¹ :=
  sorry
  -- Hint: Show φ(a⁻¹) * φ(a) = e
  -- Use uniqueness of inverse

-- Define kernel and image
def ker : Subgroup G where
  carrier := {a | φ a = e}
  one_mem := sorry  -- TODO: Use map_one
  mul_mem := sorry  -- TODO: Use map_mul
  inv_mem := sorry  -- TODO: Use map_inv

def im : Subgroup H where
  carrier := {b | ∃ a, φ a = b}
  one_mem := sorry
  mul_mem := sorry
  inv_mem := sorry

-- TODO 4.3: Prove φ is injective iff ker(φ) = {e}
theorem injective_iff_trivial_ker :
    Function.Injective φ ↔ ker φ = trivial_subgroup :=
  sorry
  -- Forward: If φ injective and φ(a) = e = φ(e), then a = e
  -- Backward: If φ(a) = φ(b), then φ(a * b⁻¹) = e, so a * b⁻¹ = e

-- TODO 4.4: Prove composition of homomorphisms
def comp {K : Type} [Group K] (ψ : H →* K) : G →* K where
  toFun := ψ ∘ φ
  map_mul := sorry  -- TODO: Use map_mul of both φ and ψ

end Homomorphisms

-- ============================================================================
-- PART 5: NORMAL SUBGROUPS AND QUOTIENT GROUPS
-- ============================================================================

section NormalSubgroups

variable {G : Type} [Group G]

-- N is normal if g * N * g⁻¹ = N for all g ∈ G
-- Equivalently: left and right cosets coincide
def IsNormal (N : Subgroup G) : Prop :=
  ∀ n ∈ N, ∀ g : G, g * n * g⁻¹ ∈ N

notation:50 N " ⊴ " G:50 => IsNormal N

-- TODO 5.1: Prove kernel of homomorphism is normal
theorem kernel_normal {H : Type} [Group H] (φ : G →* H) :
    IsNormal (ker φ) :=
  sorry
  -- Hint: If φ(n) = e, show φ(g * n * g⁻¹) = e
  -- Use φ(g * n * g⁻¹) = φ(g) * φ(n) * φ(g⁻¹) = φ(g) * e * φ(g)⁻¹ = e

-- TODO 5.2: Define quotient group
-- When N is normal, G/N has a natural group structure
def QuotientGroup (N : Subgroup G) (h : IsNormal N) : Type :=
  Quotient (sorry : Setoid G)  -- TODO: Define equivalence relation

-- TODO 5.3: Define quotient group operations
-- Need to show operations are well-defined

end NormalSubgroups

-- ============================================================================
-- PART 6: ISOMORPHISM THEOREMS
-- ============================================================================

section IsomorphismTheorems

-- TODO 6.1: FIRST ISOMORPHISM THEOREM
-- If φ : G → H is a homomorphism, then G/ker(φ) ≅ im(φ)
axiom first_isomorphism_theorem {G H : Type} [Group G] [Group H] (φ : G →* H) :
  ∃ ψ : QuotientGroup (ker φ) sorry →* sorry, Function.Bijective ψ
  -- Proof sketch:
  -- Define ψ : G/ker(φ) → im(φ) by ψ(g * ker(φ)) = φ(g)
  -- Well-defined: If g₁ * ker(φ) = g₂ * ker(φ), then φ(g₁) = φ(g₂)
  -- Homomorphism: ψ((g₁ * ker(φ)) * (g₂ * ker(φ))) = φ(g₁ * g₂) = φ(g₁) * φ(g₂)
  -- Bijective: Injective by construction, surjective by definition of image

-- TODO 6.2: SECOND ISOMORPHISM THEOREM (Diamond Isomorphism)
-- If H ≤ G and N ⊴ G, then H/(H ∩ N) ≅ (HN)/N

-- TODO 6.3: THIRD ISOMORPHISM THEOREM
-- If N, K ⊴ G with N ≤ K, then (G/N)/(K/N) ≅ G/K

end IsomorphismTheorems

/-
IMPLEMENTATION GUIDE SUMMARY:

Part 1 - Basic Properties:
  Work from axioms using only algebra
  Key technique: multiply by inverses strategically
  Right inverse requires a clever trick with double inverse
  All other properties follow logically
  
Part 2 - Subgroups:
  Subgroup criterion is practical test
  Intersection is straightforward
  These prepare for Galois correspondence
  
Part 3 - Lagrange's Theorem:
  This is a major theorem!
  Cosets partition the group
  All cosets same size (bijection)
  Counting gives the theorem
  Applications: order of element divides order of group
  
Part 4 - Homomorphisms:
  Preserve structure automatically
  Kernel measures "failure to be injective"
  Image is range of homomorphism
  Injective iff trivial kernel (important!)
  
Part 5 - Normal Subgroups:
  Normal ⟺ kernel of some homomorphism
  Enable quotient group construction
  Essential for Galois theory (normal extensions!)
  
Part 6 - Isomorphism Theorems:
  Connect quotients, kernels, and images
  First isomorphism theorem is most important
  Used constantly in Galois theory
  Provide general framework for understanding groups

KEY PROOF TECHNIQUES:
- Strategic multiplication by inverses
- Use of group axioms (especially associativity)
- Cancellation laws
- Uniqueness arguments
- Quotient reasoning

CONNECTION TO GALOIS THEORY:
- Galois group Gal(E/F) is a group
- Intermediate fields ↔ Subgroups
- Normal extensions ↔ Normal subgroups
- Isomorphism theorems explain field/group correspondence

NEXT STEPS:
After Groups.lean, move to Rings.lean
Rings add another operation (+) to group structure
This prepares for polynomial rings and fields
-/
