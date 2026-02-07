-- Ring Theory in Lean - Template
-- This file guides you through the fundamentals of ring theory
-- Rings extend groups by adding a second operation (multiplication)

/-
LEARNING OBJECTIVES:
1. Understand ring axioms and how they extend group structure
2. Prove basic ring properties (zero products, distributivity)
3. Work with subrings and ideals
4. Understand why ideals are more important than subrings
5. Study ring homomorphisms and their kernels
6. Learn about quotient rings (factor rings)
7. Understand integral domains and their properties
8. Study prime and maximal ideals
9. Connect rings to fields and prepare for field theory
10. Build foundation for polynomial rings and algebraic structures
-/

-- ============================================================================
-- PART 1: RING DEFINITION AND BASIC PROPERTIES
-- ============================================================================

-- A ring is a set R with two operations + and * satisfying:
-- 1. (R, +) is an abelian group
-- 2. (R, *) is associative with identity 1
-- 3. Distributive laws: a * (b + c) = a * b + a * c and (a + b) * c = a * c + b * c

class Ring (R : Type) where
  -- Addition forms an abelian group
  add : R → R → R
  zero : R
  neg : R → R
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  zero_add : ∀ a, add zero a = a
  add_zero : ∀ a, add a zero = a
  add_left_neg : ∀ a, add (neg a) a = zero
  add_comm : ∀ a b, add a b = add b a
  
  -- Multiplication is associative with identity
  mul : R → R → R
  one : R
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  
  -- Distributive laws
  left_distrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  right_distrib : ∀ a b c, mul (add a b) c = add (mul a c) (mul b c)

-- Notation
infixl:65 " + " => Ring.add
infixl:70 " * " => Ring.mul
notation "0" => Ring.zero
notation "1" => Ring.one
prefix:75 "-" => Ring.neg

section BasicRingProperties

variable {R : Type} [Ring R]

-- TODO 1.1: Prove 0 * a = 0 for any a
theorem zero_mul (a : R) : 0 * a = 0 :=
  sorry
  -- Hint: Start with 0 * a = (0 + 0) * a
  -- Use distributive law: = 0 * a + 0 * a
  -- Then cancel one copy of 0 * a from both sides
  -- Strategy:
  --   0 * a = (0 + 0) * a         (by zero_add)
  --       = 0 * a + 0 * a         (by right_distrib)
  -- So 0 * a + (-0 * a) = (0 * a + 0 * a) + (-0 * a)
  --                  0 = 0 * a + (0 * a + (-0 * a))
  --                  0 = 0 * a + 0
  --                  0 = 0 * a

-- TODO 1.2: Prove a * 0 = 0 for any a
theorem mul_zero (a : R) : a * 0 = 0 :=
  sorry
  -- Hint: Similar to above, use left_distrib
  -- a * 0 = a * (0 + 0) = a * 0 + a * 0

-- TODO 1.3: Prove (-1) * a = -a
theorem neg_one_mul (a : R) : (-1) * a = -a :=
  sorry
  -- Hint: Show (-1) * a + a = 0
  -- Use: (-1) * a + a = (-1) * a + 1 * a = (-1 + 1) * a = 0 * a = 0
  -- Then use uniqueness of additive inverse

-- TODO 1.4: Prove a * (-b) = -(a * b)
theorem mul_neg (a b : R) : a * (-b) = -(a * b) :=
  sorry
  -- Hint: Show a * (-b) + a * b = 0
  -- Use: a * (-b) + a * b = a * (-b + b) = a * 0 = 0

-- TODO 1.5: Prove (-a) * b = -(a * b)
theorem neg_mul (a b : R) : (-a) * b = -(a * b) :=
  sorry
  -- Hint: Similar to above

-- TODO 1.6: Prove (-a) * (-b) = a * b
theorem neg_mul_neg (a b : R) : (-a) * (-b) = a * b :=
  sorry
  -- Hint: Use previous results
  -- (-a) * (-b) = -(a * (-b)) = -(-(a * b)) = a * b

-- TODO 1.7: Prove subtraction property
def sub (a b : R) : R := a + (-b)
infixl:65 " - " => sub

theorem sub_eq_add_neg (a b : R) : a - b = a + (-b) :=
  sorry

-- TODO 1.8: Prove (a - b) * c = a * c - b * c
theorem sub_mul (a b c : R) : (a - b) * c = a * c - b * c :=
  sorry
  -- Hint: Expand using definitions and distributive law

end BasicRingProperties

-- ============================================================================
-- PART 2: SUBRINGS
-- ============================================================================

section Subrings

variable {R : Type} [Ring R]

-- A subring is a subset S ⊆ R that forms a ring under the same operations
structure Subring (R : Type) [Ring R] where
  carrier : Set R
  zero_mem : 0 ∈ carrier
  one_mem : 1 ∈ carrier
  add_mem : ∀ {a b}, a ∈ carrier → b ∈ carrier → a + b ∈ carrier
  neg_mem : ∀ {a}, a ∈ carrier → -a ∈ carrier
  mul_mem : ∀ {a b}, a ∈ carrier → b ∈ carrier → a * b ∈ carrier

-- Notation
instance : SetLike (Subring R) R where
  coe := Subring.carrier
  coe_injective' := sorry  -- Prove subrings equal if carriers equal

-- TODO 2.1: Prove subring criterion
-- S is a subring iff 1 ∈ S and S is closed under -, +, and *
theorem subring_criterion (S : Set R) :
    (1 ∈ S ∧ (∀ a b, a ∈ S → b ∈ S → a - b ∈ S) ∧ (∀ a b, a ∈ S → b ∈ S → a * b ∈ S)) ↔
    ∃ (SR : Subring R), S = SR.carrier :=
  sorry
  -- Forward: Construct subring from criterion
  -- Note: 0 ∈ S follows from 0 = 1 - 1
  -- Backward: Verify criterion from subring properties

-- TODO 2.2: Prove intersection of subrings is a subring
def subring_inter (S T : Subring R) : Subring R where
  carrier := S.carrier ∩ T.carrier
  zero_mem := sorry  -- TODO: 0 ∈ S and 0 ∈ T
  one_mem := sorry   -- TODO: 1 ∈ S and 1 ∈ T
  add_mem := sorry   -- TODO: Use add_mem of both S and T
  neg_mem := sorry   -- TODO: Use neg_mem of both S and T
  mul_mem := sorry   -- TODO: Use mul_mem of both S and T

-- TODO 2.3: Prove {0} is NOT a subring (no identity!)
-- But {0, 1, 2, ...} could be a subring in some rings

end Subrings

-- ============================================================================
-- PART 3: IDEALS
-- ============================================================================

section Ideals

variable {R : Type} [Ring R]

-- An ideal I is a special subset that "absorbs" multiplication
-- I is an ideal if:
-- 1. (I, +) is a subgroup of (R, +)
-- 2. For all r ∈ R and a ∈ I: r * a ∈ I and a * r ∈ I (absorption property)
structure Ideal (R : Type) [Ring R] where
  carrier : Set R
  zero_mem : 0 ∈ carrier
  add_mem : ∀ {a b}, a ∈ carrier → b ∈ carrier → a + b ∈ carrier
  neg_mem : ∀ {a}, a ∈ carrier → -a ∈ carrier
  mul_mem_left : ∀ {a}, a ∈ carrier → ∀ r, r * a ∈ carrier
  mul_mem_right : ∀ {a}, a ∈ carrier → ∀ r, a * r ∈ carrier

-- Notation
instance : SetLike (Ideal R) R where
  coe := Ideal.carrier
  coe_injective' := sorry

-- TODO 3.1: Prove ideal criterion
-- I is an ideal iff 0 ∈ I, closed under + and -, and absorbs multiplication
theorem ideal_criterion (I : Set R) :
    (0 ∈ I ∧ (∀ a b, a ∈ I → b ∈ I → a + b ∈ I) ∧
     (∀ a, a ∈ I → -a ∈ I) ∧
     (∀ a r, a ∈ I → r * a ∈ I ∧ a * r ∈ I)) ↔
    ∃ (IR : Ideal R), I = IR.carrier :=
  sorry
  -- Straightforward verification in both directions

-- TODO 3.2: Every ideal is a subring (if it contains 1)
-- But most interesting ideals do NOT contain 1
theorem ideal_with_one_is_subring (I : Ideal R) (h : 1 ∈ I) : I.carrier = Set.univ :=
  sorry
  -- Hint: If 1 ∈ I, then for any r ∈ R: r = r * 1 ∈ I (by absorption)
  -- So I = R (the whole ring)
  -- This is why proper ideals never contain 1!

-- TODO 3.3: Prove sum of ideals is an ideal
def ideal_add (I J : Ideal R) : Ideal R where
  carrier := {r | ∃ i j, i ∈ I ∧ j ∈ J ∧ r = i + j}
  zero_mem := sorry  -- TODO: 0 = 0 + 0
  add_mem := sorry   -- TODO: (i₁ + j₁) + (i₂ + j₂) = (i₁ + i₂) + (j₁ + j₂)
  neg_mem := sorry   -- TODO: -(i + j) = -i + -j
  mul_mem_left := sorry   -- TODO: r * (i + j) = r * i + r * j, both in respective ideals
  mul_mem_right := sorry

-- TODO 3.4: Prove intersection of ideals is an ideal
def ideal_inter (I J : Ideal R) : Ideal R where
  carrier := I.carrier ∩ J.carrier
  zero_mem := sorry
  add_mem := sorry
  neg_mem := sorry
  mul_mem_left := sorry   -- TODO: If a ∈ I ∩ J, then r * a ∈ I and r * a ∈ J
  mul_mem_right := sorry

-- TODO 3.5: Prove product of ideals is an ideal
-- I * J = {finite sums of i * j | i ∈ I, j ∈ J}
def ideal_mul (I J : Ideal R) : Ideal R where
  carrier := {r | ∃ (n : ℕ) (f : Fin n → R) (g : Fin n → R),
                   (∀ k, f k ∈ I) ∧ (∀ k, g k ∈ J) ∧
                   r = Finset.sum Finset.univ (fun k => f k * g k)}
  zero_mem := sorry  -- TODO: Empty sum is 0
  add_mem := sorry   -- TODO: Concatenate finite sums
  neg_mem := sorry   -- TODO: Negate each term
  mul_mem_left := sorry   -- TODO: r * (Σ iⱼ * jⱼ) = Σ (r * iⱼ) * jⱼ
  mul_mem_right := sorry

-- TODO 3.6: Define principal ideal generated by a
-- (a) = {r * a * s | r, s ∈ R} in general
-- (a) = {r * a | r ∈ R} in commutative rings
def principal_ideal (a : R) : Ideal R where
  carrier := {r | ∃ x y, r = x * a * y}
  zero_mem := sorry  -- TODO: 0 = 0 * a * 0
  add_mem := sorry
  neg_mem := sorry
  mul_mem_left := sorry   -- TODO: r * (x * a * y) = (r * x) * a * y
  mul_mem_right := sorry

end Ideals

-- ============================================================================
-- PART 4: RING HOMOMORPHISMS
-- ============================================================================

section RingHomomorphisms

variable {R S : Type} [Ring R] [Ring S]

-- A ring homomorphism preserves both operations and identities
structure RingHom (R S : Type) [Ring R] [Ring S] where
  toFun : R → S
  map_zero : toFun 0 = 0
  map_one : toFun 1 = 1
  map_add : ∀ a b, toFun (a + b) = toFun a + toFun b
  map_mul : ∀ a b, toFun (a * b) = toFun a * toFun b

-- Notation
infixr:25 " →+* " => RingHom
instance : CoeFun (R →+* S) (fun _ => R → S) := ⟨RingHom.toFun⟩

variable (φ : R →+* S)

-- TODO 4.1: Prove ring homomorphism preserves negation
theorem map_neg (a : R) : φ (-a) = -(φ a) :=
  sorry
  -- Hint: Show φ(-a) + φ(a) = 0
  -- Use: φ(-a) + φ(a) = φ(-a + a) = φ(0) = 0

-- TODO 4.2: Prove ring homomorphism preserves subtraction
theorem map_sub (a b : R) : φ (a - b) = φ a - φ b :=
  sorry
  -- Hint: Use definitions and map_neg

-- TODO 4.3: Define kernel as an ideal (not just a subgroup!)
def ker : Ideal R where
  carrier := {a | φ a = 0}
  zero_mem := sorry  -- TODO: Use map_zero
  add_mem := sorry   -- TODO: φ(a + b) = φ(a) + φ(b) = 0 + 0 = 0
  neg_mem := sorry   -- TODO: Use map_neg
  mul_mem_left := sorry   -- TODO: φ(r * a) = φ(r) * φ(a) = φ(r) * 0 = 0
  mul_mem_right := sorry

-- TODO 4.4: Define image as a subring
def im : Subring S where
  carrier := {b | ∃ a, φ a = b}
  zero_mem := sorry  -- TODO: φ(0) = 0
  one_mem := sorry   -- TODO: φ(1) = 1
  add_mem := sorry   -- TODO: If b₁ = φ(a₁), b₂ = φ(a₂), then b₁ + b₂ = φ(a₁ + a₂)
  neg_mem := sorry   -- TODO: Use map_neg
  mul_mem := sorry

-- TODO 4.5: Prove φ is injective iff ker(φ) = {0}
theorem injective_iff_trivial_ker :
    Function.Injective φ ↔ (ker φ).carrier = {0} :=
  sorry
  -- Forward: If φ injective and φ(a) = 0 = φ(0), then a = 0
  -- Backward: If φ(a) = φ(b), then φ(a - b) = 0, so a - b = 0

-- TODO 4.6: Prove composition of ring homomorphisms
def comp {T : Type} [Ring T] (ψ : S →+* T) : R →+* T where
  toFun := ψ ∘ φ
  map_zero := sorry  -- TODO: Chain the map_zero properties
  map_one := sorry
  map_add := sorry
  map_mul := sorry

end RingHomomorphisms

-- ============================================================================
-- PART 5: QUOTIENT RINGS
-- ============================================================================

section QuotientRings

variable {R : Type} [Ring R]

-- For any ideal I, we can form the quotient ring R/I
-- Elements are cosets a + I = {a + i | i ∈ I}
-- Addition: (a + I) + (b + I) = (a + b) + I
-- Multiplication: (a + I) * (b + I) = (a * b) + I

-- TODO 5.1: Define equivalence relation for quotient
def ideal_setoid (I : Ideal R) : Setoid R where
  r := fun a b => a - b ∈ I
  iseqv := sorry
    -- TODO: Prove reflexive: a - a = 0 ∈ I
    -- TODO: Prove symmetric: if a - b ∈ I then -(a - b) = b - a ∈ I
    -- TODO: Prove transitive: if a - b ∈ I and b - c ∈ I then (a - b) + (b - c) = a - c ∈ I

-- TODO 5.2: Define quotient ring
def QuotientRing (I : Ideal R) : Type := Quotient (ideal_setoid I)

notation:35 R " / " I:34 => QuotientRing I

-- TODO 5.3: Prove addition is well-defined on quotient
-- If a₁ - b₁ ∈ I and a₂ - b₂ ∈ I, then (a₁ + a₂) - (b₁ + b₂) ∈ I
theorem quotient_add_well_defined (I : Ideal R) (a₁ a₂ b₁ b₂ : R) :
    a₁ - b₁ ∈ I → a₂ - b₂ ∈ I → (a₁ + a₂) - (b₁ + b₂) ∈ I :=
  sorry
  -- Hint: (a₁ + a₂) - (b₁ + b₂) = (a₁ - b₁) + (a₂ - b₂)

-- TODO 5.4: Prove multiplication is well-defined on quotient
-- If a₁ - b₁ ∈ I and a₂ - b₂ ∈ I, then a₁ * a₂ - b₁ * b₂ ∈ I
theorem quotient_mul_well_defined (I : Ideal R) (a₁ a₂ b₁ b₂ : R) :
    a₁ - b₁ ∈ I → a₂ - b₂ ∈ I → a₁ * a₂ - b₁ * b₂ ∈ I :=
  sorry
  -- Hint: a₁ * a₂ - b₁ * b₂ = a₁ * a₂ - a₁ * b₂ + a₁ * b₂ - b₁ * b₂
  --                          = a₁ * (a₂ - b₂) + (a₁ - b₁) * b₂
  -- Both terms are in I by absorption property!

-- TODO 5.5: Define canonical projection
def quotient_map (I : Ideal R) : R →+* (R / I) where
  toFun := Quotient.mk (ideal_setoid I)
  map_zero := sorry
  map_one := sorry
  map_add := sorry  -- TODO: Use quotient_add_well_defined
  map_mul := sorry  -- TODO: Use quotient_mul_well_defined

-- TODO 5.6: Prove kernel of quotient map is I
theorem ker_quotient_map (I : Ideal R) :
    (ker (quotient_map I)).carrier = I.carrier :=
  sorry
  -- Hint: a ∈ ker iff Quotient.mk a = 0 iff a - 0 ∈ I iff a ∈ I

end QuotientRings

-- ============================================================================
-- PART 6: INTEGRAL DOMAINS
-- ============================================================================

section IntegralDomains

variable {R : Type} [Ring R]

-- An integral domain is a commutative ring with no zero divisors
-- Zero divisor: a ≠ 0 and b ≠ 0 but a * b = 0
class IntegralDomain (R : Type) extends Ring R where
  mul_comm : ∀ a b : R, mul a b = mul b a
  no_zero_divisors : ∀ a b : R, mul a b = zero → a = zero ∨ b = zero

-- TODO 6.1: Prove cancellation law in integral domains
theorem mul_cancel_left {R : Type} [IntegralDomain R] {a b c : R} (ha : a ≠ 0) :
    a * b = a * c → b = c :=
  sorry
  -- Hint: a * b = a * c implies a * (b - c) = 0
  -- Since a ≠ 0, must have b - c = 0

-- TODO 6.2: Prove product is zero iff one factor is zero
theorem mul_eq_zero_iff {R : Type} [IntegralDomain R] {a b : R} :
    a * b = 0 ↔ a = 0 ∨ b = 0 :=
  sorry
  -- One direction is the axiom, other direction uses zero_mul and mul_zero

-- TODO 6.3: Every field is an integral domain
-- (Will be proven after defining fields)

-- TODO 6.4: Integers ℤ form an integral domain
-- (Classic example)

end IntegralDomains

-- ============================================================================
-- PART 7: PRIME AND MAXIMAL IDEALS
-- ============================================================================

section PrimeAndMaximalIdeals

variable {R : Type} [Ring R]

-- A prime ideal P has the property:
-- If a * b ∈ P, then a ∈ P or b ∈ P
-- (Generalizes prime numbers!)
def IsPrime (P : Ideal R) : Prop :=
  P.carrier ≠ Set.univ ∧ ∀ a b, a * b ∈ P → a ∈ P ∨ b ∈ P

-- A maximal ideal M is not contained in any larger proper ideal
def IsMaximal (M : Ideal R) : Prop :=
  M.carrier ≠ Set.univ ∧ ∀ I : Ideal R, M.carrier ⊆ I.carrier → I.carrier = M.carrier ∨ I.carrier = Set.univ

-- TODO 7.1: Prove R/P is an integral domain iff P is prime
theorem quotient_is_domain_iff_prime (P : Ideal R) :
    (∃ _ : IntegralDomain (R / P), True) ↔ IsPrime P :=
  sorry
  -- Forward: If R/P is domain, then no zero divisors
  -- If (a + P) * (b + P) = 0 + P, then ab ∈ P
  -- Since no zero divisors, a + P = 0 or b + P = 0
  -- So a ∈ P or b ∈ P
  -- Backward: If P is prime and ab ∈ P, then a ∈ P or b ∈ P
  -- So in R/P, if (a + P)(b + P) = 0, then a + P = 0 or b + P = 0

-- TODO 7.2: Prove R/M is a field iff M is maximal
-- (Requires field definition from next section)

-- TODO 7.3: Every maximal ideal is prime
theorem maximal_is_prime (M : Ideal R) (hM : IsMaximal M) : IsPrime M :=
  sorry
  -- Hint: If R/M is a field, it's also an integral domain
  -- Use previous theorem

-- TODO 7.4: Prove (p) is a prime ideal in ℤ iff p is prime
-- Classic connection: ideal theory generalizes prime numbers!

end PrimeAndMaximalIdeals

-- ============================================================================
-- PART 8: CONNECTION TO FIELDS
-- ============================================================================

section ConnectionToFields

-- A field is a commutative ring where every nonzero element has a multiplicative inverse
class Field (F : Type) extends Ring F where
  mul_comm : ∀ a b : F, mul a b = mul b a
  exists_inv : ∀ a : F, a ≠ zero → ∃ b, mul a b = one

-- TODO 8.1: Prove every field is an integral domain
theorem field_is_domain {F : Type} [Field F] : IntegralDomain F where
  mul_comm := sorry  -- Use Field.mul_comm
  no_zero_divisors := sorry
    -- Hint: If a * b = 0 and a ≠ 0, then a has inverse a⁻¹
    -- Multiply both sides: a⁻¹ * (a * b) = a⁻¹ * 0
    -- So (a⁻¹ * a) * b = 0, thus 1 * b = 0, so b = 0

-- TODO 8.2: A field has only two ideals: {0} and F
theorem field_ideals {F : Type} [Field F] (I : Ideal F) :
    I.carrier = {0} ∨ I.carrier = Set.univ :=
  sorry
  -- Hint: If I ≠ {0}, then ∃ a ∈ I with a ≠ 0
  -- Then a has inverse a⁻¹, so 1 = a⁻¹ * a ∈ I
  -- By absorption, every element is in I

-- TODO 8.3: Prove R/M is a field iff M is maximal
theorem quotient_is_field_iff_maximal {R : Type} [Ring R] (M : Ideal R) :
    (∃ _ : Field (R / M), True) ↔ IsMaximal M :=
  sorry
  -- Forward: If R/M is a field, its only ideals are {0} and R/M
  -- By correspondence, M's only containing ideals are M and R
  -- Backward: If M is maximal, show every nonzero element of R/M has inverse
  -- If a ∉ M, consider ideal (M, a) generated by M and a
  -- This must be R (by maximality), so 1 ∈ (M, a)
  -- Thus 1 = m + r * a for some m ∈ M, r ∈ R
  -- In R/M: 1 = r * a, so r is inverse of a

-- TODO 8.4: Field of fractions (localization)
-- Every integral domain embeds in a field (its field of fractions)
-- Example: ℤ → ℚ

end ConnectionToFields

/-
IMPLEMENTATION GUIDE SUMMARY:

Part 1 - Basic Ring Properties:
  Key insight: 0 * a = 0 requires a clever trick
  Start with 0 = 0 + 0, distribute, then cancel
  Many properties follow from distributivity
  Negation interacts with multiplication in predictable ways
  
Part 2 - Subrings:
  Similar to subgroups but with two operations
  Must contain both 0 and 1
  Less important than ideals in ring theory
  Subrings don't lead to quotient structures
  
Part 3 - Ideals:
  THE KEY CONCEPT in ring theory!
  Absorption property: r * I ⊆ I for all r
  Ideals are kernels of ring homomorphisms
  Enable quotient ring construction
  Operations: sum, intersection, product
  Principal ideals: generated by single element
  
Part 4 - Ring Homomorphisms:
  Preserve both operations and both identities
  Kernel is an IDEAL (not just subgroup)
  Image is a subring
  Injective iff kernel is {0}
  Composition works as expected
  
Part 5 - Quotient Rings:
  R/I is a ring for any ideal I
  Well-definedness relies on absorption property
  This is WHY we need ideals (not just subrings)!
  Canonical projection: R → R/I
  Kernel of projection is exactly I
  
Part 6 - Integral Domains:
  Commutative rings with no zero divisors
  Cancellation law holds
  Examples: ℤ, ℤ[x], any field
  Bridge between general rings and fields
  
Part 7 - Prime and Maximal Ideals:
  Prime ideals ↔ quotients are integral domains
  Maximal ideals ↔ quotients are fields
  Every maximal ideal is prime
  Generalizes prime number theory
  Crucial for algebraic geometry
  
Part 8 - Connection to Fields:
  Fields are "smallest" algebraic structures with division
  Every field is an integral domain
  Fields have only trivial ideals
  Quotient by maximal ideal gives field
  Every integral domain embeds in a field

KEY PROOF TECHNIQUES:
- Strategic use of distributive laws
- Additive and multiplicative cancellation
- Absorption property of ideals
- Well-definedness for quotients
- Ideal correspondence theorems
- Prime vs maximal characterizations

CONNECTION TO GALOIS THEORY:
- Field extensions E/F are quotients of polynomial rings
- Minimal polynomial generates maximal ideal
- R/M is a field when M is maximal
- This constructs field extensions algebraically
- Quotient rings model algebraic elements

EXAMPLES TO REMEMBER:
- ℤ is an integral domain (not a field)
- ℤ/nℤ is a ring for any n
- ℤ/pℤ is a field iff p is prime
- (p) is a prime ideal in ℤ iff p is prime
- ℚ, ℝ, ℂ are fields
- Polynomial rings R[x] are integral domains if R is

NEXT STEPS:
After Rings.lean, move to Fields.lean
Then study Polynomials.lean (polynomial rings)
This leads to field extensions and Galois theory
The quotient ring R[x]/(p(x)) constructs field extensions!
-/
