-- Field Theory in Lean - Template
-- This file guides you through the fundamentals of field theory
-- Fields are the foundation of Galois theory!

/-
LEARNING OBJECTIVES:
1. Understand field axioms and structure (commutative ring with inverses)
2. Prove basic field properties and operations
3. Work with subfields and field extensions
4. Study field homomorphisms (all injective!)
5. Understand field characteristic and prime fields
6. Learn about field extensions and their degrees
7. Explore examples: ℚ, 𝔽_p, algebraic extensions
8. Build foundation for Galois theory
-/

-- ============================================================================
-- PART 1: FIELD DEFINITION AND BASIC PROPERTIES
-- ============================================================================

-- A field is a commutative ring where every non-zero element has a multiplicative inverse
-- Field axioms:
-- Addition: (F, +, 0) is an abelian group
-- Multiplication: (F \ {0}, *, 1) is an abelian group
-- Distributivity: a * (b + c) = a * b + a * c

class Field (F : Type) where
  -- Addition structure
  add : F → F → F
  zero : F
  neg : F → F
  -- Multiplication structure
  mul : F → F → F
  one : F
  inv : F → F  -- defined only for non-zero elements
  -- Addition axioms (abelian group)
  add_assoc : ∀ a b c, add (add a b) c = add a (add b c)
  zero_add : ∀ a, add zero a = a
  add_zero : ∀ a, add a zero = a
  add_left_neg : ∀ a, add (neg a) a = zero
  add_comm : ∀ a b, add a b = add b a
  -- Multiplication axioms (abelian group on non-zero elements)
  mul_assoc : ∀ a b c, mul (mul a b) c = mul a (mul b c)
  one_mul : ∀ a, mul one a = a
  mul_one : ∀ a, mul a one = a
  mul_comm : ∀ a b, mul a b = mul b a
  mul_left_inv : ∀ a, a ≠ zero → mul (inv a) a = one
  -- Distributivity
  left_distrib : ∀ a b c, mul a (add b c) = add (mul a b) (mul a c)
  -- Non-triviality
  zero_ne_one : zero ≠ one

-- Notation
infixl:65 " + " => Field.add
infixl:70 " * " => Field.mul
notation "0" => Field.zero
notation "1" => Field.one
prefix:100 "-" => Field.neg
postfix:max "⁻¹" => Field.inv

section BasicFieldProperties

variable {F : Type} [Field F]

-- TODO 1.1: Prove right distributivity
-- We have left distributivity a * (b + c) = a * b + a * c
-- Need to prove (a + b) * c = a * c + b * c
theorem right_distrib (a b c : F) : (a + b) * c = a * c + b * c :=
  sorry
  -- Hint: Use commutativity of multiplication and left_distrib
  -- (a + b) * c = c * (a + b) = c * a + c * b = a * c + b * c

-- TODO 1.2: Prove multiplication by zero
theorem mul_zero (a : F) : a * 0 = 0 :=
  sorry
  -- Hint: a * 0 = a * (0 + 0) = a * 0 + a * 0
  -- Subtract a * 0 from both sides
  -- Use: if x = x + x, then x = 0

-- TODO 1.3: Prove zero times anything is zero
theorem zero_mul (a : F) : 0 * a = 0 :=
  sorry
  -- Hint: Use commutativity and mul_zero

-- TODO 1.4: Prove right inverse for non-zero elements
theorem mul_right_inv {a : F} (h : a ≠ 0) : a * a⁻¹ = 1 :=
  sorry
  -- Hint: Use commutativity and mul_left_inv

-- TODO 1.5: Prove negation distributes over multiplication
theorem neg_mul (a b : F) : -(a * b) = (-a) * b :=
  sorry
  -- Hint: Show (-a) * b is the additive inverse of a * b
  -- Verify: a * b + (-a) * b = (a + (-a)) * b = 0 * b = 0

-- TODO 1.6: Prove multiplication of negatives
theorem neg_mul_neg (a b : F) : (-a) * (-b) = a * b :=
  sorry
  -- Hint: (-a) * (-b) = -(a * (-b)) = -(-(a * b)) = a * b
  -- Need to prove: -(-(x)) = x

-- TODO 1.7: Prove no zero divisors (integral domain property)
-- If a * b = 0 and a ≠ 0, then b = 0
theorem no_zero_divisors {a b : F} (ha : a ≠ 0) : a * b = 0 → b = 0 :=
  sorry
  -- Hint: If a * b = 0, multiply both sides by a⁻¹
  -- a⁻¹ * (a * b) = a⁻¹ * 0
  -- (a⁻¹ * a) * b = 0
  -- 1 * b = 0
  -- b = 0

-- TODO 1.8: Prove inverse is unique
theorem inv_unique {a b : F} (ha : a ≠ 0) (h : b * a = 1) : b = a⁻¹ :=
  sorry
  -- Hint: Multiply b * a = 1 by a⁻¹ on the right
  -- b * a * a⁻¹ = 1 * a⁻¹
  -- b * (a * a⁻¹) = a⁻¹
  -- b * 1 = a⁻¹

-- TODO 1.9: Prove inverse of inverse
theorem inv_inv {a : F} (ha : a ≠ 0) : (a⁻¹)⁻¹ = a :=
  sorry
  -- Hint: Use uniqueness of inverse
  -- Show a is the inverse of a⁻¹

-- TODO 1.10: Prove one is its own inverse
theorem inv_one : (1 : F)⁻¹ = 1 :=
  sorry
  -- Hint: 1 * 1 = 1, use uniqueness

-- TODO 1.11: Prove inverse of product
theorem mul_inv {a b : F} (ha : a ≠ 0) (hb : b ≠ 0) :
    (a * b)⁻¹ = b⁻¹ * a⁻¹ :=
  sorry
  -- Hint: Show b⁻¹ * a⁻¹ is the inverse of a * b
  -- (a * b) * (b⁻¹ * a⁻¹) = a * (b * b⁻¹) * a⁻¹ = a * 1 * a⁻¹ = 1

end BasicFieldProperties

-- ============================================================================
-- PART 2: DIVISION AND FIELD OPERATIONS
-- ============================================================================

section Division

variable {F : Type} [Field F]

-- Define division: a / b = a * b⁻¹
def div (a b : F) : F := a * b⁻¹

infixl:70 " / " => div

-- TODO 2.1: Prove a / a = 1 for non-zero a
theorem div_self {a : F} (ha : a ≠ 0) : a / a = 1 :=
  sorry
  -- Hint: a / a = a * a⁻¹ = 1

-- TODO 2.2: Prove cancellation in division
theorem div_mul_cancel {a b : F} (hb : b ≠ 0) : (a / b) * b = a :=
  sorry
  -- Hint: (a / b) * b = (a * b⁻¹) * b = a * (b⁻¹ * b) = a * 1 = a

-- TODO 2.3: Prove division by one
theorem div_one (a : F) : a / 1 = a :=
  sorry
  -- Hint: a / 1 = a * 1⁻¹ = a * 1 = a

-- TODO 2.4: Prove one divided by a
theorem one_div {a : F} (ha : a ≠ 0) : 1 / a = a⁻¹ :=
  sorry
  -- Hint: 1 / a = 1 * a⁻¹ = a⁻¹

-- TODO 2.5: Prove division of divisions
theorem div_div {a b c : F} (hb : b ≠ 0) (hc : c ≠ 0) :
    (a / b) / c = a / (b * c) :=
  sorry
  -- Hint: Use mul_inv and associativity

-- TODO 2.6: Prove multiplication of divisions
theorem mul_div {a b c d : F} (hb : b ≠ 0) (hd : d ≠ 0) :
    (a / b) * (c / d) = (a * c) / (b * d) :=
  sorry
  -- Hint: Expand definitions and use mul_inv

end Division

-- ============================================================================
-- PART 3: SUBFIELDS
-- ============================================================================

section Subfields

variable {F : Type} [Field F]

-- A subfield is a subset K ⊆ F that is closed under field operations
structure Subfield (F : Type) [Field F] where
  carrier : Set F
  zero_mem : 0 ∈ carrier
  one_mem : 1 ∈ carrier
  add_mem : ∀ {a b}, a ∈ carrier → b ∈ carrier → a + b ∈ carrier
  neg_mem : ∀ {a}, a ∈ carrier → -a ∈ carrier
  mul_mem : ∀ {a b}, a ∈ carrier → b ∈ carrier → a * b ∈ carrier
  inv_mem : ∀ {a}, a ∈ carrier → a ≠ 0 → a⁻¹ ∈ carrier

-- TODO 3.1: Prove intersection of subfields is a subfield
def subfield_inter (K L : Subfield F) : Subfield F where
  carrier := K.carrier ∩ L.carrier
  zero_mem := sorry  -- TODO: 0 ∈ K and 0 ∈ L
  one_mem := sorry   -- TODO: 1 ∈ K and 1 ∈ L
  add_mem := sorry   -- TODO: Use add_mem of both K and L
  neg_mem := sorry   -- TODO: Use neg_mem of both K and L
  mul_mem := sorry   -- TODO: Use mul_mem of both K and L
  inv_mem := sorry   -- TODO: Use inv_mem of both K and L

-- TODO 3.2: Prove trivial subfield criterion
-- The smallest subfield is the prime field (see Part 5)

end Subfields

-- ============================================================================
-- PART 4: FIELD HOMOMORPHISMS (Always Injective!)
-- ============================================================================

section FieldHomomorphisms

variable {F K : Type} [Field F] [Field K]

-- A field homomorphism preserves addition and multiplication
structure FieldHom (F K : Type) [Field F] [Field K] where
  toFun : F → K
  map_zero : toFun 0 = 0
  map_one : toFun 1 = 1
  map_add : ∀ a b, toFun (a + b) = toFun a + toFun b
  map_mul : ∀ a b, toFun (a * b) = toFun a * toFun b

-- Notation
infixr:25 " →+* " => FieldHom
instance : CoeFun (F →+* K) (fun _ => F → K) := ⟨FieldHom.toFun⟩

variable (φ : F →+* K)

-- TODO 4.1: Prove homomorphism preserves negation
theorem map_neg (a : F) : φ (-a) = -(φ a) :=
  sorry
  -- Hint: φ(a) + φ(-a) = φ(a + (-a)) = φ(0) = 0
  -- So φ(-a) is the additive inverse of φ(a)

-- TODO 4.2: Prove homomorphism preserves subtraction
theorem map_sub (a b : F) : φ (a + (-b)) = φ a + (-(φ b)) :=
  sorry
  -- Hint: Use map_add and map_neg

-- TODO 4.3: Prove homomorphism preserves inverses (for non-zero elements)
theorem map_inv {a : F} (ha : a ≠ 0) : φ (a⁻¹) = (φ a)⁻¹ :=
  sorry
  -- Hint: φ(a) * φ(a⁻¹) = φ(a * a⁻¹) = φ(1) = 1
  -- So φ(a⁻¹) is the multiplicative inverse of φ(a)
  -- Need to show φ(a) ≠ 0 (use injectivity!)

-- TODO 4.4: IMPORTANT - Prove field homomorphisms are injective
-- This is a KEY difference from ring homomorphisms!
theorem field_hom_injective : Function.Injective φ :=
  sorry
  -- Hint: Suppose φ(a) = φ(b)
  -- Then φ(a - b) = φ(a) - φ(b) = 0
  -- If a ≠ b, then a - b ≠ 0
  -- But then φ((a-b)⁻¹) = φ(a-b)⁻¹ = 0⁻¹ which is undefined!
  -- Contradiction. So a = b.
  -- Alternative: kernel of ring hom is an ideal, only ideals in field are {0} and F

-- TODO 4.5: Prove composition of field homomorphisms
def comp {L : Type} [Field L] (ψ : K →+* L) : F →+* L where
  toFun := ψ ∘ φ
  map_zero := sorry
  map_one := sorry
  map_add := sorry
  map_mul := sorry

-- TODO 4.6: Prove identity is a field homomorphism
def id : F →+* F where
  toFun := fun x => x
  map_zero := sorry
  map_one := sorry
  map_add := sorry
  map_mul := sorry

end FieldHomomorphisms

-- ============================================================================
-- PART 5: FIELD CHARACTERISTIC
-- ============================================================================

section Characteristic

variable {F : Type} [Field F]

-- The characteristic is the smallest n > 0 such that n·1 = 0,
-- or 0 if no such n exists
-- n·a means a + a + ... + a (n times)

def natMul (n : ℕ) (a : F) : F :=
  match n with
  | 0 => 0
  | n + 1 => a + natMul n a

notation:70 n:70 " ·· " a:71 => natMul n a

-- TODO 5.1: Prove characteristic of field is 0 or prime
-- If char(F) = n > 0, then n is prime
theorem char_is_zero_or_prime :
    (∀ n : ℕ, n > 0 → n ·· (1 : F) ≠ 0) ∨
    (∃ p : ℕ, Nat.Prime p ∧ p ·· (1 : F) = 0 ∧ ∀ m < p, m > 0 → m ·· (1 : F) ≠ 0) :=
  sorry
  -- Hint: Suppose char(F) = n = ab where 1 < a, b < n
  -- Then n·1 = 0, so (a·1) * (b·1) = 0
  -- But F has no zero divisors!
  -- So a·1 = 0 or b·1 = 0, contradicting minimality of n

-- TODO 5.2: Prove if char(F) = p > 0, then p·a = 0 for all a
theorem char_mul_all (p : ℕ) (hp : Nat.Prime p) (h : p ·· (1 : F) = 0) :
    ∀ a : F, p ·· a = 0 :=
  sorry
  -- Hint: p·a = (p·1) * a = 0 * a = 0

end Characteristic

-- ============================================================================
-- PART 6: PRIME FIELDS
-- ============================================================================

section PrimeFields

-- The prime field is the smallest subfield of F
-- If char(F) = 0, prime field ≅ ℚ
-- If char(F) = p, prime field ≅ 𝔽_p (field with p elements)

-- TODO 6.1: Define the prime subfield
-- For char 0: {m/n : m ∈ ℤ, n ∈ ℕ, n ≠ 0}
-- For char p: {0, 1, 2, ..., p-1} where k means k·1

-- TODO 6.2: Prove every field contains a unique prime field

-- TODO 6.3: Example - Finite field 𝔽_p
-- When p is prime, ℤ/pℤ is a field (𝔽_p)
axiom finite_field_exists (p : ℕ) (hp : Nat.Prime p) :
  ∃ (Fp : Type) (_ : Field Fp) (_ : Fintype Fp),
    Fintype.card Fp = p

end PrimeFields

-- ============================================================================
-- PART 7: FIELD EXTENSIONS - INTRODUCTION
-- ============================================================================

section FieldExtensions

variable {F K : Type} [Field F] [Field K]

-- A field extension E/F means F ⊆ E (F is a subfield of E)
-- We model this with an injective field homomorphism φ : F → E

structure FieldExtension (F E : Type) [Field F] [Field E] where
  embedding : F →+* E
  injective : Function.Injective embedding  -- Automatic by 4.4!

notation:25 E:25 "/" F:26 => FieldExtension F E

-- TODO 7.1: Prove identity extension F/F
def id_extension (F : Type) [Field F] : F / F where
  embedding := FieldHom.id
  injective := sorry

-- TODO 7.2: Prove extension is transitive
-- If L/K and K/F, then L/F
def trans_extension {F K L : Type} [Field F] [Field K] [Field L]
    (φ : K / F) (ψ : L / K) : L / F where
  embedding := FieldHom.comp φ.embedding ψ.embedding
  injective := sorry  -- TODO: Composition of injections is injective

-- TODO 7.3: Every field extension makes E a vector space over F
-- This is fundamental for defining degree!

end FieldExtensions

-- ============================================================================
-- PART 8: DEGREE OF FIELD EXTENSIONS
-- ============================================================================

section Degree

-- The degree [E : F] is the dimension of E as a vector space over F
-- [E : F] can be finite or infinite

variable {F E : Type} [Field F] [Field E] (φ : E / F)

-- For finite extensions, degree is a natural number
-- TODO 8.1: Define degree for finite extensions
-- This requires vector space theory, which we axiomatize here

axiom degree (F E : Type) [Field F] [Field E] (φ : E / F) : ℕ ⊕ Unit
  -- Returns ℕ for finite extensions, Unit for infinite

-- Notation
notation:25 "[" E ":" F "]" => degree F E sorry

-- TODO 8.2: Prove multiplicativity of degrees (Tower Law)
-- If L/K and K/F, then [L : F] = [L : K] * [K : F]
axiom degree_mul {F K L : Type} [Field F] [Field K] [Field L]
    (φ : K / F) (ψ : L / K) :
    ∃ (d₁ d₂ d₃ : ℕ), d₃ = d₁ * d₂
  -- Proof idea: If {e₁, ..., eₘ} is basis of K/F and
  -- {f₁, ..., fₙ} is basis of L/K, then
  -- {eᵢ * fⱼ} is basis of L/F

-- TODO 8.3: Prove [F : F] = 1
axiom degree_self (F : Type) [Field F] : ∃ n : ℕ, n = 1
  -- Basis is {1}

-- TODO 8.4: Examples of degrees
-- [ℂ : ℝ] = 2 (basis: {1, i})
-- [ℝ : ℚ] = ∞ (infinite dimensional)
-- [ℚ(√2) : ℚ] = 2 (basis: {1, √2})
-- [ℚ(∛2) : ℚ] = 3 (basis: {1, ∛2, ∛4})

end Degree

-- ============================================================================
-- PART 9: ALGEBRAIC AND TRANSCENDENTAL ELEMENTS
-- ============================================================================

section AlgebraicElements

variable {F E : Type} [Field F] [Field E] (φ : E / F)

-- An element α ∈ E is algebraic over F if it's a root of some
-- non-zero polynomial with coefficients in F
-- Otherwise, α is transcendental over F

-- TODO 9.1: Define algebraic element
-- α is algebraic if ∃ polynomial p ≠ 0 with coefficients in F such that p(α) = 0

def IsAlgebraic (α : E) : Prop :=
  ∃ (p : List F), p ≠ [] ∧ sorry  -- TODO: Define polynomial evaluation
  -- Need polynomial ring F[X] and evaluation map

-- TODO 9.2: Prove if [E : F] is finite, every element of E is algebraic over F
axiom finite_ext_algebraic :
    (∃ n : ℕ, sorry) →  -- [E : F] = n
    ∀ α : E, IsAlgebraic φ α
  -- Proof: If [E : F] = n, then {1, α, α², ..., αⁿ} are n+1 elements
  -- So they're linearly dependent: c₀ + c₁α + ... + cₙαⁿ = 0
  -- This gives a polynomial p(x) = c₀ + c₁x + ... + cₙxⁿ with p(α) = 0

-- TODO 9.3: Examples
-- √2 is algebraic over ℚ: x² - 2 = 0
-- i is algebraic over ℝ: x² + 1 = 0
-- π is transcendental over ℚ (hard to prove!)
-- e is transcendental over ℚ (hard to prove!)

end AlgebraicElements

-- ============================================================================
-- PART 10: SIMPLE EXTENSIONS
-- ============================================================================

section SimpleExtensions

variable {F : Type} [Field F]

-- A simple extension F(α) is the smallest field containing F and α
-- If α is algebraic with minimal polynomial of degree n, then [F(α) : F] = n
-- If α is transcendental, then F(α) ≅ F(x) (field of rational functions)

-- TODO 10.1: Define simple extension F(α)
-- This is the field generated by F and {α}

-- TODO 10.2: Prove primitive element theorem (advanced)
-- Every finite separable extension is a simple extension
-- E/F finite and separable ⟹ ∃ α : E such that E = F(α)

end SimpleExtensions

/-
IMPLEMENTATION GUIDE SUMMARY:

Part 1 - Field Definition:
  Field = commutative ring + multiplicative inverses for non-zero elements
  Key difference from rings: can divide by non-zero elements
  No zero divisors! If ab = 0, then a = 0 or b = 0
  Many proofs use: multiply by inverse to cancel
  
Part 2 - Division:
  Division defined as a/b = a * b⁻¹
  Natural properties: (a/b) * b = a, a/a = 1
  Division algebra follows from field axioms
  
Part 3 - Subfields:
  Subfield = subset closed under all field operations
  Intersection of subfields is a subfield
  Important for Galois theory: intermediate fields!
  
Part 4 - Field Homomorphisms:
  KEY FACT: All field homomorphisms are injective!
  This is because fields have no non-trivial ideals
  Kernel of field hom is {0} (only ideal in a field besides the field itself)
  Major difference from ring homomorphisms
  Consequence: "field homomorphism" = "field embedding"
  
Part 5 - Characteristic:
  Characteristic = smallest n such that n·1 = 0, or 0 if none exists
  THEOREM: char(F) is either 0 or prime
  Proof uses: F has no zero divisors
  If char(F) = p, then p·a = 0 for all a ∈ F
  
Part 6 - Prime Fields:
  Every field contains a unique smallest subfield (prime field)
  If char(F) = 0: prime field ≅ ℚ
  If char(F) = p: prime field ≅ 𝔽_p
  𝔽_p = {0, 1, ..., p-1} with arithmetic mod p
  
Part 7 - Field Extensions:
  Extension E/F means F ⊆ E (F is subfield of E)
  Model with injective homomorphism F → E
  Extensions compose: if L/K and K/F, then L/F
  Foundation for Galois theory!
  
Part 8 - Degree of Extensions:
  [E : F] = dimension of E as vector space over F
  Can be finite (n ∈ ℕ) or infinite
  TOWER LAW: [L : F] = [L : K] × [K : F]
  Examples: [ℂ : ℝ] = 2, [ℚ(√2) : ℚ] = 2
  
Part 9 - Algebraic Elements:
  α is algebraic over F if it satisfies some polynomial equation
  Transcendental = not algebraic
  If [E : F] finite, all elements algebraic
  Examples: √2, i algebraic over ℚ; π, e transcendental
  
Part 10 - Simple Extensions:
  F(α) = smallest field containing F and α
  If α algebraic: [F(α) : F] = degree of minimal polynomial
  If α transcendental: F(α) ≅ F(x) (rational functions)
  Primitive element theorem: finite separable = simple extension

KEY PROOF TECHNIQUES:
- Multiply by inverses to cancel and isolate terms
- Use injectivity of field homomorphisms
- No zero divisors property: ab = 0 ⟹ a = 0 or b = 0
- Vector space dimension arguments for degrees
- Polynomial evaluation and minimal polynomials

CONNECTION TO GALOIS THEORY:
- Galois group Gal(E/F) consists of field automorphisms
- Fundamental theorem: intermediate fields ↔ subgroups of Gal(E/F)
- Degree of extension relates to order of Galois group
- Normal extensions ↔ all conjugates included
- Separable extensions ↔ no repeated roots in char p

NEXT STEPS:
After mastering Fields.lean, you're ready for:
1. Polynomial rings F[X] and irreducibility
2. Algebraic closures and splitting fields
3. Galois extensions and the Galois group
4. Fundamental theorem of Galois theory
5. Applications: solvability by radicals, ruler-and-compass constructions

FURTHER READING:
- Dummit & Foote: Abstract Algebra (Chapters 13-14)
- Lang: Algebra (Chapter V-VI)
- Artin: Algebra (Chapter 13)
- Milne: Fields and Galois Theory (online notes)
-/
