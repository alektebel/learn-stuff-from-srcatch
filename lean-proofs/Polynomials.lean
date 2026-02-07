-- Polynomial Theory in Lean - Template
-- This file guides you through the theory of polynomials over fields
-- Polynomials are ESSENTIAL for field extensions and Galois theory!

/-
LEARNING OBJECTIVES:
1. Understand polynomials as finitely supported functions ℕ → F
2. Define and prove properties of polynomial operations
3. Construct the polynomial ring F[X] and prove it's a ring
4. Implement and prove the division algorithm for polynomials
5. Prove the remainder theorem and factor theorem
6. Study irreducible polynomials and their properties
7. Prove unique factorization in F[X]
8. Apply Eisenstein's criterion for irreducibility
9. Understand polynomial evaluation and the relationship to roots
10. Connect polynomials to field extensions (minimal polynomials)

WHY THIS MATTERS FOR GALOIS THEORY:
- Minimal polynomials characterize algebraic elements
- Irreducible polynomials generate field extensions
- Roots of polynomials live in extension fields
- Degree of minimal polynomial = dimension of extension
- Galois groups permute roots of polynomials
-/

-- ============================================================================
-- PART 1: POLYNOMIAL DEFINITION AND BASIC STRUCTURE
-- ============================================================================

-- A polynomial is a function ℕ → F with finite support
-- p(0), p(1), p(2), ... are the coefficients
-- p(n) is the coefficient of Xⁿ

structure Polynomial (F : Type) [Field F] where
  coeff : ℕ → F  -- coefficient function
  support_finite : ∃ N : ℕ, ∀ n ≥ N, coeff n = 0

-- Notation
notation:max F "[X]" => Polynomial F

section PolynomialBasics

variable {F : Type} [Field F]

-- TODO 1.1: Define the zero polynomial
def Polynomial.zero : F[X] where
  coeff := fun _ => 0
  support_finite := sorry
  -- Hint: Choose N = 0, all coefficients are 0

instance : Zero F[X] := ⟨Polynomial.zero⟩

-- TODO 1.2: Define constant polynomial
def Polynomial.const (c : F) : F[X] where
  coeff := fun n => if n = 0 then c else 0
  support_finite := sorry
  -- Hint: Choose N = 1, coefficients 0 for n ≥ 1

-- TODO 1.3: Define the variable X (the polynomial representing x)
def Polynomial.X : F[X] where
  coeff := fun n => if n = 1 then 1 else 0
  support_finite := sorry
  -- Hint: Choose N = 2

-- TODO 1.4: Define polynomial equality
-- Two polynomials are equal iff all coefficients are equal
theorem Polynomial.ext {p q : F[X]} : 
    (∀ n, p.coeff n = q.coeff n) → p = q :=
  sorry
  -- Hint: Use structure equality
  -- Only coeff function matters (support_finite is a proof)

-- TODO 1.5: Characterize when polynomial is zero
theorem Polynomial.eq_zero_iff (p : F[X]) :
    p = 0 ↔ ∀ n, p.coeff n = 0 :=
  sorry
  -- Hint: Use Polynomial.ext

end PolynomialBasics

-- ============================================================================
-- PART 2: POLYNOMIAL OPERATIONS
-- ============================================================================

section PolynomialOperations

variable {F : Type} [Field F]

-- TODO 2.1: Define polynomial addition
def Polynomial.add (p q : F[X]) : F[X] where
  coeff := fun n => p.coeff n + q.coeff n
  support_finite := sorry
  -- Hint: If p has support ≤ N₁ and q has support ≤ N₂,
  -- then p + q has support ≤ max(N₁, N₂)

instance : Add F[X] := ⟨Polynomial.add⟩

-- TODO 2.2: Prove addition is associative
theorem Polynomial.add_assoc (p q r : F[X]) : 
    (p + q) + r = p + (q + r) :=
  sorry
  -- Hint: Use Polynomial.ext
  -- Show coefficients equal using field addition associativity

-- TODO 2.3: Prove addition is commutative
theorem Polynomial.add_comm (p q : F[X]) : p + q = q + p :=
  sorry
  -- Hint: Use Polynomial.ext and field addition commutativity

-- TODO 2.4: Prove 0 is additive identity
theorem Polynomial.zero_add (p : F[X]) : 0 + p = p :=
  sorry
  -- Hint: Show (0 + p).coeff n = p.coeff n for all n

theorem Polynomial.add_zero (p : F[X]) : p + 0 = p :=
  sorry

-- TODO 2.5: Define polynomial negation
def Polynomial.neg (p : F[X]) : F[X] where
  coeff := fun n => -p.coeff n
  support_finite := sorry
  -- Hint: If p has support ≤ N, so does -p

instance : Neg F[X] := ⟨Polynomial.neg⟩

-- TODO 2.6: Prove additive inverse
theorem Polynomial.add_left_neg (p : F[X]) : -p + p = 0 :=
  sorry
  -- Hint: Show coefficients are all 0 using field properties

-- TODO 2.7: Define polynomial multiplication
-- (p * q).coeff(n) = Σᵢ₊ⱼ₌ₙ p.coeff(i) * q.coeff(j)
def Polynomial.mul (p q : F[X]) : F[X] where
  coeff := fun n => Finset.sum (Finset.range (n + 1)) 
                      (fun i => p.coeff i * q.coeff (n - i))
  support_finite := sorry
  -- Hint: If p has support ≤ N₁ and q has support ≤ N₂,
  -- then p * q has support ≤ N₁ + N₂
  -- For n ≥ N₁ + N₂, all terms in sum are 0

instance : Mul F[X] := ⟨Polynomial.mul⟩

-- TODO 2.8: Prove 1 (constant polynomial) is multiplicative identity
instance : One F[X] := ⟨Polynomial.const 1⟩

theorem Polynomial.one_mul (p : F[X]) : 1 * p = p :=
  sorry
  -- Hint: Show coefficients equal
  -- (1 * p).coeff(n) = Σᵢ₊ⱼ₌ₙ 1.coeff(i) * p.coeff(j)
  -- Only i = 0 contributes (since 1.coeff(i) = 0 for i > 0)

theorem Polynomial.mul_one (p : F[X]) : p * 1 = p :=
  sorry

-- TODO 2.9: Prove multiplication is associative
theorem Polynomial.mul_assoc (p q r : F[X]) : 
    (p * q) * r = p * (q * r) :=
  sorry
  -- Hint: This is tedious algebra with finite sums
  -- Show coefficients equal using distributivity of field multiplication
  -- Both sides compute Σᵢ₊ⱼ₊ₖ₌ₙ p.coeff(i) * q.coeff(j) * r.coeff(k)

-- TODO 2.10: Prove distributivity
theorem Polynomial.mul_add (p q r : F[X]) : 
    p * (q + r) = p * q + p * r :=
  sorry
  -- Hint: Use Polynomial.ext and field distributivity
  -- Σᵢ p.coeff(i) * (q + r).coeff(n-i) 
  -- = Σᵢ p.coeff(i) * (q.coeff(n-i) + r.coeff(n-i))
  -- = Σᵢ (p.coeff(i) * q.coeff(n-i) + p.coeff(i) * r.coeff(n-i))

theorem Polynomial.add_mul (p q r : F[X]) : 
    (p + q) * r = p * r + q * r :=
  sorry

-- TODO 2.11: Prove multiplication is commutative
theorem Polynomial.mul_comm (p q : F[X]) : p * q = q * p :=
  sorry
  -- Hint: Show coefficients equal
  -- Σᵢ p.coeff(i) * q.coeff(n-i) = Σⱼ q.coeff(j) * p.coeff(n-j)
  -- Use change of variables and field commutativity

end PolynomialOperations

-- ============================================================================
-- PART 3: POLYNOMIAL RING STRUCTURE
-- ============================================================================

section PolynomialRing

variable {F : Type} [Field F]

-- TODO 3.1: Prove F[X] is a commutative ring
instance : CommRing F[X] where
  add := Polynomial.add
  add_assoc := Polynomial.add_assoc
  zero := Polynomial.zero
  zero_add := Polynomial.zero_add
  add_zero := Polynomial.add_zero
  neg := Polynomial.neg
  add_left_neg := Polynomial.add_left_neg
  add_comm := Polynomial.add_comm
  mul := Polynomial.mul
  mul_assoc := Polynomial.mul_assoc
  one := Polynomial.const 1
  one_mul := Polynomial.one_mul
  mul_one := Polynomial.mul_one
  left_distrib := Polynomial.mul_add
  right_distrib := Polynomial.add_mul
  mul_comm := Polynomial.mul_comm

-- TODO 3.2: Define degree of polynomial
-- degree is the largest n with non-zero coefficient
def Polynomial.degree (p : F[X]) : WithBot ℕ :=
  if h : p = 0 then ⊥ 
  else ↑(Nat.find sorry)  -- TODO: prove ∃ n, p.coeff n ≠ 0 ∧ ∀ m > n, p.coeff m = 0
  -- WithBot ℕ means ℕ ∪ {-∞}, degree of 0 is -∞

-- TODO 3.3: Define leading coefficient
def Polynomial.leadingCoeff (p : F[X]) : F :=
  match p.degree with
  | ⊥ => 0  -- zero polynomial
  | ↑n => p.coeff n

-- TODO 3.4: Prove degree properties
theorem Polynomial.degree_zero : (0 : F[X]).degree = ⊥ :=
  sorry
  -- Hint: Direct from definition

theorem Polynomial.degree_const {c : F} (hc : c ≠ 0) : 
    (Polynomial.const c).degree = ↑0 :=
  sorry
  -- Hint: Coefficient at 0 is c ≠ 0, all others are 0

theorem Polynomial.degree_X : (Polynomial.X : F[X]).degree = ↑1 :=
  sorry
  -- Hint: Coefficient at 1 is 1 ≠ 0, all others are 0

-- TODO 3.5: Prove degree of sum
theorem Polynomial.degree_add_le (p q : F[X]) : 
    (p + q).degree ≤ max p.degree q.degree :=
  sorry
  -- Hint: Coefficients of p + q at n are p.coeff(n) + q.coeff(n)
  -- If both are 0, sum is 0
  -- Degree can decrease if leading terms cancel!

-- TODO 3.6: Prove degree of product (KEY THEOREM!)
theorem Polynomial.degree_mul (p q : F[X]) (hp : p ≠ 0) (hq : q ≠ 0) :
    (p * q).degree = p.degree + q.degree :=
  sorry
  -- Hint: If deg(p) = m and deg(q) = n,
  -- Then (p * q).coeff(m + n) = p.coeff(m) * q.coeff(n) ≠ 0
  -- And (p * q).coeff(k) = 0 for k > m + n
  -- This uses that F is a field (no zero divisors!)

end PolynomialRing

-- ============================================================================
-- PART 4: DIVISION ALGORITHM FOR POLYNOMIALS
-- ============================================================================

section PolynomialDivision

variable {F : Type} [Field F]

-- TODO 4.1: Prove Division Algorithm for Polynomials
-- This is THE KEY THEOREM for polynomial theory!
-- Given f, g with g ≠ 0, there exist unique q, r such that:
-- f = q * g + r and degree(r) < degree(g)
theorem Polynomial.div_algorithm (f g : F[X]) (hg : g ≠ 0) :
    ∃! (q r : F[X]), f = q * g + r ∧ r.degree < g.degree :=
  sorry
  -- Proof strategy (IMPORTANT!):
  -- Existence: Strong induction on degree of f
  -- Base: If deg(f) < deg(g), take q = 0, r = f
  -- Step: If deg(f) ≥ deg(g), 
  --   Let c = leadingCoeff(f) / leadingCoeff(g)  (division in field!)
  --   Let d = deg(f) - deg(g)
  --   Consider f₁ = f - c * X^d * g
  --   Then deg(f₁) < deg(f) (leading terms cancel!)
  --   By IH: f₁ = q₁ * g + r with deg(r) < deg(g)
  --   So f = (q₁ + c * X^d) * g + r
  -- Uniqueness: If f = q₁ * g + r₁ = q₂ * g + r₂,
  --   Then (q₁ - q₂) * g = r₂ - r₁
  --   If q₁ ≠ q₂, LHS has degree ≥ deg(g)
  --   But RHS has degree < deg(g), contradiction!

-- TODO 4.2: Define quotient and remainder
def Polynomial.div (f g : F[X]) : F[X] :=
  if hg : g = 0 then 0 else Classical.choose (Polynomial.div_algorithm f g hg)

def Polynomial.mod (f g : F[X]) : F[X] :=
  if hg : g = 0 then f 
  else Classical.choose (Classical.choose_spec (Polynomial.div_algorithm f g hg))

instance : Div F[X] := ⟨Polynomial.div⟩
instance : Mod F[X] := ⟨Polynomial.mod⟩

-- TODO 4.3: Prove division algorithm properties
theorem Polynomial.div_mod_eq (f g : F[X]) (hg : g ≠ 0) :
    f = (f / g) * g + (f % g) :=
  sorry
  -- Hint: Use Classical.choose_spec

theorem Polynomial.mod_lt (f g : F[X]) (hg : g ≠ 0) :
    (f % g).degree < g.degree :=
  sorry

end PolynomialDivision

-- ============================================================================
-- PART 5: POLYNOMIAL EVALUATION AND ROOTS
-- ============================================================================

section PolynomialEvaluation

variable {F : Type} [Field F]

-- TODO 5.1: Define polynomial evaluation
-- Evaluate p at a ∈ F: p(a) = Σᵢ p.coeff(i) * a^i
def Polynomial.eval (p : F[X]) (a : F) : F :=
  sorry  -- TODO: Define using finite sum over support
  -- Hint: Use p.support_finite to get N
  -- Then Σᵢ₌₀ᴺ p.coeff(i) * a^i

notation:max p "(" a ")" => Polynomial.eval p a

-- TODO 5.2: Prove evaluation is a ring homomorphism
theorem Polynomial.eval_add (p q : F[X]) (a : F) :
    (p + q)(a) = p(a) + q(a) :=
  sorry
  -- Hint: Use distributivity of field operations

theorem Polynomial.eval_mul (p q : F[X]) (a : F) :
    (p * q)(a) = p(a) * q(a) :=
  sorry
  -- Hint: Cauchy product formula
  -- Σₙ (Σᵢ₊ⱼ₌ₙ p.coeff(i) * q.coeff(j)) * a^n
  -- = (Σᵢ p.coeff(i) * a^i) * (Σⱼ q.coeff(j) * a^j)

theorem Polynomial.eval_zero (a : F) : (0 : F[X])(a) = 0 :=
  sorry

theorem Polynomial.eval_one (a : F) : (1 : F[X])(a) = 1 :=
  sorry

-- TODO 5.3: Define root
def Polynomial.isRoot (p : F[X]) (a : F) : Prop := p(a) = 0

-- TODO 5.4: REMAINDER THEOREM
-- The remainder when dividing p by (X - a) is p(a)
theorem Polynomial.remainder_theorem (p : F[X]) (a : F) :
    p % (Polynomial.X - Polynomial.const a) = Polynomial.const (p(a)) :=
  sorry
  -- Hint: By division algorithm: p = q * (X - a) + r
  -- where deg(r) < deg(X - a) = 1
  -- So r is a constant: r = Polynomial.const c
  -- Evaluate at a: p(a) = q(a) * (a - a) + c = c

-- TODO 5.5: FACTOR THEOREM  
-- a is a root of p iff (X - a) divides p
theorem Polynomial.factor_theorem (p : F[X]) (a : F) :
    p.isRoot a ↔ (Polynomial.X - Polynomial.const a) ∣ p :=
  sorry
  -- Forward: If p(a) = 0, by remainder theorem p % (X - a) = 0
  -- So p = q * (X - a) for some q
  -- Backward: If p = q * (X - a), then p(a) = q(a) * 0 = 0

-- TODO 5.6: Prove polynomial of degree n has at most n roots
theorem Polynomial.card_roots_le_degree (p : F[X]) (hp : p ≠ 0) :
    ∀ S : Finset F, (∀ a ∈ S, p.isRoot a) → S.card ≤ p.degree.unbot hp :=
  sorry
  -- Proof by induction on degree:
  -- Base: deg = 0 means p is nonzero constant, no roots
  -- Step: If a is a root, p = (X - a) * q by factor theorem
  -- Other roots are roots of q, which has smaller degree
  -- At most 1 + deg(q) = deg(p) roots total

end PolynomialEvaluation

-- ============================================================================
-- PART 6: IRREDUCIBLE POLYNOMIALS
-- ============================================================================

section IrreduciblePolynomials

variable {F : Type} [Field F]

-- TODO 6.1: Define irreducible polynomial
-- p is irreducible if p is not a unit and whenever p = a * b,
-- either a or b is a unit
def Polynomial.Irreducible (p : F[X]) : Prop :=
  p ≠ 0 ∧ 
  (p.degree : WithBot ℕ) ≠ ↑0 ∧  -- not a nonzero constant
  ∀ a b : F[X], p = a * b → (a.degree = 0 ∨ b.degree = 0)
  -- Hint: In F[X], units are exactly nonzero constants

-- TODO 6.2: Prove degree 1 polynomials are irreducible
theorem Polynomial.irreducible_of_degree_one {p : F[X]} 
    (h : p.degree = ↑1) : p.Irreducible :=
  sorry
  -- Hint: If p = a * b with deg(p) = 1,
  -- Then deg(a) + deg(b) = 1
  -- So one factor has degree 0 (is a unit)

-- TODO 6.3: Prove irreducible polynomials are prime
-- If p is irreducible and p | a * b, then p | a or p | b
theorem Polynomial.prime_of_irreducible {p : F[X]} (hp : p.Irreducible) :
    ∀ a b : F[X], p ∣ a * b → p ∣ a ∨ p ∣ b :=
  sorry
  -- Hint: This requires GCD theory for polynomials
  -- If p ∤ a, then gcd(p, a) = 1 (p irreducible)
  -- By Bezout: ∃ u, v, u * p + v * a = 1
  -- Multiply by b: u * p * b + v * a * b = b
  -- Since p | a * b and p | p * b, we have p | b

end IrreduciblePolynomials

-- ============================================================================
-- PART 7: UNIQUE FACTORIZATION IN F[X]
-- ============================================================================

section UniqueFactorization

variable {F : Type} [Field F]

-- TODO 7.1: Define associate polynomials
-- p and q are associates if p = c * q for some nonzero constant c
def Polynomial.Associated (p q : F[X]) : Prop :=
  ∃ c : F, c ≠ 0 ∧ p = Polynomial.const c * q

-- TODO 7.2: Prove F[X] is a Euclidean domain
-- The division algorithm makes F[X] Euclidean with norm = degree
-- This implies F[X] is a PID (every ideal is principal)
axiom Polynomial.euclidean_domain : EuclideanDomain F[X]
  -- Proof: Define euclideanDomain with
  -- quotient = Polynomial.div
  -- remainder = Polynomial.mod
  -- Verify axioms using division algorithm

-- TODO 7.3: UNIQUE FACTORIZATION THEOREM
-- Every nonzero non-unit polynomial factors uniquely into irreducibles
-- p = c * p₁^e₁ * p₂^e₂ * ... * pₖ^eₖ
-- where c is a nonzero constant and pᵢ are monic irreducible
axiom Polynomial.unique_factorization (p : F[X]) (hp : p ≠ 0) 
    (hdeg : (p.degree : WithBot ℕ) ≠ ↑0) :
  ∃! (c : F) (factors : List F[X]),
    c ≠ 0 ∧
    (∀ q ∈ factors, q.Irreducible ∧ q.leadingCoeff = 1) ∧
    p = Polynomial.const c * factors.prod
  -- Proof strategy:
  -- Existence: Induction on degree
  --   If p irreducible, done
  --   Otherwise p = a * b with smaller degrees
  --   By IH, factor a and b
  -- Uniqueness: Use that irreducibles are prime
  --   If p₁ * ... * pₙ = q₁ * ... * qₘ
  --   Then p₁ | q₁ * ... * qₘ
  --   So p₁ | qⱼ for some j (p₁ prime)
  --   Since qⱼ irreducible, p₁ and qⱼ associates
  --   Cancel and continue by induction

end UniqueFactorization

-- ============================================================================
-- PART 8: EISENSTEIN'S CRITERION
-- ============================================================================

section EisensteinCriterion

-- Eisenstein's criterion works over ℤ, but we state it generally
-- For polynomial over ℤ to be irreducible over ℚ

-- TODO 8.1: State Eisenstein's Criterion
-- Let p(X) = aₙXⁿ + ... + a₁X + a₀ with aᵢ ∈ ℤ
-- If there exists prime p such that:
--   1. p ∤ aₙ (p doesn't divide leading coefficient)
--   2. p | aᵢ for all i < n (p divides all other coefficients)
--   3. p² ∤ a₀ (p² doesn't divide constant term)
-- Then p(X) is irreducible over ℚ
axiom eisenstein_criterion (p : Polynomial ℤ) (prime : ℕ) 
    (h_prime : Nat.Prime prime)
    (h_leading : ¬(↑prime : ℤ) ∣ p.leadingCoeff)
    (h_others : ∀ i < p.degree.unbot sorry, (↑prime : ℤ) ∣ p.coeff i)
    (h_constant : ¬(↑prime : ℤ)^2 ∣ p.coeff 0) :
  sorry  -- p is irreducible over ℚ (need to define this properly)
  -- Proof idea (sketch):
  -- Suppose p = f * g over ℚ[X]
  -- Clear denominators: p = (1/d) * f₁ * g₁ where f₁, g₁ ∈ ℤ[X]
  -- So d * p = f₁ * g₁
  -- Reduce mod prime: d * p ≡ f₁ * g₁ (mod prime)
  -- LHS: a_n X^n (other terms ≡ 0)
  -- So f₁ * g₁ ≡ (constant) * X^n (mod prime)
  -- This means f₁ ≡ c₁ * X^k, g₁ ≡ c₂ * X^(n-k) (mod prime)
  -- So constant terms of f₁ and g₁ both divisible by prime
  -- But then constant term of f₁ * g₁ divisible by prime²
  -- Contradicting h_constant!

-- TODO 8.2: Example - Show X^n - 2 is irreducible over ℚ
example (n : ℕ) (hn : n > 0) : 
    (Polynomial.X^n - Polynomial.const (2 : ℚ) : ℚ[X]).Irreducible :=
  sorry
  -- Hint: View as polynomial over ℤ: X^n - 2
  -- Apply Eisenstein with prime = 2
  -- 2 ∤ 1 (leading coeff), 2 | 0 (all middle coeffs), 2 | -2 but 4 ∤ -2

end EisensteinCriterion

-- ============================================================================
-- PART 9: MINIMAL POLYNOMIALS AND ALGEBRAIC ELEMENTS
-- ============================================================================

section MinimalPolynomials

variable {F E : Type} [Field F] [Field E] [Algebra F E]

-- TODO 9.1: Define algebraic element
-- α ∈ E is algebraic over F if there exists nonzero p ∈ F[X] with p(α) = 0
def IsAlgebraic (α : E) : Prop :=
  ∃ p : F[X], p ≠ 0 ∧ sorry  -- TODO: p(α) = 0 (need evaluation in E)
  -- Hint: Need to extend polynomial evaluation to E
  -- Use algebra structure: F → E

-- TODO 9.2: Define minimal polynomial
-- The minimal polynomial is the unique monic irreducible polynomial 
-- of smallest degree having α as a root
def minimalPolynomial (α : E) (h : IsAlgebraic α) : F[X] :=
  sorry  -- TODO: Find monic polynomial of minimal degree with α as root
  -- Hint: Use well-ordering on degree
  -- Prove uniqueness: if p, q both minimal, then p - q has α as root
  -- and smaller degree, contradiction!

-- TODO 9.3: Prove minimal polynomial is irreducible
theorem minimalPolynomial_irreducible (α : E) (h : IsAlgebraic α) :
    (minimalPolynomial α h).Irreducible :=
  sorry
  -- Proof: Suppose m = p * q with m minimal
  -- Then m(α) = p(α) * q(α) = 0
  -- So p(α) = 0 or q(α) = 0 (E is a field, no zero divisors)
  -- WLOG p(α) = 0
  -- But deg(p) < deg(m), contradicting minimality!
  -- Unless deg(q) = 0, i.e., q is a unit

-- TODO 9.4: Prove minimal polynomial is unique
theorem minimalPolynomial_unique (α : E) (h : IsAlgebraic α) :
    ∀ p : F[X], p.Irreducible → 
    p.leadingCoeff = 1 → 
    sorry →  -- p(α) = 0
    p = minimalPolynomial α h :=
  sorry
  -- Hint: By division algorithm: m = q * p + r with deg(r) < deg(p)
  -- Evaluate at α: 0 = m(α) = q(α) * p(α) + r(α) = 0 + r(α)
  -- So r(α) = 0 with deg(r) < deg(m), contradicting minimality
  -- Unless r = 0, i.e., p | m
  -- But p irreducible and deg(p) ≤ deg(m), so p and m are associates
  -- Both monic, so p = m

-- TODO 9.5: Prove degree of minimal polynomial
-- [F(α) : F] = deg(minimalPolynomial(α))
-- This is CRUCIAL for field extensions!
axiom degree_eq_dim (α : E) (h : IsAlgebraic α) :
  sorry  -- TODO: State properly with field extension degree
  -- Proof idea:
  -- F(α) = F[X]/(m(X)) where m is minimal polynomial
  -- Dimension is degree of m
  -- Basis: {1, α, α², ..., α^(n-1)} where n = deg(m)

end MinimalPolynomials

/-
IMPLEMENTATION GUIDE SUMMARY:

Part 1 - Definition:
  Polynomials as finitely supported functions
  Key: finite support allows well-defined operations
  Much cleaner than lists of coefficients
  Zero polynomial and constants are basic building blocks

Part 2 - Operations:
  Addition is pointwise (easy)
  Multiplication is convolution (harder!)
  Must prove finite support preserved
  All field axioms lift to polynomial operations

Part 3 - Ring Structure:
  F[X] is a commutative ring
  Degree is multiplicative (KEY property!)
  Uses that F has no zero divisors
  Leading coefficient determines behavior

Part 4 - Division Algorithm:
  THE MOST IMPORTANT THEOREM!
  Makes F[X] a Euclidean domain
  Existence: induction on degree
  Uniqueness: easy degree argument
  Essential for all subsequent theory

Part 5 - Evaluation and Roots:
  Evaluation is ring homomorphism
  Remainder theorem: direct from division algorithm
  Factor theorem: connects roots and divisibility
  Finite number of roots (uses degree)

Part 6 - Irreducibility:
  Irreducible = cannot factor (except trivially)
  Degree 1 always irreducible
  Irreducibles are prime (needs GCD theory)
  Analogous to prime numbers in ℤ

Part 7 - Unique Factorization:
  F[X] is a UFD (unique factorization domain)
  Every polynomial factors uniquely into irreducibles
  Follows from Euclidean domain structure
  Fundamental theorem of algebra over F[X]

Part 8 - Eisenstein's Criterion:
  Practical test for irreducibility
  Works by reduction modulo prime
  Most famous application: Xⁿ - 2 irreducible over ℚ
  Crucial for constructing field extensions

Part 9 - Minimal Polynomials:
  Characterize algebraic elements
  Unique monic irreducible having α as root
  Degree = dimension of field extension
  Bridge between polynomials and field extensions!

KEY PROOF TECHNIQUES:
- Coefficient comparison (Polynomial.ext)
- Induction on degree
- Division algorithm (repeatedly!)
- Evaluation at specific points
- Reduction modulo primes (Eisenstein)
- Dimension counting in field extensions

CONNECTION TO GALOIS THEORY:
- Minimal polynomials define field extensions
- F(α) ≅ F[X]/(m(X)) where m is minimal polynomial
- Degree of minimal polynomial = [F(α) : F]
- Roots of minimal polynomial are conjugates
- Galois group permutes roots of irreducible polynomials
- Splitting fields are built from polynomial roots

WHY THIS MATTERS:
Every field extension comes from a polynomial!
- √2 comes from X² - 2
- ∛2 comes from X³ - 2
- i comes from X² + 1
- ζₙ comes from Xⁿ - 1 (cyclotomic)

The study of field extensions is really the study of 
polynomial roots and their symmetries!

NEXT STEPS:
After mastering Polynomials.lean:
1. Study Fields.lean (field extensions, degrees)
2. Connect minimal polynomials to simple extensions
3. Build splitting fields (add all roots)
4. Define Galois groups (automorphisms fixing base field)
5. Prove Fundamental Theorem of Galois Theory!

The polynomial ring F[X] is the KEY to understanding
all of Galois theory. Master this file and the rest
becomes much clearer!
-/
