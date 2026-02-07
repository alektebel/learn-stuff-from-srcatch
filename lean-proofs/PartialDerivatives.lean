-- Partial Derivatives and Multivariable Calculus in Lean - Template
-- This file guides you through proving theorems about partial derivatives, gradients, and optimization
-- Essential foundation for multivariable calculus and differential geometry

/-
LEARNING OBJECTIVES:
1. Understand partial derivatives and directional derivatives
2. Prove properties of the gradient and Jacobian
3. Master the chain rule for multivariable functions
4. Work with Taylor's theorem in multiple dimensions
5. Understand optimization via critical points
6. Develop intuition for multivariable analysis
-/

-- ============================================================================
-- PART 1: PARTIAL DERIVATIVES
-- ============================================================================

namespace PartialDerivatives

-- Definition: Partial derivative with respect to i-th coordinate at point a
-- ∂f/∂xᵢ at point a is the limit of (f(a + hєᵢ) - f(a)) / h as h → 0
-- where єᵢ is the i-th standard basis vector
-- This predicate determines whether L is the value of the partial derivative at point a

-- For simplicity, we'll work with functions ℝⁿ → ℝ
-- Represent ℝⁿ as (Fin n → ℝ)

def partial_derivative (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (i : Fin n) : ℝ → Prop :=
  fun L => ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ →
    |((f (fun j => if j = i then a j + h else a j)) - f a) / h - L| < ε

-- Notation: ∂f/∂xᵢ at point a
-- Usage: ∂[i] f a L means the partial derivative of f at a in direction i is L
notation "∂[" i "]" f " " a => fun L => partial_derivative f a i L

-- TODO 1.1: Prove that partial derivatives are unique (if they exist)
theorem partial_derivative_unique (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (i : Fin n) (L₁ L₂ : ℝ)
  (h1 : ∂[i] f a L₁) (h2 : ∂[i] f a L₂) : L₁ = L₂ :=
  sorry  -- TODO: Similar to uniqueness of limits

-- TODO 1.2: Prove linearity of partial derivatives
-- If ∂f/∂xᵢ and ∂g/∂xᵢ exist, then ∂(f + g)/∂xᵢ = ∂f/∂xᵢ + ∂g/∂xᵢ
theorem partial_derivative_add (f g : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (i : Fin n)
  (Lf Lg : ℝ) (hf : ∂[i] f a Lf) (hg : ∂[i] g a Lg) :
  ∂[i] (fun x => f x + g x) a (Lf + Lg) :=
  sorry  -- TODO: Split the difference quotient and use triangle inequality

-- TODO 1.3: Prove product rule for partial derivatives
-- ∂(fg)/∂xᵢ = f · ∂g/∂xᵢ + g · ∂f/∂xᵢ (at point a)
theorem partial_derivative_mul (f g : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (i : Fin n)
  (Lf Lg : ℝ) (hf : ∂[i] f a Lf) (hg : ∂[i] g a Lg) :
  ∂[i] (fun x => f x * g x) a (f a * Lg + g a * Lf) :=
  sorry  -- TODO: Use (fg)' = f·g' + g·f' from single-variable calculus

-- ============================================================================
-- PART 2: GRADIENT AND DIRECTIONAL DERIVATIVES
-- ============================================================================

-- Definition: Gradient is the vector of all partial derivatives
def has_gradient (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (∇f : Fin n → ℝ) : Prop :=
  ∀ i, ∂[i] f a (∇f i)

-- Notation: ∇f
notation "∇" f => has_gradient f

-- Definition: Directional derivative in direction v
-- D_v f(a) = lim_{h→0} (f(a + hv) - f(a)) / h
def directional_derivative (f : (Fin n → ℝ) → ℝ) (a v : Fin n → ℝ) (L : ℝ) : Prop :=
  ∀ ε > 0, ∃ δ > 0, ∀ h, 0 < |h| ∧ |h| < δ →
    |((f (fun i => a i + h * v i)) - f a) / h - L| < ε

-- TODO 2.1: Prove that if f is differentiable, D_v f = ∇f · v
-- The directional derivative equals the dot product of gradient and direction
theorem directional_derivative_is_gradient_dot (f : (Fin n → ℝ) → ℝ) (a v : Fin n → ℝ) (∇f : Fin n → ℝ)
  (h_grad : ∇ f a ∇f) :
  directional_derivative f a v (Finset.sum Finset.univ fun i => ∇f i * v i) :=
  sorry  -- TODO: Use linearity and definition of gradient

-- TODO 2.2: Prove that gradient points in direction of steepest ascent
-- Among all unit vectors v, ∇f·v is maximized when v = ∇f/‖∇f‖
theorem gradient_steepest_ascent (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (∇f : Fin n → ℝ)
  (h_grad : ∇ f a ∇f) (h_nonzero : ∃ i, ∇f i ≠ 0) :
  ∀ v : Fin n → ℝ, (Finset.sum Finset.univ fun i => v i ^ 2) = 1 →
    Finset.sum Finset.univ (fun i => ∇f i * v i) ≤
    Real.sqrt (Finset.sum Finset.univ fun i => ∇f i ^ 2) :=
  sorry  -- TODO: Use Cauchy-Schwarz inequality

-- ============================================================================
-- PART 3: DIFFERENTIABILITY AND THE JACOBIAN
-- ============================================================================

-- Definition: f is differentiable at a if there exists a linear map L such that
-- f(a + h) = f(a) + L(h) + o(‖h‖)
-- where o(‖h‖)/‖h‖ → 0 as ‖h‖ → 0

def differentiable_at (f : (Fin m → ℝ) → (Fin n → ℝ)) (a : Fin m → ℝ)
  (L : (Fin m → ℝ) → (Fin n → ℝ)) : Prop :=
  -- L is linear
  (∀ x y, L (fun i => x i + y i) = fun i => L x i + L y i) ∧
  (∀ c x, L (fun i => c * x i) = fun i => c * L x i) ∧
  -- f(a + h) - f(a) - L(h) = o(‖h‖)
  (∀ ε > 0, ∃ δ > 0, ∀ h,
    Real.sqrt (Finset.sum Finset.univ fun i => h i ^ 2) < δ →
    Real.sqrt (Finset.sum Finset.univ fun i => (f (fun j => a j + h j) i - f a i - L h i) ^ 2) ≤
    ε * Real.sqrt (Finset.sum Finset.univ fun i => h i ^ 2))

-- Definition: Jacobian matrix - matrix of all partial derivatives
-- J_ij = ∂fᵢ/∂xⱼ
def jacobian (f : (Fin m → ℝ) → (Fin n → ℝ)) (a : Fin m → ℝ) (J : Fin n → Fin m → ℝ) : Prop :=
  ∀ i j, ∂[j] (fun x => f x i) a (J i j)

-- TODO 3.1: Prove that differentiability implies continuity
theorem differentiable_continuous (f : (Fin m → ℝ) → (Fin n → ℝ)) (a : Fin m → ℝ) (L : (Fin m → ℝ) → (Fin n → ℝ))
  (h : differentiable_at f a L) :
  ∀ ε > 0, ∃ δ > 0, ∀ x,
    Real.sqrt (Finset.sum Finset.univ fun i => (x i - a i) ^ 2) < δ →
    Real.sqrt (Finset.sum Finset.univ fun i => (f x i - f a i) ^ 2) < ε :=
  sorry  -- TODO: Use definition with ε/2 for linear part

-- TODO 3.2: Prove that differentiability implies existence of all partial derivatives
theorem differentiable_implies_partials (f : (Fin m → ℝ) → (Fin n → ℝ)) (a : Fin m → ℝ)
  (L : (Fin m → ℝ) → (Fin n → ℝ)) (h : differentiable_at f a L) :
  ∃ J : Fin n → Fin m → ℝ, jacobian f a J :=
  sorry  -- TODO: Extract partials from linear map L

-- TODO 3.3: Prove that if Jacobian exists and is continuous, then f is differentiable
-- (This is the converse - requires more work)
axiom continuous_jacobian_implies_differentiable :
  ∀ (f : (Fin m → ℝ) → (Fin n → ℝ)) (a : Fin m → ℝ) (J : Fin n → Fin m → ℝ),
  jacobian f a J →
  (∀ i j ε > 0, ∃ δ > 0, ∀ x,
    Real.sqrt (Finset.sum Finset.univ fun k => (x k - a k) ^ 2) < δ →
    |∂[j] (fun x => f x i) x (J i j) - J i j| < ε) →
  ∃ L, differentiable_at f a L

-- ============================================================================
-- PART 4: CHAIN RULE
-- ============================================================================

-- TODO 4.1: Prove the multivariable chain rule
-- If g: ℝᵐ → ℝⁿ is differentiable at a with Jacobian J_g
-- and f: ℝⁿ → ℝᵖ is differentiable at g(a) with Jacobian J_f
-- then f ∘ g is differentiable at a with Jacobian J_f · J_g (matrix multiplication)
theorem chain_rule
  (f : (Fin n → ℝ) → (Fin p → ℝ)) (g : (Fin m → ℝ) → (Fin n → ℝ))
  (a : Fin m → ℝ) (L_g : (Fin m → ℝ) → (Fin n → ℝ)) (L_f : (Fin n → ℝ) → (Fin p → ℝ))
  (h_g : differentiable_at g a L_g) (h_f : differentiable_at f (g a) L_f) :
  differentiable_at (fun x => f (g x)) a (fun h => L_f (L_g h)) :=
  sorry  -- TODO: Use composition of linear approximations

-- TODO 4.2: Chain rule in terms of Jacobians
-- J_{f∘g} = J_f · J_g (matrix product)
theorem jacobian_chain_rule
  (f : (Fin n → ℝ) → (Fin p → ℝ)) (g : (Fin m → ℝ) → (Fin n → ℝ))
  (a : Fin m → ℝ) (J_f : Fin p → Fin n → ℝ) (J_g : Fin n → Fin m → ℝ)
  (h_f : jacobian f (g a) J_f) (h_g : jacobian g a J_g) :
  jacobian (fun x => f (g x)) a (fun i k => Finset.sum Finset.univ fun j => J_f i j * J_g j k) :=
  sorry  -- TODO: Apply chain rule to each component

-- ============================================================================
-- PART 5: TAYLOR'S THEOREM AND HIGHER DERIVATIVES
-- ============================================================================

-- Definition: Second partial derivative
-- ∂²f/∂xᵢ∂xⱼ
def second_partial (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (i j : Fin n) (L : ℝ) : Prop :=
  ∃ ∂f_j : (Fin n → ℝ) → ℝ,
    (∀ x, ∂[j] f x (∂f_j x)) ∧
    ∂[i] ∂f_j a L

-- TODO 5.1: Prove Schwarz's theorem (equality of mixed partials)
-- If ∂²f/∂xᵢ∂xⱼ and ∂²f/∂xⱼ∂xᵢ exist and are continuous, they are equal
theorem schwarz_mixed_partials (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (i j : Fin n)
  (L_ij L_ji : ℝ) (h_ij : second_partial f a i j L_ij) (h_ji : second_partial f a j i L_ji)
  (h_cont : True) :  -- Continuity assumption simplified
  L_ij = L_ji :=
  sorry  -- TODO: Use the limit definition and interchange limits (requires continuity)

-- Definition: Hessian matrix (matrix of second partial derivatives)
def hessian (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (H : Fin n → Fin n → ℝ) : Prop :=
  ∀ i j, second_partial f a i j (H i j)

-- TODO 5.2: Prove that the Hessian is symmetric (under continuity assumptions)
theorem hessian_symmetric (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (H : Fin n → Fin n → ℝ)
  (h : hessian f a H) (h_cont : True) :
  ∀ i j, H i j = H j i :=
  sorry  -- TODO: Apply Schwarz's theorem

-- TODO 5.3: State Taylor's theorem (second order)
-- f(a + h) = f(a) + ∇f(a)·h + (1/2)h^T H(a) h + o(‖h‖²)
axiom taylor_second_order :
  ∀ (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (∇f : Fin n → ℝ) (H : Fin n → Fin n → ℝ),
  has_gradient f a ∇f →
  hessian f a H →
  ∀ ε > 0, ∃ δ > 0, ∀ h,
    Real.sqrt (Finset.sum Finset.univ fun i => h i ^ 2) < δ →
    |f (fun i => a i + h i) - f a -
     Finset.sum Finset.univ (fun i => ∇f i * h i) -
     (1/2) * Finset.sum Finset.univ (fun i => Finset.sum Finset.univ (fun j => H i j * h i * h j))| ≤
    ε * Finset.sum Finset.univ (fun i => h i ^ 2)

-- ============================================================================
-- PART 6: OPTIMIZATION AND CRITICAL POINTS
-- ============================================================================

-- Definition: Critical point (gradient is zero)
def is_critical_point (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) : Prop :=
  ∃ ∇f : Fin n → ℝ, has_gradient f a ∇f ∧ ∀ i, ∇f i = 0

-- TODO 6.1: Prove first-order necessary condition for local minimum
-- If a is a local minimum, then ∇f(a) = 0
theorem local_min_critical (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ)
  (h_min : ∃ δ > 0, ∀ x, Real.sqrt (Finset.sum Finset.univ fun i => (x i - a i) ^ 2) < δ → f a ≤ f x)
  (h_diff : ∃ L, differentiable_at (fun x => f x) a L) :
  is_critical_point f a :=
  sorry  -- TODO: If ∇f ≠ 0, can move in direction -∇f to decrease f

-- TODO 6.2: Prove second-order sufficient condition for local minimum
-- If ∇f(a) = 0 and Hessian is positive definite, then a is a local minimum
theorem positive_definite_hessian_implies_min
  (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (H : Fin n → Fin n → ℝ)
  (h_crit : is_critical_point f a)
  (h_hess : hessian f a H)
  (h_pos_def : ∀ v : Fin n → ℝ, (∃ i, v i ≠ 0) →
    0 < Finset.sum Finset.univ (fun i => Finset.sum Finset.univ (fun j => H i j * v i * v j))) :
  ∃ δ > 0, ∀ x, 0 < Real.sqrt (Finset.sum Finset.univ fun i => (x i - a i) ^ 2) →
    Real.sqrt (Finset.sum Finset.univ fun i => (x i - a i) ^ 2) < δ →
    f a < f x :=
  sorry  -- TODO: Use Taylor's theorem with positive definite quadratic form

-- TODO 6.3: Classify critical points using eigenvalues of Hessian
-- - All positive eigenvalues → local minimum
-- - All negative eigenvalues → local maximum  
-- - Mixed signs → saddle point
axiom classify_critical_point :
  ∀ (f : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (H : Fin n → Fin n → ℝ),
  is_critical_point f a →
  hessian f a H →
  -- Eigenvalue conditions would go here
  True

-- ============================================================================
-- PART 7: CONSTRAINED OPTIMIZATION (LAGRANGE MULTIPLIERS)
-- ============================================================================

-- TODO 7.1: State Lagrange multiplier theorem
-- To optimize f subject to g = 0, critical points satisfy ∇f = λ∇g
theorem lagrange_multipliers
  (f g : (Fin n → ℝ) → ℝ) (a : Fin n → ℝ) (∇f ∇g : Fin n → ℝ)
  (h_grad_f : has_gradient f a ∇f)
  (h_grad_g : has_gradient g a ∇g)
  (h_constraint : g a = 0)
  (h_opt : ∀ x, g x = 0 →
    Real.sqrt (Finset.sum Finset.univ fun i => (x i - a i) ^ 2) < 1 →
    f a ≤ f x ∨ f x ≤ f a) :
  ∃ λ : ℝ, ∀ i, ∇f i = λ * ∇g i :=
  sorry  -- TODO: Prove using the fact that ∇g is normal to constraint surface

-- TODO 7.2: Generalize to multiple constraints
-- Optimize f subject to g₁ = 0, ..., gₘ = 0
-- Critical points satisfy ∇f = λ₁∇g₁ + ... + λₘ∇gₘ
axiom lagrange_multipliers_multiple :
  ∀ (m : ℕ) (f : (Fin n → ℝ) → ℝ) (g : Fin m → (Fin n → ℝ) → ℝ) (a : Fin n → ℝ),
  (∀ i, g i a = 0) →
  -- Optimization condition
  True →
  ∃ λ : Fin m → ℝ, True  -- Would state gradient condition here

-- ============================================================================
-- PART 8: IMPLICIT FUNCTION THEOREM
-- ============================================================================

-- TODO 8.1: State the Implicit Function Theorem
-- If F(x,y) = 0 and ∂F/∂y ≠ 0, then locally y can be written as a function of x
axiom implicit_function_theorem :
  ∀ (F : ℝ → ℝ → ℝ) (a b : ℝ),
  F a b = 0 →
  (∃ L, ∂[1] (fun p => F (p 0) (p 1)) (fun i => if i = 0 then a else b) 1 L ∧ L ≠ 0) →
  ∃ (f : ℝ → ℝ) (δ : ℝ), 0 < δ ∧
    f a = b ∧
    (∀ x, |x - a| < δ → F x (f x) = 0)

-- TODO 8.2: Compute derivative of implicitly defined function
-- If F(x, y(x)) = 0, then dy/dx = -(∂F/∂x) / (∂F/∂y)
theorem implicit_derivative (F : ℝ → ℝ → ℝ) (a b : ℝ) (f : ℝ → ℝ)
  (h_impl : ∀ x, F x (f x) = 0)
  (Fx Fy : ℝ) (hx : ∂[0] (fun p => F (p 0) (p 1)) (fun i => if i = 0 then a else b) 0 Fx)
  (hy : ∂[1] (fun p => F (p 0) (p 1)) (fun i => if i = 0 then a else b) 1 Fy)
  (h_nonzero : Fy ≠ 0) :
  ∃ L, ∂[0] (fun p => f (p 0)) (fun _ => a) 0 L ∧ L = -Fx / Fy :=
  sorry  -- TODO: Differentiate F(x, y(x)) = 0 using chain rule

end PartialDerivatives

/-
IMPLEMENTATION GUIDE:

Phase 1: Partial Derivatives (3-4 days)
- Understand limit definition in each coordinate
- Prove uniqueness and linearity
- Product rule is like single-variable case

Phase 2: Gradient and Directional Derivatives (2-3 days)
- Gradient collects all partial derivatives
- Directional derivative = gradient · direction
- Cauchy-Schwarz for steepest ascent

Phase 3: Differentiability and Jacobian (4-5 days)
- Full differentiability is stronger than partial derivatives
- Linear approximation f(a+h) ≈ f(a) + Df(a)·h
- Jacobian matrix represents linear part

Phase 4: Chain Rule (3-4 days)
- Most important theorem for multivariable calculus
- Matrix multiplication of Jacobians
- Composition of linear approximations

Phase 5: Taylor and Higher Derivatives (4-5 days)
- Second derivatives → Hessian matrix
- Schwarz: mixed partials are equal (need continuity)
- Taylor expansion uses Hessian for quadratic term

Phase 6: Optimization (3-4 days)
- Critical points: ∇f = 0
- Hessian determines local behavior
- Positive definite → minimum
- Negative definite → maximum
- Indefinite → saddle point

Phase 7: Constrained Optimization (3-4 days)
- Lagrange multipliers: ∇f = λ∇g
- Powerful technique for constrained problems
- Geometric interpretation: gradients parallel

Phase 8: Implicit Functions (2-3 days)
- Solve F(x,y) = 0 for y = f(x)
- Requires non-zero partial derivative
- Formula: dy/dx = -(∂F/∂x)/(∂F/∂y)

Common Tactics:
- intro/intros
- apply
- obtain/rcases
- have
- calc for chaining inequalities
- field_simp for algebraic manipulation

Key Techniques:
- Epsilon-delta arguments (from Limits.lean)
- Triangle inequality
- Linear approximation
- Matrix multiplication
- Cauchy-Schwarz inequality

Learning Resources:
- Multivariable Calculus by Stewart
- Vector Calculus by Marsden & Tromba
- Analysis on Manifolds by Munkres
- Mathlib4: Analysis.Calculus
- Spivak's Calculus on Manifolds

Applications:
- Optimization problems
- Machine learning (gradient descent)
- Physics (Lagrangian mechanics)
- Economics (utility maximization)
- Engineering (constrained design)

Difficulty: ⭐⭐⭐⭐ Hard
Estimated Time: 15-20 days
Prerequisites: Limits.lean, linear algebra
Next Steps: Differential geometry, manifolds, tensor calculus
-/
