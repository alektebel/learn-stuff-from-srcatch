-- Set Theory - Solution Guide
-- This file contains PROOF STRATEGIES and KEY INSIGHTS, not complete implementations
-- The goal is to guide your thinking, not provide copy-paste solutions

/-
OVERALL STRATEGY FOR SET THEORY PROOFS:

1. Set Extensionality: To prove S = T, use 'ext' and show ∀ x, x ∈ S ↔ x ∈ T
2. Subset Proofs: Unfold definition, introduce element and hypothesis
3. Membership: Often reduces to logical propositions from BasicLogic
4. Relations: Define properties, prove by checking definition
5. Functions: Composition follows clear patterns

KEY INSIGHT: In Lean, sets are predicates (α → Prop)
This means x ∈ S is really just S x (applying the predicate to x)
-/

-- ============================================================================
-- PART 1: BASIC SET OPERATIONS - PROOF STRATEGIES
-- ============================================================================

-- subset_refl: S ⊆ S
-- STRATEGY: Unfold subset definition → intro x and h → exact h
-- KEY INSIGHT: This is literally the identity function on elements

-- subset_trans: S ⊆ T → T ⊆ U → S ⊆ U  
-- STRATEGY: Intro hypotheses → intro x → apply first subset, then second
-- KEY INSIGHT: Function composition at the element level

-- subset_antisymm: S ⊆ T → T ⊆ S → S = T
-- STRATEGY: Use Set.ext to reduce to ∀ x, x ∈ S ↔ x ∈ T
-- KEY INSIGHT: Two subsets in both directions mean equality
-- PROOF SKETCH:
--   ext x
--   constructor
--   · intro h; apply hST; exact h
--   · intro h; apply hTS; exact h

-- ============================================================================
-- PART 2: UNION AND INTERSECTION - PROOF STRATEGIES
-- ============================================================================

-- union_comm: S ∪ T = T ∪ S
-- STRATEGY: Use ext, reduce to showing x ∈ S ∨ x ∈ T ↔ x ∈ T ∨ x ∈ S
-- KEY INSIGHT: This reduces to or_comm from BasicLogic
-- PROOF SKETCH:
--   ext x
--   constructor
--   · intro h; cases h; right; exact h; left; exact h
--   · intro h; cases h; right; exact h; left; exact h
-- ALTERNATIVE: simp [Set.mem_union, or_comm]

-- inter_distrib_union: S ∩ (T ∪ U) = (S ∩ T) ∪ (S ∩ U)
-- STRATEGY: ext, unfold to logical formula, use and_or_distrib
-- KEY INSIGHT: Set operations mirror logical operations
-- PROOF SKETCH:
--   ext x
--   simp [Set.mem_inter, Set.mem_union]
--   -- Now have: (x ∈ S ∧ (x ∈ T ∨ x ∈ U)) ↔ (x ∈ S ∧ x ∈ T) ∨ (x ∈ S ∧ x ∈ U)
--   -- This is exactly and_or_distrib from BasicLogic

-- ============================================================================
-- PART 3: COMPLEMENT - PROOF STRATEGIES
-- ============================================================================

-- compl_union: (S ∪ T)ᶜ = Sᶜ ∩ Tᶜ
-- STRATEGY: ext, unfold to negations, use de_morgan_or
-- KEY INSIGHT: Complement = negation, so De Morgan's laws apply
-- PROOF SKETCH:
--   ext x
--   simp [Set.mem_compl, Set.mem_union, Set.mem_inter]
--   -- ¬(x ∈ S ∨ x ∈ T) ↔ ¬(x ∈ S) ∧ ¬(x ∈ T)
--   -- This is de_morgan_or!

-- compl_inter: (S ∩ T)ᶜ = Sᶜ ∪ Tᶜ
-- STRATEGY: Similar to above, but may need classical logic
-- KEY INSIGHT: One direction of De Morgan is constructive, other needs LEM
-- PROOF SKETCH:
--   ext x
--   constructor
--   · intro h; by_contra hneg; push_neg at hneg
--     -- Get contradiction from h and hneg
--   · intro h; intro hmem; cases h
--     -- Show both cases lead to contradiction

-- ============================================================================
-- PART 4: RELATIONS - PROOF STRATEGIES
-- ============================================================================

-- eq_refl: Reflexive (@Eq α)
-- STRATEGY: Unfold Reflexive, intro x, use rfl
-- KEY INSIGHT: Equality is reflexive by definition
-- ONE LINE: fun x => rfl

-- equiv_class_eq_iff: EquivClass R a = EquivClass R b ↔ R a b
-- STRATEGY: This is THE KEY THEOREM for equivalence classes
-- PROOF SKETCH:
--   constructor
--   -- Forward direction: If [a] = [b], then R a b
--   · intro heq
--     -- a ∈ [a] by reflexivity
--     have ha : a ∈ EquivClass R a := hrefl a
--     -- [a] = [b], so a ∈ [b]
--     rw [heq] at ha
--     -- a ∈ [b] means R b a
--     -- Use symmetry to get R a b
--     exact hsymm ha
--   -- Backward direction: If R a b, then [a] = [b]
--   · intro hab
--     ext x
--     constructor
--     · intro hax  -- x ∈ [a], so R a x
--       -- R a b and R a x, use transitivity
--       exact htrans (hsymm hab) hax
--     · intro hbx  -- x ∈ [b], so R b x
--       -- R a b and R b x, use transitivity
--       exact htrans hab hbx

-- ============================================================================
-- PART 5: FUNCTIONS - PROOF STRATEGIES
-- ============================================================================

-- comp_injective: Injective f → Injective g → Injective (g ∘ f)
-- STRATEGY: Unfold injective, assume (g ∘ f) a₁ = (g ∘ f) a₂
-- KEY INSIGHT: Apply injectivity of g, then injectivity of f
-- PROOF SKETCH:
--   intros hf hg a₁ a₂ h
--   -- h : g (f a₁) = g (f a₂)
--   have : f a₁ = f a₂ := hg h
--   exact hf this

-- comp_surjective: Surjective f → Surjective g → Surjective (g ∘ f)
-- STRATEGY: Unfold surjective, given c, need a with (g ∘ f) a = c
-- KEY INSIGHT: Get b from surjectivity of g, then a from surjectivity of f
-- PROOF SKETCH:
--   intros hf hg c
--   obtain ⟨b, hb⟩ := hg c  -- b such that g b = c
--   obtain ⟨a, ha⟩ := hf b  -- a such that f a = b
--   use a
--   simp [ha, hb]

-- bijective_has_inverse: Bijective f → ∃ g, (∀ a, g (f a) = a) ∧ (∀ b, f (g b) = b)
-- STRATEGY: This requires axiom of choice!
-- KEY INSIGHT: For each b, use surjectivity to get preimage
-- PROOF SKETCH:
--   intro ⟨hinj, hsurj⟩
--   -- Use Classical.choice to select preimage for each b
--   use fun b => Classical.choose (hsurj b)
--   constructor
--   · intro a
--     -- Need: choose (f a) = a
--     -- Use injectivity
--   · intro b
--     -- Need: f (choose b) = b
--     -- This follows from Classical.choose_spec

-- ============================================================================
-- COMMON MISTAKES TO AVOID
-- ============================================================================

/-
MISTAKE 1: Forgetting to use 'ext' for set equality
CORRECT: ext x; constructor; ...
WRONG: Try to directly manipulate sets

MISTAKE 2: Not unfolding definitions enough
CORRECT: Unfold subset, membership, etc. to see logical structure
WRONG: Try to work with abstract notions

MISTAKE 3: Not recognizing when you need classical logic
CORRECT: Use 'open Classical' and 'em' for LEM when needed
WRONG: Try to prove classically true theorem constructively

MISTAKE 4: For equivalence classes, not using transitivity
CORRECT: Chain: R a b + R b x = R a x
WRONG: Try to prove directly without intermediate steps

MISTAKE 5: For function composition, wrong order
CORRECT: (g ∘ f) x = g (f x), apply injectivity of g FIRST
WRONG: Apply injectivity of f first
-/

-- ============================================================================
-- TESTING YOUR PROOFS
-- ============================================================================

/-
For each theorem, after proving it:

1. State it in English and verify it makes sense
2. Try a concrete example (e.g., specific sets of numbers)
3. Check if you used only allowed lemmas
4. Verify no 'sorry' remains
5. Ask: Could I explain this proof to someone else?

Example test for union_comm:
- English: "Union is commutative"
- Concrete: {1,2} ∪ {3,4} = {3,4} ∪ {1,2} ✓
- Used: or_comm from BasicLogic ✓
- No sorry ✓
- Can explain: "Just reorder the OR" ✓
-/

-- ============================================================================
-- NEXT STEPS
-- ============================================================================

/-
After completing SetTheory.lean, you should:

1. Understand how sets = predicates in Lean
2. Be comfortable with extensionality proofs
3. Know when to use classical vs constructive logic
4. Understand equivalence relations (key for quotients!)
5. Be able to prove function properties

These skills are ESSENTIAL for:
- Groups: Cosets are equivalence classes
- Rings: Quotient rings use equivalence relations
- Galois: Field extensions use function properties

Move on to NaturalNumbers.lean to learn induction!
-/
