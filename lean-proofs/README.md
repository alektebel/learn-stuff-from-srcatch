# Lean Proofs - Journey to Galois Theory

This directory contains mathematical proofs implemented in Lean 4, providing a structured learning path from basic logic to proving the Fundamental Theorem of Galois Theory.

## 🎯 Ultimate Goal

**Prove the Fundamental Theorem of Galois Theory**: Establish the correspondence between intermediate field extensions and subgroups of the Galois group, connecting field theory and group theory in a profound way.

## 📚 Comprehensive Learning Path

This learning path is designed to build your knowledge incrementally, with each stage preparing you for the next. The journey takes you from basic propositional logic to one of the most beautiful theorems in abstract algebra.

### Phase 1: Logical Foundations (Weeks 1-2)

#### 1. **BasicLogic.lean** - Propositional and Predicate Logic
- **Topics**: 
  - Propositional connectives (∧, ∨, →, ¬, ↔)
  - Logical equivalences and tautologies
  - Quantifiers (∀, ∃)
  - Proof techniques: direct proof, contradiction, contrapositive
- **Key Theorems**: Modus ponens, De Morgan's laws, proof by contradiction
- **Prerequisites**: None - start here!
- **Estimated Time**: 3-5 days
- **Why Important**: Fundamental to all mathematical reasoning and Lean syntax

#### 2. **SetTheory.lean** - Sets, Relations, and Functions  
- **Topics**:
  - Set operations (union, intersection, complement)
  - Relations (reflexive, symmetric, transitive, equivalence)
  - Functions (injective, surjective, bijective)
  - Cardinality basics
- **Key Theorems**: Composition of bijections, equivalence class properties
- **Prerequisites**: BasicLogic.lean
- **Estimated Time**: 5-7 days
- **Why Important**: Functions and relations are essential for homomorphisms and field extensions

#### 3. **NaturalNumbers.lean** - Peano Axioms and Mathematical Induction
- **Topics**:
  - Peano axioms definition
  - Principle of mathematical induction
  - Strong induction and well-ordering
  - Recursive definitions
  - Basic arithmetic operations and their properties
- **Key Theorems**: Induction principle, associativity/commutativity of addition/multiplication
- **Prerequisites**: SetTheory.lean
- **Estimated Time**: 4-6 days
- **Why Important**: Induction is crucial for proving properties of algebraic structures

### Phase 2: Algebraic Foundations (Weeks 3-5)

#### 4. **Groups.lean** - Group Theory Fundamentals
- **Topics**:
  - Group axioms (associativity, identity, inverses)
  - Subgroups and cosets
  - Group homomorphisms and isomorphisms
  - Normal subgroups and quotient groups
  - Cyclic groups
  - Lagrange's theorem
- **Key Theorems**: Lagrange's theorem, first isomorphism theorem, subgroup criterion
- **Prerequisites**: SetTheory.lean, NaturalNumbers.lean
- **Estimated Time**: 7-10 days
- **Why Important**: Galois groups are the central object in Galois theory

#### 5. **Rings.lean** - Ring Theory
- **Topics**:
  - Ring axioms (addition group + multiplication)
  - Subrings and ideals
  - Ring homomorphisms and isomorphisms
  - Quotient rings
  - Integral domains
  - Prime and maximal ideals
- **Key Theorems**: First isomorphism theorem for rings, ideal correspondence theorem
- **Prerequisites**: Groups.lean
- **Estimated Time**: 7-10 days
- **Why Important**: Polynomial rings are fundamental to field extensions

#### 6. **Fields.lean** - Field Theory Basics
- **Topics**:
  - Field axioms (ring + multiplicative inverses)
  - Subfields
  - Field homomorphisms (always injective!)
  - Field characteristic
  - Prime fields (ℚ, 𝔽_p)
  - Field extensions basics
- **Key Theorems**: Field homomorphisms are injective, characteristic properties
- **Prerequisites**: Rings.lean
- **Estimated Time**: 5-7 days
- **Why Important**: Fields are the primary objects in Galois theory

### Phase 3: Advanced Algebra - Road to Galois (Weeks 6-10)

#### 7. **Polynomials.lean** - Polynomial Ring Theory
- **Topics**:
  - Polynomial ring construction F[X]
  - Degree and leading coefficient
  - Division algorithm for polynomials
  - Irreducible polynomials
  - Unique factorization in F[X]
  - Roots and the remainder theorem
  - Eisenstein's criterion
- **Key Theorems**: Division algorithm, unique factorization, irreducibility tests
- **Prerequisites**: Fields.lean
- **Estimated Time**: 8-12 days
- **Why Important**: Field extensions are quotients of polynomial rings

#### 8. **FieldExtensions.lean** - Theory of Field Extensions
- **Topics**:
  - Simple extensions F(α)
  - Algebraic vs transcendental elements
  - Minimal polynomials
  - Degree of extensions [E:F]
  - Tower law: [E:F] = [E:K][K:F]
  - Algebraic extensions
  - Algebraic closure
- **Key Theorems**: Tower law, minimal polynomial existence and uniqueness
- **Prerequisites**: Polynomials.lean
- **Estimated Time**: 8-12 days
- **Why Important**: Core concept connecting polynomials to fields

#### 9. **SplittingFields.lean** - Splitting Fields
- **Topics**:
  - Splitting fields definition
  - Existence of splitting fields
  - Uniqueness up to isomorphism
  - Normal extensions
  - Separable polynomials and extensions
  - Perfect fields
- **Key Theorems**: Splitting field existence and uniqueness, separability criteria
- **Prerequisites**: FieldExtensions.lean
- **Estimated Time**: 7-10 days
- **Why Important**: Galois extensions are normal and separable

### Phase 4: Galois Theory (Weeks 11-14)

#### 10. **GaloisTheory.lean** - The Fundamental Theorem
- **Topics**:
  - Galois extensions (normal + separable + algebraic)
  - Galois group Gal(E/F)
  - Fixed fields
  - Fundamental theorem of Galois theory:
    - Correspondence between subgroups and intermediate fields
    - Anti-isomorphism of lattices
    - Normal subgroups ↔ normal extensions
  - Applications:
    - Solvability by radicals
    - Insolvability of quintic
    - Ruler and compass constructions
- **Key Theorems**: 
  - Fundamental theorem of Galois theory
  - Correspondence between subgroups and subfields
  - Galois group of finite fields
- **Prerequisites**: All previous files
- **Estimated Time**: 10-15 days
- **Why Important**: This is the ultimate goal!

## 🛠️ Implementation Approach

### For Each File
1. **Read the template** with TODO comments and guidelines
2. **Study the theory** from recommended resources
3. **Attempt proofs** yourself using Lean tactics
4. **Check solutions** for reference when stuck
5. **Understand deeply** - don't just copy proofs

### Lean Development Workflow
```bash
# Install Lean 4
curl https://raw.githubusercontent.com/leanprover/elan/master/elan-init.sh -sSf | sh

# Create a Lean project
lake new my_galois_project
cd my_galois_project

# Edit files with VS Code + Lean extension
code .

# Build and check proofs
lake build
```

### Using VS Code with Lean
- Install "lean4" extension
- Open .lean files to see interactive proof state
- Hover over tactics to see documentation
- Use Ctrl+Space for autocomplete
- Check goal window for current proof state

## 📖 Essential Resources

### Lean 4 Documentation
- **Theorem Proving in Lean 4**: https://leanprover.github.io/theorem_proving_in_lean4/
- **Mathematics in Lean**: https://leanprover-community.github.io/mathematics_in_lean/
- **Lean 4 Manual**: https://leanprover.github.io/lean4/doc/
- **Mathlib4 Documentation**: https://leanprover-community.github.io/mathlib4_docs/

### Algebra Textbooks
- **Abstract Algebra** by Dummit & Foote (comprehensive reference)
- **Algebra** by Michael Artin (excellent for Galois theory)
- **Galois Theory** by Ian Stewart (accessible introduction)
- **Fields and Galois Theory** by John M. Howie

### Online Courses
- **Natural Number Game**: https://www.ma.imperial.ac.uk/~buzzard/xena/natural_number_game/
- **Kevin Buzzard's Lean Course**: https://www.ma.imperial.ac.uk/~buzzard/xena/
- **Lean for the Curious Mathematician**: Videos and notes online

### Community
- **Lean Zulip Chat**: https://leanprover.zulipchat.com/
- **Lean Community**: https://leanprover-community.github.io/
- **Mathlib Contributing Guide**: https://leanprover-community.github.io/contribute/

## 🎓 Learning Tips

### General Strategy
1. **Start Simple**: Begin with BasicLogic even if you know the content
2. **Be Patient**: Galois theory is deep - expect 3-4 months of study
3. **Prove Everything**: Don't skip proofs or use `sorry` unless experimenting
4. **Use Mathlib**: Lean's math library has many helper lemmas
5. **Ask for Help**: The Lean community is very welcoming

### Debugging Proofs
- Read error messages carefully - they're usually helpful
- Check the tactic state frequently
- Break complex proofs into smaller lemmas
- Use `sorry` temporarily to see if rest of proof works
- Compare with similar proofs in Mathlib

### Common Pitfalls
- Not understanding the difference between definitional and propositional equality
- Forgetting to introduce hypotheses with `intro`
- Misusing `apply` vs `exact`
- Not checking implicit arguments
- Trying to prove too much at once

## 📊 Progress Tracking

Track your progress through the learning path:

```
Phase 1: Logical Foundations
[ ] BasicLogic.lean (3-5 days)
[ ] SetTheory.lean (5-7 days)
[ ] NaturalNumbers.lean (4-6 days)

Phase 2: Algebraic Foundations  
[ ] Groups.lean (7-10 days)
[ ] Rings.lean (7-10 days)
[ ] Fields.lean (5-7 days)

Phase 3: Advanced Algebra
[ ] Polynomials.lean (8-12 days)
[ ] FieldExtensions.lean (8-12 days)
[ ] SplittingFields.lean (7-10 days)

Phase 4: Galois Theory
[ ] GaloisTheory.lean (10-15 days)
```

**Total Estimated Time**: 3-4 months of dedicated study

## 🏆 Milestones

- **Milestone 1**: Complete BasicLogic.lean - You understand Lean syntax
- **Milestone 2**: Complete NaturalNumbers.lean - You can use induction
- **Milestone 3**: Complete Groups.lean - You understand algebraic structures
- **Milestone 4**: Complete Fields.lean - You're ready for Galois theory
- **Milestone 5**: Complete Polynomials.lean - You understand the tools
- **Milestone 6**: Complete GaloisTheory.lean - You proved Galois theorem! 🎉

## 💡 Why Galois Theory?

Galois theory is one of the most beautiful results in mathematics because it:
- Connects two different areas: field theory and group theory
- Explains why certain equations can't be solved by radicals
- Provides a complete answer to ruler-and-compass constructions
- Demonstrates the power of abstract algebra
- Shows the deep structure underlying polynomial equations

## 📁 Directory Structure

```
lean-proofs/
├── README.md                    # This file
├── IMPLEMENTATION_GUIDE.md      # Detailed implementation guidelines
├── BasicLogic.lean              # Template with TODOs
├── SetTheory.lean              # Template
├── NaturalNumbers.lean         # Template
├── Groups.lean                  # Template
├── Rings.lean                   # Template
├── Fields.lean                  # Template
├── Polynomials.lean            # Template
├── FieldExtensions.lean        # Template
├── SplittingFields.lean        # Template
├── GaloisTheory.lean           # Template
└── solutions/                   # Complete implementations
    ├── README.md               # Solution guide
    ├── BasicLogic.lean         # Complete proofs
    ├── SetTheory.lean         # Complete proofs
    └── ...                     # All solutions
```

## 🚀 Getting Started

1. **Start with BasicLogic.lean** to learn Lean syntax
2. **Read IMPLEMENTATION_GUIDE.md** for detailed guidance on each file
3. **Work through templates** following TODO comments
4. **Build incrementally** - test each theorem before moving on
5. **Use solutions** as reference, not as answer key
6. **Join the community** - ask questions on Zulip

Good luck on your journey to proving Galois theorem! 🎓✨
