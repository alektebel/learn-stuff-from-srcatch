# Solutions

This directory contains complete implementations of Lean proofs.

## Files

- **BasicLogic.lean** - Proofs of fundamental logical theorems

## Running

```bash
# Using Lean 4
lean BasicLogic.lean

# Or in VSCode with Lean extension
# Open the file to see proof states interactively
```

## Proofs Included

### Propositional Logic

1. **Modus Ponens**: (P → Q) ∧ P → Q
2. **Conjunction Introduction**: P ∧ Q → P ∧ Q
3. **Conjunction Elimination**: P ∧ Q → P and P ∧ Q → Q
4. **Disjunction Introduction**: P → P ∨ Q and Q → P ∨ Q
5. **OR Commutativity**: P ∨ Q ↔ Q ∨ P

### Implication and Negation

6. **Implication Transitivity**: (P → Q) → (Q → R) → (P → R)
7. **Double Negation Introduction**: P → ¬¬P
8. **Ex Falso**: P ∧ ¬P → Q

### Equivalences

9. **AND Commutativity**: P ∧ Q ↔ Q ∧ P
10. **AND Associativity**: (P ∧ Q) ∧ R ↔ P ∧ (Q ∧ R)
11. **OR Associativity**: (P ∨ Q) ∨ R ↔ P ∨ (Q ∨ R)
12. **Distributive Law**: P ∧ (Q ∨ R) ↔ (P ∧ Q) ∨ (P ∧ R)
13. **De Morgan's Law**: ¬(P ∨ Q) ↔ ¬P ∧ ¬Q

## Learning Points

### Lean Basics

- **Propositions as Types**: Proofs are programs
- **Curry-Howard Correspondence**: Logic and type theory connection
- **Tactics**: Automated proof construction
- **Proof Terms**: Direct proof construction

### Common Tactics Used

```lean
intro      -- Introduce hypothesis
apply      -- Apply theorem/function
exact      -- Provide exact proof term
cases      -- Case analysis
constructor -- Use type constructor
split      -- Split conjunction/biconditional
left/right -- Choose disjunction side
```

### Proof Styles

1. **Term Mode**: Direct proof construction
   ```lean
   theorem example (P : Prop) (hp : P) : P := hp
   ```

2. **Tactic Mode**: Step-by-step construction
   ```lean
   theorem example (P : Prop) (hp : P) : P := by
     exact hp
   ```

3. **Anonymous Functions**: Lambda notation
   ```lean
   theorem example (P : Prop) : P → P :=
     fun hp => hp
   ```

## Understanding Lean Proofs

### Type Signatures

```lean
theorem modus_ponens (P Q : Prop) (h1 : P → Q) (h2 : P) : Q
```

- `(P Q : Prop)`: P and Q are propositions
- `(h1 : P → Q)`: h1 is a proof that P implies Q
- `(h2 : P)`: h2 is a proof of P
- `: Q`: The theorem proves Q

### Proof Construction

Proofs are constructed by:
1. Introducing hypotheses
2. Applying existing theorems
3. Case analysis on disjunctions
4. Using constructors for conjunctions
5. Providing exact proof terms

## Next Steps

After mastering basic logic:

1. **Natural Numbers**: Peano axioms, induction
2. **Set Theory**: Sets, relations, functions
3. **Groups**: Abstract algebra basics
4. **Rings and Fields**: More algebra
5. **Galois Theory**: Ultimate goal!

## Resources

- Theorem Proving in Lean 4: https://leanprover.github.io/theorem_proving_in_lean4/
- Mathematics in Lean: https://leanprover-community.github.io/mathematics_in_lean/
- Lean Community: https://leanprover.zulipchat.com/
- Natural Number Game: https://www.ma.imperial.ac.uk/~buzzard/xena/natural_number_game/

## Tips

- Use VSCode with Lean extension for interactive proof development
- Check proof state with cursor position
- Use `sorry` as placeholder while developing proofs
- Read error messages carefully - they guide you
- Practice with simple proofs first
