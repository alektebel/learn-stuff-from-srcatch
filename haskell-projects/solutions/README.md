# Solutions

This directory contains complete implementations of Haskell projects.

## Files

- **Calculator.hs** - Complete calculator with parser and evaluator

## Building and Running

```bash
# Using GHC
ghc -o calculator Calculator.hs
./calculator

# Or using runhaskell
runhaskell Calculator.hs

# Interactive (GHCi)
ghci Calculator.hs
> calculate "2 + 3 * 4"
```

## Calculator Implementation

The solution demonstrates:

1. **Algebraic Data Types**: Expr type representing mathematical expressions
2. **Pattern Matching**: Evaluating expressions recursively
3. **Recursive Descent Parsing**: Building expressions from strings
4. **Error Handling**: Using Either for parse and evaluation errors
5. **Operator Precedence**: Correct handling of * / before + -

### Expression Type

```haskell
data Expr = Num Int
          | Add Expr Expr
          | Sub Expr Expr
          | Mul Expr Expr
          | Div Expr Expr
```

### Usage Examples

```haskell
calculate "2 + 3"          -- Right 5
calculate "10 - 4"         -- Right 6
calculate "5 * 6"          -- Right 30
calculate "20 / 4"         -- Right 5
calculate "2 + 3 * 4"      -- Right 14 (respects precedence)
calculate "(2 + 3) * 4"    -- Right 20
calculate "10 / 0"         -- Left "Division by zero"
```

## Learning Points

- **Functional Programming**: Pure functions, immutability
- **Type Safety**: Compile-time guarantees
- **Algebraic Data Types**: Modeling problems with types
- **Pattern Matching**: Elegant case analysis
- **Recursion**: Natural solution for recursive structures
- **Monads**: Either monad for error handling
- **Parsing**: Building parsers from scratch

## Features

1. **Correct Operator Precedence**
   - Multiplication and division before addition and subtraction
   - Parentheses for grouping

2. **Error Handling**
   - Parse errors for invalid syntax
   - Division by zero detection
   - Clear error messages

3. **Clean Separation**
   - Parsing phase: String → Expr
   - Evaluation phase: Expr → Int

## Extensions to Explore

- More operators (modulo, exponentiation)
- Floating-point numbers
- Variables and assignment
- Functions (sin, cos, sqrt, etc.)
- Using parser combinator libraries (Parsec, Megaparsec)
- Pretty-printing expressions
- Simplification and optimization

## Haskell Concepts Demonstrated

- Data types and constructors
- Recursive functions
- Maybe and Either types
- List processing
- String manipulation
- Type inference
- Pure functions
- Immutability

## Resources

- "Learn You a Haskell for Great Good!" by Miran Lipovača
- "Programming in Haskell" by Graham Hutton
- Real World Haskell
- Haskell Wiki: https://wiki.haskell.org/
