-- Simple Calculator in Haskell - Template
-- This template guides you through building a calculator with a parser and evaluator

module Calculator where

import Data.Char (isDigit, isSpace)

{- |
TODO 1: Define the expression data type

Guidelines:
- Represent numbers as integers
- Represent binary operations (addition, subtraction, etc.)
- Consider recursive structure for nested expressions

Example structure:
  data Expr = Num Int
            | Add Expr Expr
            | Sub Expr Expr
            | Mul Expr Expr
            | Div Expr Expr
-}
data Expr = Num Int
          -- TODO: Add more constructors
  deriving (Show, Eq)

{- |
TODO 2: Implement the eval function

Guidelines:
- Pattern match on expression constructors
- Recursively evaluate sub-expressions
- Apply the appropriate operation
- Handle division by zero (return Maybe Int or Either String Int)

Example:
  eval (Num 5) = 5
  eval (Add (Num 2) (Num 3)) = 5
-}
eval :: Expr -> Int
eval (Num n) = n
-- TODO: Implement evaluation for other operations
eval _ = 0

{- |
TODO 3: Implement the parse function

Guidelines:
- Parse a string into an Expr
- Handle operator precedence (* and / before + and -)
- Handle parentheses
- Skip whitespace
- Use recursive descent parsing or parser combinators

Suggested approach:
1. Start with parseNumber
2. Add parseFactor (numbers and parentheses)
3. Add parseTerm (multiplication and division)
4. Add parseExpr (addition and subtraction)
-}

-- Parse a number from the input
parseNumber :: String -> Maybe (Int, String)
parseNumber s = Nothing  -- TODO: Implement

-- Parse a factor (number or parenthesized expression)
parseFactor :: String -> Maybe (Expr, String)
parseFactor s = Nothing  -- TODO: Implement

-- Parse a term (multiplication and division)
parseTerm :: String -> Maybe (Expr, String)
parseTerm s = Nothing  -- TODO: Implement

-- Parse an expression (addition and subtraction)
parseExpr :: String -> Maybe (Expr, String)
parseExpr s = Nothing  -- TODO: Implement

-- Main parse function
parse :: String -> Maybe Expr
parse s = case parseExpr s of
  Just (expr, "") -> Just expr
  _ -> Nothing

{- |
TODO 4: Implement the calculator function

Guidelines:
- Combine parse and eval
- Handle parsing errors gracefully
- Return the result or an error message
-}
calculate :: String -> Either String Int
calculate input = Left "Not implemented"  -- TODO: Implement

-- Main function for testing
main :: IO ()
main = do
  putStrLn "Simple Calculator - Enter expressions (or 'quit' to exit)"
  
  -- Test cases
  let tests = ["2 + 3", "10 - 4", "5 * 6", "20 / 4", "2 + 3 * 4", "(2 + 3) * 4"]
  
  putStrLn "\nTest cases:"
  mapM_ (\t -> putStrLn $ t ++ " = " ++ show (calculate t)) tests

{- |
IMPLEMENTATION GUIDE:

Step 1: Define the Expr data type
        Include all operators you want to support

Step 2: Implement eval function
        Start with simple cases, add complexity gradually

Step 3: Implement parseNumber
        Extract digits from the beginning of the string

Step 4: Implement parseFactor
        Handle numbers and parentheses

Step 5: Implement parseTerm
        Handle * and / with correct precedence

Step 6: Implement parseExpr
        Handle + and - with correct precedence

Step 7: Test thoroughly
        Test with various expressions
        Test error cases (division by zero, parse errors)

Learning Points:
- Algebraic data types
- Pattern matching
- Recursive data structures
- Parsing techniques
- Maybe/Either for error handling
-}
