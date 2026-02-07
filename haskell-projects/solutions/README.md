# Solutions

This directory contains complete, working implementations of all Haskell projects.

**⚠️ IMPORTANT**: Try to implement the projects yourself first! These solutions are here for:
- **Reference** when you're stuck
- **Verification** of your approach
- **Learning** alternative implementations
- **Comparison** with your own solution

## Philosophy

The best way to learn is by doing. Use these solutions wisely:
1. ✅ Attempt the project yourself first
2. ✅ Get stuck and struggle (this is where learning happens!)
3. ✅ Try to debug and fix issues on your own
4. ✅ Consult hints and guidelines in the template
5. ✅ Only then peek at solutions for specific functions
6. ✅ Understand the solution, don't just copy it
7. ✅ Implement it yourself after understanding

## Files

### 1. Calculator.hs
**Complete calculator with parser and evaluator**

A mathematical expression parser and evaluator demonstrating:
- **Algebraic Data Types**: Expression representation
- **Pattern Matching**: Recursive evaluation
- **Recursive Descent Parsing**: Building AST from strings
- **Error Handling**: Either monad for errors
- **Operator Precedence**: Correct * / before + -

**Complexity**: Beginner  
**Lines**: ~130  
**Key Concepts**: ADTs, Pattern Matching, Parsing, Recursion

#### Features:
- ✅ Integer arithmetic: +, -, *, /
- ✅ Operator precedence (PEMDAS without exponents)
- ✅ Parentheses for grouping
- ✅ Division by zero detection
- ✅ Clear error messages
- ✅ Whitespace handling

#### Architecture:
```
String → Parser → Expr → Evaluator → Int
         (parse)        (eval)
```

Two-phase design:
1. **Parsing**: Converts string to expression tree
2. **Evaluation**: Walks tree and computes result

#### Expression Type:
```haskell
data Expr = Num Int
          | Add Expr Expr
          | Sub Expr Expr
          | Mul Expr Expr
          | Div Expr Expr
```

#### Usage Examples:
```haskell
calculate "2 + 3"          -- Right 5
calculate "10 - 4"         -- Right 6
calculate "5 * 6"          -- Right 30
calculate "20 / 4"         -- Right 5
calculate "2 + 3 * 4"      -- Right 14 (respects precedence)
calculate "(2 + 3) * 4"    -- Right 20
calculate "10 / 0"         -- Left "Division by zero"
calculate "2 +"            -- Left "Parse error"
```

### 2. JSONParser.hs
**Complete JSON parser from scratch**

A full JSON parser implementing:
- **Recursive Data Structures**: Nested objects and arrays
- **String Parsing**: Escape sequences and Unicode
- **Multiple Data Types**: Null, bool, number, string, array, object
- **Pretty Printing**: Human-readable JSON output

**Complexity**: Intermediate  
**Lines**: ~300  
**Key Concepts**: Recursive parsing, Unicode handling, String processing

#### Features:
- ✅ All JSON types: null, boolean, number, string, array, object
- ✅ String escapes: \n, \t, \", \\, Unicode \uXXXX
- ✅ Nested structures (arbitrary depth)
- ✅ Number formats: integers, floats, scientific notation
- ✅ Pretty-printing with indentation
- ✅ Round-trip: parse → toJSON → parse

#### JSON Value Type:
```haskell
data JSONValue 
  = JSONNull
  | JSONBool Bool
  | JSONNumber Double
  | JSONString String
  | JSONArray [JSONValue]
  | JSONObject [(String, JSONValue)]
```

#### Usage Examples:
```haskell
parse "null"                          -- Right JSONNull
parse "42"                            -- Right (JSONNumber 42.0)
parse "[1, 2, 3]"                     -- Right (JSONArray [..])
parse "{\"name\": \"Alice\"}"         -- Right (JSONObject [..])
parse "{\"users\": [{\"age\": 30}]}"  -- Nested structures
```

### 3. WebScraper.hs
**Web scraper with HTML parsing**

A web scraping tool demonstrating:
- **IO Monad**: HTTP requests and side effects
- **HTML Parsing**: Using TagSoup library
- **Concurrent Operations**: Parallel page scraping
- **Error Handling**: Network and parse errors

**Complexity**: Intermediate  
**Lines**: ~250  
**Key Concepts**: IO monad, External libraries, Concurrency

#### Features:
- ✅ HTTP fetching with http-conduit
- ✅ HTML parsing with TagSoup
- ✅ Link extraction from pages
- ✅ Metadata extraction
- ✅ Concurrent multi-page scraping
- ✅ Pattern-based filtering
- ✅ Error recovery

#### Usage Examples:
```haskell
-- Scrape single page
result <- scrapePage "https://example.com"

-- Get external links only
links <- getExternalLinks "https://example.com"

-- Find specific links
matches <- findLinks "github" "https://example.com"

-- Scrape multiple pages concurrently
results <- scrapePages ["url1", "url2", "url3"]
```

### 4. BuildTool.hs
**Build system similar to Make**

A dependency-based build tool showing:
- **Graph Algorithms**: Topological sorting, cycle detection
- **File System**: Timestamps, path operations
- **Process Execution**: Running shell commands
- **Configuration Parsing**: Buildfile parsing

**Complexity**: Advanced  
**Lines**: ~400  
**Key Concepts**: Graphs, System programming, Algorithms

#### Features:
- ✅ Dependency tracking
- ✅ Topological build ordering
- ✅ Incremental builds (timestamp-based)
- ✅ Phony targets
- ✅ Cycle detection
- ✅ Parallel builds (optional)
- ✅ Makefile-like syntax

#### Buildfile Format:
```makefile
.PHONY: clean test

app: main.o utils.o
	gcc -o app main.o utils.o

main.o: main.c
	gcc -c main.c

clean:
	rm -f *.o app
```

#### Usage Examples:
```bash
# Build default target
./build-tool

# Build specific target
./build-tool app

# Build with custom buildfile
./build-tool -f my-build.txt test
```

## Building and Running

### Using GHC (Compile first, then run)
```bash
# Navigate to solutions directory
cd haskell-projects/solutions/

# Compile a project
ghc -o calculator Calculator.hs

# Run the compiled program
./calculator

# For projects with dependencies (like WebScraper)
ghc -o scraper WebScraper.hs -package http-conduit -package tagsoup
```

### Using runhaskell (Interpret directly)
```bash
# Run without compilation
runhaskell Calculator.hs

# Faster for testing and development
runhaskell JSONParser.hs
```

### Using GHCi (Interactive)
```bash
# Load a module
ghci Calculator.hs

# Test functions interactively
> calculate "2 + 3 * 4"
Right 14

> eval (Add (Num 2) (Num 3))
Right 5

> parse "10 / 2"
Right (Div (Num 10) (Num 2))

# Reload after changes
> :reload

# Get type information
> :type calculate
calculate :: String -> Either String Int

# Exit
> :quit
```

## Solution Structure

Each solution follows consistent patterns:

### 1. Module Declaration
```haskell
module Main where
-- or
module ModuleName where
```

### 2. Imports
```haskell
import Data.Char (isDigit, isSpace)
import Data.Maybe (mapMaybe)
-- etc.
```

### 3. Data Type Definitions
```haskell
data Expr = Num Int | Add Expr Expr | ...
```

### 4. Core Functions
```haskell
parse :: String -> Either String Expr
eval :: Expr -> Either String Int
```

### 5. Helper Functions
```haskell
skipSpace :: String -> String
parseNumber :: String -> Maybe (Int, String)
```

### 6. Main Function
```haskell
main :: IO ()
main = do
  -- Test cases and examples
```

## Learning from Solutions

### How to Study a Solution

1. **Read Top-Down**:
   - Start with data types
   - Understand the main functions
   - Then dive into helpers

2. **Trace Execution**:
   - Pick a simple example
   - Follow it through each function
   - Draw diagrams if needed

3. **Identify Patterns**:
   - How are errors handled?
   - How is recursion used?
   - What helper functions exist?

4. **Compare with Your Code**:
   - What's different?
   - Which approach is clearer?
   - What can you learn?

5. **Modify and Experiment**:
   - Change the code
   - Break it intentionally
   - Fix it again
   - Add features

### Key Patterns in Solutions

#### Pattern 1: Maybe for Optional Results
```haskell
parseNumber :: String -> Maybe (Int, String)
-- Returns Nothing on failure, Just (value, rest) on success
```

#### Pattern 2: Either for Error Messages
```haskell
calculate :: String -> Either String Int
-- Returns Left "error" on failure, Right value on success
```

#### Pattern 3: Recursive Descent Parsing
```haskell
parseExpr s = case parseTerm s of
  Just (expr, rest) -> parseExpr' expr rest
  Nothing -> Nothing
  where parseExpr' left rest = ...
```

#### Pattern 4: Accumulator Pattern
```haskell
processItems acc [] = acc
processItems acc (x:xs) = processItems (acc + process x) xs
```

#### Pattern 5: do-notation for Sequencing
```haskell
calculate input = do
  expr <- parse input    -- Either monad
  eval expr              -- Chains operations
```

## Testing Solutions

### Manual Testing
```bash
# Run the main function
runhaskell Calculator.hs

# Or in GHCi
ghci Calculator.hs
> main
```

### Interactive Testing
```bash
ghci Calculator.hs
> calculate "2 + 3"
Right 5
> calculate "invalid"
Left "Parse error"
```

### Creating Test Suites
```haskell
import Test.HUnit  -- or QuickCheck

tests = TestList
  [ "addition" ~: calculate "2+3" ~?= Right 5
  , "precedence" ~: calculate "2+3*4" ~?= Right 14
  , "error" ~: calculate "2+" ~?= Left "Parse error"
  ]

main = runTestTT tests
```

## Common Haskell Idioms Used

### 1. Pattern Matching
```haskell
eval (Num n) = Right n
eval (Add e1 e2) = ...
```

### 2. Guards
```haskell
needsRebuild target
  | targetPhony target = True
  | not (fileExists target) = True
  | otherwise = False
```

### 3. List Comprehensions
```haskell
links = [Link text href | tag <- tags, isLink tag]
```

### 4. Map and Filter
```haskell
let numbers = map read ["1", "2", "3"] :: [Int]
    evens = filter even numbers
```

### 5. Fold
```haskell
sum = foldl (+) 0 [1,2,3,4,5]
```

### 6. Function Composition
```haskell
process = filter valid . map transform . parse
```

## Performance Considerations

The solutions prioritize **clarity over performance**:

### Optimizations NOT Used:
- ❌ Strict evaluation (bang patterns)
- ❌ Unboxed types
- ❌ Custom parsing libraries (attoparsec)
- ❌ ByteString instead of String
- ❌ Space leak prevention

### Why?
- **Educational focus**: Easier to understand
- **Standard library**: Uses common functions
- **Correctness first**: Performance can come later

### When You Need Performance:
1. Profile first: Use GHC profiler
2. ByteString: For large text processing
3. Vector: For array operations
4. Strict evaluation: Prevent space leaks
5. Specialized libraries: attoparsec, etc.

## Troubleshooting

### Common Issues

**"Module not found"**
```bash
# Install missing packages
cabal update
cabal install <package-name>
```

**"Parse error"**
- Check indentation (Haskell is whitespace-sensitive)
- Ensure proper alignment in do-blocks and where clauses

**"Type mismatch"**
- Read error message carefully
- Check function signatures
- Use :type in GHCi to inspect types

**"Non-exhaustive patterns"**
- Add catch-all pattern: `_ -> ...`
- Or handle all constructors explicitly

## Next Steps

After studying these solutions:

1. **Implement Yourself**: Don't just read, code!
2. **Add Features**: Extend the projects
3. **Refactor**: Make the code better
4. **Test**: Write comprehensive tests
5. **Optimize**: Profile and improve performance
6. **Share**: Help others learn

## Extensions to Try

### Calculator
- [ ] Floating-point numbers
- [ ] More operators (%, ^)
- [ ] Variables and assignment
- [ ] Functions (sin, cos, sqrt)
- [ ] REPL (Read-Eval-Print Loop)

### JSON Parser
- [ ] Better error messages with position
- [ ] Streaming parser for large files
- [ ] JSON Schema validation
- [ ] JSON path queries
- [ ] Benchmarking against Aeson

### Web Scraper
- [ ] Rate limiting
- [ ] Robot.txt checking
- [ ] Cookie handling
- [ ] User agent spoofing
- [ ] Data persistence (database)

### Build Tool
- [ ] Parallel builds
- [ ] Pattern rules (%.o: %.c)
- [ ] Variables in buildfiles
- [ ] Automatic dependency detection
- [ ] Build caching

## Resources

### Official Documentation
- **Haskell.org**: https://www.haskell.org/
- **Hackage**: https://hackage.haskell.org/ (package repository)
- **Hoogle**: https://hoogle.haskell.org/ (function search)

### Books
- **"Learn You a Haskell"**: Beginner-friendly introduction
- **"Programming in Haskell"**: Comprehensive textbook
- **"Real World Haskell"**: Practical applications
- **"Haskell Programming from First Principles"**: In-depth learning

### Online Resources
- **Haskell Wiki**: https://wiki.haskell.org/
- **School of Haskell**: Interactive tutorials
- **r/haskell**: Reddit community
- **#haskell** on IRC: Real-time help

### Practice
- **Exercism.io**: Haskell track with mentoring
- **Project Euler**: Mathematical problems
- **Advent of Code**: Annual programming challenges
- **99 Haskell Problems**: Classic exercises

## Contributing

Found a bug or improvement in the solutions?
- Open an issue
- Submit a pull request
- Suggest better approaches

## License

These educational implementations are provided as learning resources.
Feel free to use, modify, and share.

### 5. NeuralNetwork.hs
**Feedforward neural network with backpropagation**

A complete neural network implementation from scratch demonstrating:
- **Matrix Operations**: Pure functional linear algebra
- **Forward Propagation**: Data flow through layers
- **Backpropagation**: Gradient calculation algorithm
- **Training**: Gradient descent optimization
- **Multiple Activations**: Sigmoid, ReLU, Tanh

**Complexity**: Advanced  
**Lines**: ~250  
**Key Concepts**: Numerical computing, Machine learning algorithms, Functional composition

#### Features:
- ✅ Multiple activation functions (Sigmoid, Tanh, ReLU, Identity)
- ✅ Forward propagation through arbitrary network architectures
- ✅ Backpropagation algorithm for gradient computation
- ✅ Gradient descent training with configurable learning rate
- ✅ Mean squared error loss function
- ✅ Training examples (XOR problem)
- ✅ Pure functional matrix/vector operations

#### Network Architecture:
```haskell
data Layer = Layer 
  { weights :: Matrix      -- Weight matrix
  , biases :: Vector       -- Bias vector  
  , activation :: Activation
  }

type Network = [Layer]     -- Sequence of layers
```

#### Usage Examples:
```haskell
-- Create a 2-3-1 network (2 inputs, 3 hidden, 1 output)
let network = createNetwork [(2, 3, Sigmoid), (3, 1, Sigmoid)]

-- Train on XOR data
let xorData = [([0,0], [0]), ([0,1], [1]), ([1,0], [1]), ([1,1], [0])]
let (trainedNet, loss) = trainEpoch network xorData 0.1

-- Make prediction
let output = predict trainedNet [1, 0]  -- Should be close to [1]
```

### 6. DecisionTree.hs
**Decision tree classifier with recursive splitting**

A decision tree implementation using information theory:
- **Recursive Trees**: Natural tree structure with Haskell ADTs
- **Entropy**: Measure of dataset impurity
- **Information Gain**: Split quality metric
- **ID3 Algorithm**: Greedy top-down induction
- **Overfitting Prevention**: Max depth and minimum samples

**Complexity**: Intermediate-Advanced  
**Lines**: ~300  
**Key Concepts**: Recursive algorithms, Information theory, Classification

#### Features:
- ✅ Binary decision tree construction
- ✅ Entropy and information gain calculations
- ✅ Best split selection (greedy algorithm)
- ✅ Recursive tree building (ID3/C4.5 style)
- ✅ Tree traversal for prediction
- ✅ Accuracy metrics
- ✅ Overfitting prevention (max depth, min samples)
- ✅ Pretty-printing of tree structure
- ✅ Multiple test datasets (AND, OR, XOR)

#### Tree Structure:
```haskell
data Tree a = Leaf a                          -- Terminal prediction
            | Node Int Double (Tree a) (Tree a)  -- Decision node
  deriving (Show, Eq)
-- Node: feature_index threshold left_tree right_tree
```

#### Usage Examples:
```haskell
-- Create training data
let data = [([1.0, 2.0], "A"), ([5.0, 6.0], "B"), ([2.0, 3.0], "A")]

-- Build tree (maxDepth=5, minSamples=2)
let tree = buildTree data 5 0 2

-- Make prediction
let prediction = predict tree [3.0, 4.0]  -- Returns "A" or "B"

-- Evaluate accuracy
let predictions = predictDataset tree testFeatures
let acc = accuracy predictions actualLabels  -- Returns 0.0 to 1.0
```

#### Example Tree Output:
```
Node: feature[0] <= 3.5
├─ True:
  Leaf: "A"
└─ False:
  Node: feature[1] <= 5.5
  ├─ True:
    Leaf: "B"
  └─ False:
    Leaf: "B"
```

## Additional Extensions for New Projects

### Neural Network Extensions
- [ ] Different optimizers (Momentum, Adam, RMSprop)
- [ ] Batch normalization
- [ ] Dropout for regularization
- [ ] Convolutional layers
- [ ] Different loss functions (cross-entropy, hinge loss)
- [ ] Mini-batch gradient descent
- [ ] Learning rate scheduling
- [ ] Weight initialization strategies (Xavier, He)
- [ ] Validation set evaluation
- [ ] Early stopping

### Decision Tree Extensions
- [ ] Handle categorical features
- [ ] Pruning (pre-pruning and post-pruning)
- [ ] Random Forests (ensemble learning)
- [ ] Feature importance calculation
- [ ] Regression trees (predict continuous values)
- [ ] Multi-way splits (not just binary)
- [ ] Missing value handling
- [ ] Gini impurity (alternative to entropy)
- [ ] Tree visualization (export to graphviz)
- [ ] Cross-validation
- [ ] Gradient boosting
- [ ] Class weights for imbalanced data

## Machine Learning Concepts Demonstrated

### Neural Networks
- **Forward Propagation**: Linear transformations + activations
- **Backpropagation**: Chain rule for gradient computation
- **Gradient Descent**: Iterative optimization
- **Non-linearity**: Activation functions enable complex patterns
- **Universal Approximation**: Networks can learn any function

### Decision Trees
- **Greedy Algorithm**: Locally optimal splits at each node
- **Recursive Partitioning**: Divide-and-conquer approach
- **Information Theory**: Entropy measures uncertainty
- **Overfitting**: Trees can memorize training data
- **Interpretability**: Easy to understand and visualize

## Testing Machine Learning Projects

### Neural Network Testing
```haskell
-- Test XOR (classic non-linear problem)
let xorData = [([0,0], [0]), ([0,1], [1]), ([1,0], [1]), ([1,1], [0])]

-- Train for multiple epochs
let trainLoop net 0 = net
    trainLoop net n = 
      let (newNet, loss) = trainEpoch net xorData 0.5
      in putStrLn ("Epoch " ++ show n ++ ": loss = " ++ show loss) `seq` 
         trainLoop newNet (n-1)

let trained = trainLoop network 1000

-- Test predictions
mapM_ (\(input, target) -> 
  let output = predict trained input
  in print (input, output, target)) xorData
```

### Decision Tree Testing
```haskell
-- Test with AND function
let andData = [([0,0], False), ([0,1], False), ([1,0], False), ([1,1], True)]
let tree = buildTree andData 5 0 1

-- Verify perfect accuracy
let preds = predictDataset tree (map fst andData)
let acc = accuracy preds (map snd andData)
print acc  -- Should be 1.0 (100%)

-- Test with XOR (needs depth > 1)
let xorData = [([0,0], False), ([0,1], True), ([1,0], True), ([1,1], False)]
let xorTree = buildTree xorData 5 0 1
```

## Performance Notes

### Neural Network
- **Time Complexity**: O(L × N × M) per training example
  - L = number of layers
  - N = neurons per layer (average)
  - M = training examples
- **Space Complexity**: O(L × N²) for weights
- **Training Speed**: Can be slow without optimizations
- **Optimization Tips**:
  - Use strict evaluation (`foldl'` instead of `foldl`)
  - Consider batch processing
  - Use optimized BLAS libraries for matrix operations
  - Implement mini-batch gradient descent

### Decision Tree
- **Time Complexity**: O(N × M × log M) for building
  - N = number of features
  - M = number of examples
- **Space Complexity**: O(tree depth × M)
- **Prediction**: O(tree depth) - very fast!
- **Optimization Tips**:
  - Limit max depth (prevents overfitting and speeds up)
  - Use feature sampling (random forests)
  - Cache entropy calculations
  - Use approximate splits for large datasets

## Comparison with Libraries

### Neural Networks
**Our Implementation vs. Popular Libraries:**
- ✅ **Pros**: Educational, transparent, pure functional
- ❌ **Cons**: Slow, no GPU support, limited features
- **Real Libraries**: TensorFlow, PyTorch, MXNet
  - GPU acceleration
  - Automatic differentiation
  - Advanced architectures
  - Production-ready

### Decision Trees
**Our Implementation vs. Popular Libraries:**
- ✅ **Pros**: Simple, understandable, self-contained
- ❌ **Cons**: Basic features only, no advanced pruning
- **Real Libraries**: scikit-learn, XGBoost, LightGBM
  - Advanced pruning techniques
  - Parallel tree building
  - Feature importance
  - Ensemble methods
  - Production optimizations

## When to Use Each Algorithm

### Neural Networks
**Best for:**
- Image recognition
- Natural language processing
- Speech recognition
- Complex non-linear patterns
- Large datasets
- When features are not well-understood

**Avoid when:**
- Small datasets (overfitting risk)
- Need interpretability
- Limited computational resources
- Quick prototyping needed

### Decision Trees
**Best for:**
- Tabular data
- Feature importance analysis
- Interpretable models
- Mixed feature types (categorical + numerical)
- Quick prototyping
- Non-linear relationships

**Avoid when:**
- High-dimensional sparse data
- Need smooth decision boundaries
- Prone to overfitting without ensembles
- Unstable (small changes in data → different tree)

## Learning Outcomes

After completing these projects, you'll understand:

### Neural Networks
- ✅ How neural networks transform inputs to outputs
- ✅ The mathematics behind backpropagation
- ✅ Why activation functions are necessary
- ✅ How gradient descent optimizes weights
- ✅ Implementing numerical algorithms functionally
- ✅ Matrix operations in pure functional style

### Decision Trees
- ✅ How trees partition feature space
- ✅ Information theory basics (entropy, information gain)
- ✅ Greedy algorithms and their limitations
- ✅ Recursive tree construction
- ✅ Overfitting and how to prevent it
- ✅ Tree-based classification

## Real-World Applications

### Neural Networks
- Computer Vision: Object detection, image classification
- NLP: Machine translation, sentiment analysis
- Speech: Recognition, synthesis
- Games: AlphaGo, game playing agents
- Recommendation Systems: Netflix, YouTube
- Autonomous Vehicles: Perception systems

### Decision Trees
- Credit Scoring: Loan approval decisions
- Medical Diagnosis: Disease prediction
- Fraud Detection: Transaction classification
- Customer Segmentation: Marketing analysis
- Recommendation Systems: Product suggestions
- Risk Assessment: Insurance pricing

