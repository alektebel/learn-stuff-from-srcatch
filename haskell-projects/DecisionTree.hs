-- Decision Tree Classifier in Haskell - Template
-- Build a decision tree classifier using recursion to learn from data
--
-- LEARNING OBJECTIVES:
-- 1. Define recursive tree data structures
-- 2. Implement information gain and entropy calculations
-- 3. Use recursion to build trees from data
-- 4. Apply functional patterns for data splitting
-- 5. Implement prediction through tree traversal
--
-- ESTIMATED TIME: 6-8 hours for intermediate, 4-6 hours for advanced

module Main where

import Data.List (group, sort, nub, maximumBy)
import Data.Ord (comparing)

{- |
TODO 1: Define the decision tree data type

CONCEPT: Recursive Tree Structure
A decision tree is a binary (or multi-way) tree where:
- Leaf nodes: Contain class predictions (final decisions)
- Internal nodes: Contain decisions based on features
  - Feature index: which feature to test
  - Threshold: value to compare against
  - Left/Right: subtrees for each outcome

WHY RECURSIVE?
Trees are naturally recursive: each subtree is itself a tree.
This allows elegant implementation using pattern matching.

GUIDELINES:
1. Define a Tree type with two constructors:
   
   a) Leaf: Terminal node with prediction
      - Contains: class label (String or Int)
      - Example: Leaf "Yes" means predict "Yes"
   
   b) Node: Decision node
      - Contains:
        * featureIndex: which feature to test (Int)
        * threshold: value to split on (Double)
        * leftTree: Tree for values <= threshold
        * rightTree: Tree for values > threshold
      - Example: Node 0 5.0 leftTree rightTree
        Tests feature 0, splits at 5.0

2. Derive Show and Eq for debugging and testing

TREE STRUCTURE EXAMPLE:
  Node 0 5.0
    ├─ (≤5.0) Leaf "No"
    └─ (>5.0) Node 1 10.0
        ├─ (≤10.0) Leaf "Maybe"
        └─ (>10.0) Leaf "Yes"

This represents:
  If feature[0] <= 5.0: predict "No"
  Else if feature[1] <= 10.0: predict "Maybe"
  Else: predict "Yes"

TYPES TO DEFINE:
- Tree a: Generic tree with class type 'a'
- Use type parameters for flexibility
-}
-- TODO: Define Tree data type with Leaf and Node constructors
data Tree a = TreeStub
  deriving (Show, Eq)

{- |
TODO 2: Define data representation types

CONCEPT: Dataset Representation
Need types to represent training data:
- Features: Vector of numerical values for one example
- Label: Class/target value for that example
- Dataset: Collection of (features, label) pairs

GUIDELINES:
1. Features: list of Double values
   Example: [5.1, 3.5, 1.4, 0.2] (Iris flower measurements)

2. Label: Generic type (String, Int, etc.)
   Example: "Setosa" or 0 for class

3. DataPoint: pair of (Features, Label)
   Example: ([5.1, 3.5, 1.4, 0.2], "Setosa")

4. Dataset: list of DataPoints
   Example: [([5.1, 3.5], "A"), ([6.2, 2.9], "B"), ...]

TYPE DEFINITIONS:
These create type aliases for clarity
-}
type Features = [Double]
type Label a = a
type DataPoint a = (Features, Label a)
type Dataset a = [DataPoint a]

{- |
TODO 3: Implement entropy calculation

CONCEPT: Entropy (Information Theory)
Entropy measures uncertainty/impurity in a dataset.
- High entropy: many different classes (mixed)
- Low entropy: mostly one class (pure)
- Zero entropy: all same class (completely pure)

FORMULA:
  H(S) = -Σ(p_i * log2(p_i))
  
  where:
  - S is the dataset
  - p_i is proportion of class i
  - log2 is logarithm base 2
  - Sum over all classes

GUIDELINES:

1. Count occurrences of each class
   - Extract all labels from dataset
   - Group identical labels
   - Count size of each group

2. Calculate proportions
   - Divide each count by total examples
   - Result: probability of each class

3. Apply entropy formula
   - For each probability p:
     * If p > 0: add -p * log2(p)
     * If p = 0: add 0 (since lim p→0 of p*log(p) = 0)
   - Sum all terms

IMPLEMENTATION TIPS:
- Use map and fold for calculations
- log base 2 = log(x) / log(2) in Haskell
- Filter out zero probabilities before log
- Empty dataset has entropy 0 by convention

EXAMPLES:
  Dataset: ["A", "A", "A", "A"]
    All same class: entropy = 0

  Dataset: ["A", "B", "A", "B"]
    50-50 split: entropy = 1.0

  Dataset: ["A", "A", "B"]
    2/3 A, 1/3 B: entropy ≈ 0.918

  Dataset: ["A", "B", "C", "D"]
    All different: entropy = 2.0

EDGE CASES:
- Empty dataset: return 0
- Single example: entropy = 0
- All same class: entropy = 0
-}
entropy :: (Ord a) => [a] -> Double
entropy labels
  | null labels = 0
  | otherwise = undefined
    -- TODO: Implement entropy calculation
    -- Steps:
    -- 1. Count occurrences: group . sort
    -- 2. Calculate proportions: map (/ total)
    -- 3. Apply formula: sum [-p * log2(p) | p <- proportions, p > 0]

{- |
TODO 4: Implement information gain calculation

CONCEPT: Information Gain
Information gain measures how much a split reduces entropy.
We want splits that maximize information gain (reduce uncertainty).

FORMULA:
  IG(S, feature, threshold) = H(S) - Weighted_Average(H(S_left), H(S_right))
  
  where:
  - H(S) is entropy of original dataset
  - S_left: examples with feature <= threshold
  - S_right: examples with feature > threshold
  - Weighted average: (|S_left|/|S|)*H(S_left) + (|S_right|/|S|)*H(S_right)

GUIDELINES:

1. Calculate parent entropy
   - Call entropy on all labels

2. Split dataset
   - Separate into left (≤ threshold) and right (> threshold)
   - Based on specific feature index

3. Calculate child entropies
   - Compute entropy for left split
   - Compute entropy for right split

4. Calculate weighted average
   - Weight each by proportion of examples
   - left_weight = |left| / |total|
   - right_weight = |right| / |total|

5. Compute information gain
   - IG = parent_entropy - weighted_child_entropy
   - Higher is better (more information gained)

IMPLEMENTATION APPROACH:
  informationGain dataset featureIndex threshold =
    let parent_entropy = entropy (labels of dataset)
        (left, right) = split dataset on featureIndex at threshold
        left_entropy = entropy (labels of left)
        right_entropy = entropy (labels of right)
        total = length dataset
        left_weight = (length left) / total
        right_weight = (length right) / total
        weighted = left_weight * left_entropy + right_weight * right_entropy
    in parent_entropy - weighted

EXAMPLES:
  Dataset: [(5, "A"), (10, "B"), (15, "A"), (20, "B")]
  
  Split at threshold 12.5:
    Left: [(5, "A"), (10, "B")]  - entropy = 1.0
    Right: [(15, "A"), (20, "B")] - entropy = 1.0
    IG = H(all) - 0.5*1.0 - 0.5*1.0 = 0 (bad split)
  
  Split at threshold 7.5:
    Left: [(5, "A")]  - entropy = 0.0
    Right: [(10, "B"), (15, "A"), (20, "B")] - entropy ≈ 0.918
    IG = H(all) - 0.25*0.0 - 0.75*0.918 > 0 (better split)

EDGE CASES:
- Empty splits: entropy = 0
- Split puts all in one side: IG may be 0
-}
informationGain :: (Ord a) => Dataset a -> Int -> Double -> Double
informationGain dataset featureIndex threshold = undefined
  -- TODO: Implement information gain
  -- 1. Extract labels from dataset
  -- 2. Calculate parent entropy
  -- 3. Split dataset: (left, right) = partition by threshold
  -- 4. Calculate child entropies
  -- 5. Return parent_entropy - weighted_child_entropy

{- |
TODO 5: Implement dataset splitting

CONCEPT: Dataset Partitioning
Split data into two groups based on a feature value.

GUIDELINES:

1. splitDataset: Split on feature and threshold
   Input: dataset, feature index, threshold
   Output: (left dataset, right dataset)
   Logic:
   - left: examples where features[featureIndex] <= threshold
   - right: examples where features[featureIndex] > threshold
   
   Implementation:
   - Use partition or filter
   - Access feature: (features !! featureIndex)

2. Helper: Extract specific feature value
   getFeatureValue :: Features -> Int -> Double
   Simply index into the features list

EXAMPLES:
  dataset = [([1.0, 2.0], "A"), ([3.0, 4.0], "B"), ([5.0, 6.0], "C")]
  splitDataset dataset 0 3.0
    → left:  [([1.0, 2.0], "A")]
    → right: [([3.0, 4.0], "B"), ([5.0, 6.0], "C")]

  splitDataset dataset 1 5.0
    → left:  [([1.0, 2.0], "A"), ([3.0, 4.0], "B")]
    → right: [([5.0, 6.0], "C")]

IMPLEMENTATION TIPS:
- Use Haskell's partition function
- Define predicate: (\(features, _) -> features !! featureIndex <= threshold)
- Handle edge cases: empty dataset, invalid feature index
-}
splitDataset :: Dataset a -> Int -> Double -> (Dataset a, Dataset a)
splitDataset dataset featureIndex threshold = undefined
  -- TODO: Implement dataset splitting
  -- Use: partition (\(features, _) -> features !! featureIndex <= threshold) dataset

{- |
TODO 6: Implement finding best split

CONCEPT: Optimal Split Selection
Find the feature and threshold that maximize information gain.

STRATEGY:
1. Try all features (each column)
2. For each feature, try multiple thresholds
3. Compute information gain for each (feature, threshold) pair
4. Return the pair with maximum information gain

GUIDELINES:

1. possibleThresholds: Generate threshold candidates for a feature
   Input: dataset, feature index
   Output: list of threshold values to try
   Approach:
   - Extract all values of that feature
   - Sort them
   - Use midpoints between consecutive values
   - Example: values [1, 3, 5, 7] → thresholds [2, 4, 6]
   
   Why midpoints? 
   - Values between actual data points are natural split points
   - Reduces number of thresholds to try (efficient)

2. findBestSplit: Find optimal split
   Input: dataset
   Output: Maybe (feature index, threshold, information gain)
   - Returns Nothing if no good split exists
   - Returns Just (feat, thresh, gain) for best split
   
   Algorithm:
   - For each feature:
       For each threshold:
         Calculate information gain
   - Track maximum gain and corresponding (feature, threshold)
   - Return best found

IMPLEMENTATION APPROACH:
  findBestSplit dataset =
    let numFeatures = length (features of first example)
        allSplits = [(feat, thresh, gain) | 
                     feat <- [0..numFeatures-1],
                     thresh <- possibleThresholds dataset feat,
                     let gain = informationGain dataset feat thresh]
    in if null allSplits 
       then Nothing
       else Just (maximumBy (comparing gain) allSplits)

OPTIMIZATION TIPS:
- Don't try every possible value (too slow)
- Use midpoints or quantiles
- Can limit to top K features (feature selection)
- Skip features with no variance

EXAMPLES:
  dataset = [([1.0, 5.0], "A"), ([2.0, 10.0], "B"), ([3.0, 15.0], "A")]
  
  Feature 0 thresholds: [1.5, 2.5]
  Feature 1 thresholds: [7.5, 12.5]
  
  Try all 4 combinations, return best
-}
possibleThresholds :: Dataset a -> Int -> [Double]
possibleThresholds dataset featureIndex = undefined
  -- TODO: Generate threshold candidates
  -- 1. Extract feature values: map (!! featureIndex . fst) dataset
  -- 2. Sort values
  -- 3. Generate midpoints: zipWith avg consecutive_pairs
  -- 4. Remove duplicates

findBestSplit :: (Ord a) => Dataset a -> Maybe (Int, Double, Double)
findBestSplit dataset = undefined
  -- TODO: Find best (feature, threshold) split
  -- 1. Get number of features
  -- 2. Generate all (feature, threshold) pairs
  -- 3. Calculate information gain for each
  -- 4. Return pair with maximum gain

{- |
TODO 7: Implement majority vote for classification

CONCEPT: Majority Class
When creating a leaf node, predict the most common class in the data.

GUIDELINES:

1. Count occurrences of each class
2. Find class with maximum count
3. Return that class

IMPLEMENTATION:
- Extract all labels
- Group and count: map (head &&& length) . group . sort
- Find maximum by count
- Return the label

EXAMPLES:
  labels = ["A", "B", "A", "A", "C"]
    Counts: A→3, B→1, C→1
    Majority: "A"

  labels = ["X", "Y", "X", "Y"]
    Counts: X→2, Y→2
    Majority: either "X" or "Y" (tie)

EDGE CASES:
- Empty list: undefined or error
- Tie: return any (or first alphabetically)
- Single element: return it
-}
majorityClass :: (Ord a) => [a] -> a
majorityClass labels = undefined
  -- TODO: Find most common element
  -- Use: head . maximumBy (comparing length) . group . sort

{- |
TODO 8: Implement tree building (main recursive function)

CONCEPT: Recursive Tree Construction (ID3/C4.5 Algorithm)
Build tree top-down by recursively splitting data.

ALGORITHM:
  buildTree(dataset, maxDepth, currentDepth):
    # Base cases - create leaf
    if currentDepth >= maxDepth:
      return Leaf(majorityClass(dataset))
    if all examples have same class:
      return Leaf(that class)
    if no features left or dataset too small:
      return Leaf(majorityClass(dataset))
    
    # Recursive case - create decision node
    (feature, threshold) = findBestSplit(dataset)
    (leftData, rightData) = splitDataset(dataset, feature, threshold)
    
    leftTree = buildTree(leftData, maxDepth, currentDepth + 1)
    rightTree = buildTree(rightData, maxDepth, currentDepth + 1)
    
    return Node(feature, threshold, leftTree, rightTree)

GUIDELINES:

1. Base cases (return Leaf):
   a) Maximum depth reached
      - Prevents overfitting
      - Typical max depth: 5-20
   
   b) Pure node (all same class)
      - entropy = 0
      - No point splitting further
   
   c) Too few examples
      - e.g., < 2 examples
      - Can't meaningfully split
   
   d) No valid split found
      - All features identical
      - Or information gain ≤ 0

2. Recursive case (return Node):
   - Find best split
   - Partition data
   - Recursively build left subtree
   - Recursively build right subtree
   - Create Node with feature, threshold, and subtrees

3. Parameters:
   - maxDepth: limit tree depth (prevent overfitting)
   - minSamples: minimum examples to split (e.g., 2)

IMPLEMENTATION STRUCTURE:
  buildTree dataset maxDepth currentDepth minSamples
    -- Base case 1: max depth reached
    | currentDepth >= maxDepth = Leaf (majorityClass labels)
    
    -- Base case 2: pure node
    | isPure dataset = Leaf (head labels)
    
    -- Base case 3: too few samples
    | length dataset < minSamples = Leaf (majorityClass labels)
    
    -- Recursive case
    | otherwise = 
        case findBestSplit dataset of
          Nothing -> Leaf (majorityClass labels)
          Just (feat, thresh, _) ->
            let (left, right) = splitDataset dataset feat thresh
                leftTree = buildTree left maxDepth (currentDepth + 1) minSamples
                rightTree = buildTree right maxDepth (currentDepth + 1) minSamples
            in Node feat thresh leftTree rightTree
    where
      labels = map snd dataset

WHY RECURSION?
- Trees are recursive structures
- Each subtree is built the same way
- Base cases handle termination
- Natural and elegant in functional programming

EXAMPLES:
  Simple dataset:
    [([1], "A"), ([2], "A"), ([5], "B"), ([6], "B")]
  
  Build tree with maxDepth=2:
    Best split: feature 0, threshold 3.5
    Left: [([1], "A"), ([2], "A")] → all "A" → Leaf "A"
    Right: [([5], "B"), ([6], "B")] → all "B" → Leaf "B"
    Result: Node 0 3.5 (Leaf "A") (Leaf "B")

COMMON ISSUES:
- Infinite recursion: ensure base cases are correct
- Empty splits: one side gets all data (check for this)
- Overfitting: tree too deep (use maxDepth)
- Underfitting: tree too shallow (increase maxDepth)
-}
isPure :: (Eq a) => Dataset a -> Bool
isPure dataset = undefined
  -- TODO: Check if all labels are the same
  -- Hint: length (nub labels) == 1

buildTree :: (Ord a) => Dataset a -> Int -> Int -> Int -> Tree a
buildTree dataset maxDepth currentDepth minSamples = undefined
  -- TODO: Implement recursive tree building
  -- Follow the algorithm described above

{- |
TODO 9: Implement prediction

CONCEPT: Tree Traversal for Classification
Walk down the tree using feature values until reaching a leaf.

ALGORITHM:
  predict(tree, features):
    case tree of
      Leaf class → return class
      Node featureIdx threshold leftTree rightTree →
        if features[featureIdx] <= threshold:
          predict(leftTree, features)
        else:
          predict(rightTree, features)

GUIDELINES:

1. Base case: Leaf node
   - Return the class label stored in leaf

2. Recursive case: Node
   - Extract relevant feature value
   - Compare with threshold
   - Recurse on appropriate subtree

IMPLEMENTATION:
  predict :: Tree a -> Features -> a
  predict (Leaf cls) _ = cls
  predict (Node featIdx thresh left right) features =
    if features !! featIdx <= thresh
      then predict left features
      else predict right features

EXAMPLES:
  Tree: Node 0 5.0 (Leaf "A") (Leaf "B")
  
  predict tree [3.0, 1.0]:
    3.0 <= 5.0 → go left → return "A"
  
  predict tree [7.0, 2.0]:
    7.0 > 5.0 → go right → return "B"

EDGE CASES:
- Empty features: would cause error (assume valid input)
- Feature index out of bounds: would error (assume valid tree)

USAGE:
  Once tree is built, use predict for all new examples
  This is much faster than training (just tree traversal)
-}
predict :: Tree a -> Features -> a
predict tree features = undefined
  -- TODO: Implement prediction by tree traversal

{- |
TODO 10: Implement batch prediction and evaluation

CONCEPT: Model Evaluation
Predict multiple examples and calculate accuracy.

GUIDELINES:

1. predictDataset: Predict for multiple examples
   Input: tree, list of feature vectors
   Output: list of predictions
   Implementation: map (predict tree) over features

2. accuracy: Calculate prediction accuracy
   Input: predictions, actual labels
   Output: accuracy percentage (0 to 1)
   Formula: (correct predictions) / (total predictions)
   Implementation:
   - Zip predictions with actuals
   - Count matches
   - Divide by total

3. confusionMatrix: Build confusion matrix (optional, advanced)
   Shows true positives, false positives, etc.

EXAMPLES:
  predictions = ["A", "B", "A", "B"]
  actuals =     ["A", "B", "B", "B"]
  accuracy = 3/4 = 0.75 (75%)

IMPLEMENTATION:
  accuracy predictions actuals =
    let correct = length $ filter (uncurry (==)) $ zip predictions actuals
        total = length actuals
    in fromIntegral correct / fromIntegral total
-}
predictDataset :: Tree a -> [Features] -> [a]
predictDataset tree featuresLi = undefined
  -- TODO: Predict for multiple examples

accuracy :: (Eq a) => [a] -> [a] -> Double
accuracy predictions actuals = undefined
  -- TODO: Calculate accuracy percentage

{- |
IMPLEMENTATION GUIDE
--------------------

Step-by-step approach:

1. START WITH DATA STRUCTURES (TODO 1-2)
   - Define Tree type
   - Define data types
   - Test: Create tree manually

2. IMPLEMENT ENTROPY (TODO 3)
   - Most fundamental calculation
   - Test with known examples
   - Verify: all same → 0, half-half → 1.0

3. IMPLEMENT INFORMATION GAIN (TODO 4)
   - Builds on entropy
   - Test with simple splits
   - Verify: good splits have high gain

4. IMPLEMENT SPLITTING (TODO 5)
   - Simple data partitioning
   - Test: verify left/right splits correct

5. IMPLEMENT BEST SPLIT FINDER (TODO 6)
   - Combines previous functions
   - Test: finds expected best split

6. IMPLEMENT MAJORITY VOTE (TODO 7)
   - Simple utility function
   - Test with various label lists

7. IMPLEMENT TREE BUILDING (TODO 8)
   - Most complex - take your time
   - Start with simple dataset
   - Test: verify tree structure makes sense

8. IMPLEMENT PREDICTION (TODO 9)
   - Tree traversal
   - Test: manual tree, known predictions

9. IMPLEMENT EVALUATION (TODO 10)
   - Batch prediction and metrics
   - Test: verify accuracy calculation

TESTING STRATEGY:

1. Unit Tests:
   - Test each function with known inputs
   - Verify edge cases

2. Simple Dataset:
   AND function:
   [([0,0], False), ([0,1], False), ([1,0], False), ([1,1], True)]
   Should build simple tree

3. Classic Dataset:
   Iris or simple classification
   Verify reasonable accuracy

DEBUGGING TIPS:

1. Print tree structure (Show instance helps)
2. Verify splits are logical
3. Check if tree is too deep or too shallow
4. Test entropy calculations independently
5. Verify information gain is positive

COMMON PITFALLS:

1. Off-by-one errors in feature indexing
2. Infinite recursion (check base cases)
3. Empty dataset handling
4. Division by zero in entropy
5. Not handling ties in majority vote

TEST DATASETS:

1. Simple AND/OR gates:
   Easy to verify correctness

2. XOR (requires depth > 1):
   Tests if tree can handle non-linearly separable data

3. Iris dataset:
   Classic machine learning benchmark
   3 classes, 4 features

EXTENSIONS (after basic implementation):

1. Handle categorical features
2. Implement pruning (reduce overfitting)
3. Add random forests (ensemble of trees)
4. Support regression (predict continuous values)
5. Implement missing value handling
6. Add feature importance calculation
7. Visualize tree structure
8. Support multi-way splits (not just binary)
9. Implement cost-complexity pruning
10. Add cross-validation
-}

-- Example usage and tests
main :: IO ()
main = do
  putStrLn "Decision Tree Classifier in Haskell - Template\n"
  putStrLn "This is a template with detailed implementation guidelines."
  putStrLn "Follow the TODOs above to implement a working decision tree.\n"
  
  putStrLn "Example dataset (AND function):"
  let andData = [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]
  print andData
  putStrLn ""
  
  putStrLn "Example dataset (simple classification):"
  let simpleData = [([1.0, 2.0], "A"), ([2.0, 3.0], "A"), 
                    ([5.0, 6.0], "B"), ([6.0, 7.0], "B")]
  print simpleData
  putStrLn ""
  
  putStrLn "Once implemented, you can:"
  putStrLn "  1. Build tree: buildTree dataset 5 0 2"
  putStrLn "  2. Predict: predict tree [3.0, 4.0]"
  putStrLn "  3. Evaluate: accuracy predictions actuals\n"
  
  putStrLn "Key concepts:"
  putStrLn "  - Entropy: measures impurity"
  putStrLn "  - Information Gain: measures split quality"
  putStrLn "  - Recursion: builds tree top-down"
  putStrLn "  - Greedy: locally optimal splits\n"
  
  putStrLn "Complete the TODOs above to build a working decision tree!"
  putStrLn "Start with TODO 1 and work through sequentially."
