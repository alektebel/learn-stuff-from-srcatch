-- Decision Tree Classifier in Haskell - Complete Solution

module Main where

import Data.List (group, sort, nub, maximumBy, partition)
import Data.Ord (comparing)

-- Data type for decision tree
data Tree a = Leaf a
            | Node Int Double (Tree a) (Tree a)
  deriving (Show, Eq)

-- Type aliases
type Features = [Double]
type Label a = a
type DataPoint a = (Features, Label a)
type Dataset a = [DataPoint a]

-- Entropy calculation
entropy :: (Ord a) => [a] -> Double
entropy labels
  | null labels = 0
  | otherwise = 
      let total = fromIntegral $ length labels
          counts = map length $ group $ sort labels
          proportions = map (\c -> fromIntegral c / total) counts
          terms = filter (> 0) proportions
      in negate $ sum [p * logBase 2 p | p <- terms]

-- Information gain calculation
informationGain :: (Ord a) => Dataset a -> Int -> Double -> Double
informationGain dataset featureIndex threshold = 
  let labels = map snd dataset
      parentEntropy = entropy labels
      (left, right) = splitDataset dataset featureIndex threshold
      total = fromIntegral $ length dataset
      leftWeight = fromIntegral (length left) / total
      rightWeight = fromIntegral (length right) / total
      leftEntropy = entropy (map snd left)
      rightEntropy = entropy (map snd right)
      weightedChildEntropy = leftWeight * leftEntropy + rightWeight * rightEntropy
  in parentEntropy - weightedChildEntropy

-- Dataset splitting
splitDataset :: Dataset a -> Int -> Double -> (Dataset a, Dataset a)
splitDataset dataset featureIndex threshold = 
  partition (\(features, _) -> features !! featureIndex <= threshold) dataset

-- Find best split
possibleThresholds :: Dataset a -> Int -> [Double]
possibleThresholds dataset featureIndex = 
  let values = nub $ sort $ map ((!! featureIndex) . fst) dataset
      midpoints = zipWith (\a b -> (a + b) / 2) values (tail values)
  in midpoints

findBestSplit :: (Ord a) => Dataset a -> Maybe (Int, Double, Double)
findBestSplit dataset 
  | null dataset = Nothing
  | otherwise = 
      let numFeatures = length $ fst $ head dataset
          allSplits = [(feat, thresh, informationGain dataset feat thresh) 
                      | feat <- [0..numFeatures-1]
                      , thresh <- possibleThresholds dataset feat
                      ]
      in if null allSplits
         then Nothing
         else Just $ maximumBy (comparing (\(_, _, gain) -> gain)) allSplits

-- Majority class
majorityClass :: (Ord a) => [a] -> a
majorityClass labels = 
  head $ maximumBy (comparing length) $ group $ sort labels

-- Check if dataset is pure
isPure :: (Eq a) => Dataset a -> Bool
isPure dataset = 
  let labels = map snd dataset
  in length (nub labels) == 1

-- Build decision tree recursively
buildTree :: (Ord a) => Dataset a -> Int -> Int -> Int -> Tree a
buildTree dataset maxDepth currentDepth minSamples
  | currentDepth >= maxDepth = Leaf (majorityClass labels)
  | isPure dataset = Leaf (head labels)
  | length dataset < minSamples = Leaf (majorityClass labels)
  | otherwise = 
      case findBestSplit dataset of
        Nothing -> Leaf (majorityClass labels)
        Just (feat, thresh, gain) ->
          if gain <= 0
            then Leaf (majorityClass labels)
            else
              let (left, right) = splitDataset dataset feat thresh
              in if null left || null right
                 then Leaf (majorityClass labels)
                 else
                   let leftTree = buildTree left maxDepth (currentDepth + 1) minSamples
                       rightTree = buildTree right maxDepth (currentDepth + 1) minSamples
                   in Node feat thresh leftTree rightTree
  where
    labels = map snd dataset

-- Prediction
predict :: Tree a -> Features -> a
predict (Leaf cls) _ = cls
predict (Node featIdx thresh left right) features =
  if features !! featIdx <= thresh
    then predict left features
    else predict right features

-- Batch prediction
predictDataset :: Tree a -> [Features] -> [a]
predictDataset tree = map (predict tree)

-- Accuracy calculation
accuracy :: (Eq a) => [a] -> [a] -> Double
accuracy predictions actuals = 
  let correct = length $ filter (uncurry (==)) $ zip predictions actuals
      total = length actuals
  in fromIntegral correct / fromIntegral total

-- Pretty print tree
printTree :: (Show a) => Tree a -> Int -> String
printTree (Leaf cls) indent = 
  replicate (indent * 2) ' ' ++ "Leaf: " ++ show cls
printTree (Node feat thresh left right) indent = 
  replicate (indent * 2) ' ' ++ "Node: feature[" ++ show feat ++ "] <= " ++ show thresh ++ "\n" ++
  replicate (indent * 2) ' ' ++ "├─ True:\n" ++ printTree left (indent + 1) ++ "\n" ++
  replicate (indent * 2) ' ' ++ "└─ False:\n" ++ printTree right (indent + 1)

-- Main with examples
main :: IO ()
main = do
  putStrLn "Decision Tree Classifier in Haskell - Complete Solution\n"
  
  -- Test 1: Simple AND function
  putStrLn "=== Test 1: AND Function ==="
  let andData = [([0, 0], False), ([0, 1], False), ([1, 0], False), ([1, 1], True)]
  putStrLn "Training data:"
  mapM_ print andData
  
  let andTree = buildTree andData 5 0 1
  putStrLn "\nLearned tree:"
  putStrLn $ printTree andTree 0
  
  putStrLn "\nPredictions:"
  mapM_ (\(features, actual) -> 
    let pred = predict andTree features
    in putStrLn $ "  " ++ show features ++ " -> " ++ show pred ++ " (actual: " ++ show actual ++ ")")
    andData
  
  let andPreds = predictDataset andTree (map fst andData)
  let andAcc = accuracy andPreds (map snd andData)
  putStrLn $ "\nAccuracy: " ++ show (andAcc * 100) ++ "%\n"
  
  -- Test 2: Simple classification
  putStrLn "=== Test 2: Simple Classification ==="
  let simpleData = 
        [ ([1.0, 2.0], "A")
        , ([2.0, 3.0], "A")
        , ([1.5, 2.5], "A")
        , ([5.0, 6.0], "B")
        , ([6.0, 7.0], "B")
        , ([5.5, 6.5], "B")
        ]
  
  putStrLn "Training data:"
  mapM_ print simpleData
  
  let simpleTree = buildTree simpleData 5 0 1
  putStrLn "\nLearned tree:"
  putStrLn $ printTree simpleTree 0
  
  putStrLn "\nPredictions:"
  mapM_ (\(features, actual) -> 
    let pred = predict simpleTree features
    in putStrLn $ "  " ++ show features ++ " -> " ++ show pred ++ " (actual: " ++ show actual ++ ")")
    simpleData
  
  let simplePreds = predictDataset simpleTree (map fst simpleData)
  let simpleAcc = accuracy simplePreds (map snd simpleData)
  putStrLn $ "\nAccuracy: " ++ show (simpleAcc * 100) ++ "%\n"
  
  -- Test 3: XOR (non-linearly separable)
  putStrLn "=== Test 3: XOR Function ==="
  let xorData = [([0, 0], False), ([0, 1], True), ([1, 0], True), ([1, 1], False)]
  putStrLn "Training data:"
  mapM_ print xorData
  
  let xorTree = buildTree xorData 5 0 1
  putStrLn "\nLearned tree:"
  putStrLn $ printTree xorTree 0
  
  putStrLn "\nPredictions:"
  mapM_ (\(features, actual) -> 
    let pred = predict xorTree features
    in putStrLn $ "  " ++ show features ++ " -> " ++ show pred ++ " (actual: " ++ show actual ++ ")")
    xorData
  
  let xorPreds = predictDataset xorTree (map fst xorData)
  let xorAcc = accuracy xorPreds (map snd xorData)
  putStrLn $ "\nAccuracy: " ++ show (xorAcc * 100) ++ "%\n"
  
  -- Test entropy function
  putStrLn "=== Entropy Tests ==="
  putStrLn $ "Entropy of [A, A, A, A]: " ++ show (entropy ["A", "A", "A", "A"])
  putStrLn $ "Entropy of [A, B, A, B]: " ++ show (entropy ["A", "B", "A", "B"])
  putStrLn $ "Entropy of [A, A, B]: " ++ show (entropy ["A", "A", "B"])
  putStrLn $ "Entropy of [A, B, C, D]: " ++ show (entropy ["A", "B", "C", "D"])
