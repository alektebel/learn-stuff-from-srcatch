-- Neural Network in Haskell - Complete Solution

module Main where

import Data.List (foldl')

-- Type definitions
type Matrix = [[Double]]
type Vector = [Double]

data Activation = Sigmoid | Tanh | ReLU | Identity
  deriving (Show, Eq)

data Layer = Layer 
  { weights :: Matrix
  , biases :: Vector
  , activation :: Activation
  } deriving (Show)

type Network = [Layer]

-- Activation functions and derivatives
applyActivation :: Activation -> Double -> Double
applyActivation Sigmoid x = 1 / (1 + exp (-x))
applyActivation Tanh x = tanh x
applyActivation ReLU x = max 0 x
applyActivation Identity x = x

activationDerivative :: Activation -> Double -> Double
activationDerivative Sigmoid x = 
  let s = applyActivation Sigmoid x
  in s * (1 - s)
activationDerivative Tanh x = 
  let t = tanh x
  in 1 - t * t
activationDerivative ReLU x = if x > 0 then 1 else 0
activationDerivative Identity _ = 1

-- Matrix and vector operations
dotProduct :: Vector -> Vector -> Double
dotProduct v1 v2 = sum $ zipWith (*) v1 v2

matrixVectorMult :: Matrix -> Vector -> Vector
matrixVectorMult m v = map (`dotProduct` v) m

vectorAdd :: Vector -> Vector -> Vector
vectorAdd = zipWith (+)

vectorSub :: Vector -> Vector -> Vector
vectorSub = zipWith (-)

vectorScale :: Double -> Vector -> Vector
vectorScale k v = map (* k) v

outerProduct :: Vector -> Vector -> Matrix
outerProduct v1 v2 = [[x * y | y <- v2] | x <- v1]

matrixAdd :: Matrix -> Matrix -> Matrix
matrixAdd = zipWith (zipWith (+))

matrixSub :: Matrix -> Matrix -> Matrix
matrixSub = zipWith (zipWith (-))

matrixScale :: Double -> Matrix -> Matrix
matrixScale k m = map (map (* k)) m

transposeMatrix :: Matrix -> Matrix
transposeMatrix ([]:_) = []
transposeMatrix m = map head m : transposeMatrix (map tail m)

-- Forward propagation
forwardLayer :: Layer -> Vector -> (Vector, Vector)
forwardLayer layer input = 
  let z = vectorAdd (matrixVectorMult (weights layer) input) (biases layer)
      a = map (applyActivation (activation layer)) z
  in (a, z)

forward :: Network -> Vector -> (Vector, [(Vector, Vector)])
forward network input = 
  let (finalOutput, layerOutputs) = foldl' processLayer (input, []) network
  in (finalOutput, reverse layerOutputs)
  where
    processLayer (currentInput, acc) layer = 
      let (output, preActivation) = forwardLayer layer currentInput
      in (output, (output, preActivation) : acc)

-- Backpropagation
outputDelta :: Vector -> Vector -> Activation -> Vector -> Vector
outputDelta predicted target activation preActivation = 
  let errors = vectorSub predicted target
      derivatives = map (activationDerivative activation) preActivation
  in zipWith (*) errors derivatives

backpropLayer :: Vector -> Matrix -> Vector -> Activation -> Vector
backpropLayer delta nextWeights preActivation activation = 
  let transposed = transposeMatrix nextWeights
      propagated = matrixVectorMult transposed delta
      derivatives = map (activationDerivative activation) preActivation
  in zipWith (*) propagated derivatives

computeGradients :: Vector -> Vector -> (Matrix, Vector)
computeGradients delta prevActivation = 
  (outerProduct delta prevActivation, delta)

backpropagate :: Network -> Vector -> Vector -> (Vector, [(Vector, Vector)]) -> [(Matrix, Vector)]
backpropagate network input target (output, layerOutputs) = 
  let layers = reverse network
      outputs = reverse layerOutputs
      outputLayer = head layers
      (_, lastZ) = head outputs
      
      -- Initial delta for output layer
      initialDelta = outputDelta output target (activation outputLayer) lastZ
      
      -- Get activations (input to each layer)
      activations = input : map fst (init (reverse layerOutputs))
      
      -- Compute all deltas
      deltas = reverse $ scanl computeDelta initialDelta (zip (tail layers) (tail outputs))
        where
          computeDelta prevDelta (layer, (_, z)) = 
            backpropLayer prevDelta (weights layer) z (activation layer)
      
      -- Add initial delta
      allDeltas = initialDelta : deltas
      
  in zipWith computeGradients allDeltas activations

-- Training functions
meanSquaredError :: Vector -> Vector -> Double
meanSquaredError predicted target = 
  let errors = vectorSub predicted target
      squaredErrors = map (** 2) errors
  in sum squaredErrors / (2 * fromIntegral (length errors))

updateLayer :: Layer -> (Matrix, Vector) -> Double -> Layer
updateLayer layer (gradWeights, gradBiases) learningRate = 
  let newWeights = matrixSub (weights layer) (matrixScale learningRate gradWeights)
      newBiases = vectorSub (biases layer) (vectorScale learningRate gradBiases)
  in layer { weights = newWeights, biases = newBiases }

updateNetwork :: Network -> [(Matrix, Vector)] -> Double -> Network
updateNetwork network gradients learningRate = 
  zipWith (\l g -> updateLayer l g learningRate) network gradients

trainSingle :: Network -> Vector -> Vector -> Double -> (Network, Double)
trainSingle network input target learningRate = 
  let forwardResult@(output, _) = forward network input
      loss = meanSquaredError output target
      gradients = backpropagate network input target forwardResult
      newNetwork = updateNetwork network gradients learningRate
  in (newNetwork, loss)

trainEpoch :: Network -> [(Vector, Vector)] -> Double -> (Network, Double)
trainEpoch network trainingData learningRate = 
  let (finalNetwork, totalLoss) = foldl' trainExample (network, 0) trainingData
      avgLoss = totalLoss / fromIntegral (length trainingData)
  in (finalNetwork, avgLoss)
  where
    trainExample (net, lossAcc) (input, target) = 
      let (newNet, loss) = trainSingle net input target learningRate
      in (newNet, lossAcc + loss)

-- Network initialization
initializeLayer :: Int -> Int -> Activation -> Layer
initializeLayer inputSize outputSize activation = 
  let w = [[0.1 * fromIntegral (i + j + 1) | j <- [1..inputSize]] | i <- [1..outputSize]]
      b = [0.01 | _ <- [1..outputSize]]
  in Layer w b activation

createNetwork :: [(Int, Int, Activation)] -> Network
createNetwork specs = map (\(i, o, a) -> initializeLayer i o a) specs

predict :: Network -> Vector -> Vector
predict network input = fst $ forward network input

-- Main with examples
main :: IO ()
main = do
  putStrLn "Neural Network in Haskell - Complete Solution\n"
  
  -- Create a simple 2-3-1 network for XOR problem
  putStrLn "Creating network: 2 inputs -> 3 hidden (Sigmoid) -> 1 output (Sigmoid)"
  let network = createNetwork [(2, 3, Sigmoid), (3, 1, Sigmoid)]
  
  -- XOR training data
  let trainingData = 
        [ ([0, 0], [0])
        , ([0, 1], [1])
        , ([1, 0], [1])
        , ([1, 1], [0])
        ]
  
  putStrLn "\nTraining on XOR problem..."
  putStrLn "Epoch | Average Loss"
  putStrLn "------|-------------"
  
  -- Train for a few epochs
  let trainedNetwork = trainMultipleEpochs network trainingData 0.5 10
  
  putStrLn "\nTesting predictions:"
  mapM_ (testPrediction trainedNetwork) trainingData
  
  where
    trainMultipleEpochs net dat lr epochs = 
      foldl' (\n epoch -> 
        let (newNet, loss) = trainEpoch n dat lr
        in if epoch `mod` 10 == 0 || epoch < 5
           then putStrLn (show epoch ++ "     | " ++ show loss) `seq` newNet
           else newNet
      ) net [1..epochs]
    
    testPrediction net (input, target) = 
      let output = predict net input
          rounded = map (\x -> if x > 0.5 then 1.0 else 0.0) output
      in putStrLn $ "  " ++ show input ++ " -> " ++ show output ++ " (target: " ++ show target ++ ")"
