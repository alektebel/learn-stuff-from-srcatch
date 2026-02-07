-- Neural Network in Haskell - Template
-- Build a simple feedforward neural network with backpropagation from scratch
--
-- LEARNING OBJECTIVES:
-- 1. Model neural network layers using algebraic data types
-- 2. Implement matrix operations functionally
-- 3. Understand forward propagation through layers
-- 4. Implement backpropagation algorithm
-- 5. Apply functional composition for network operations
--
-- ESTIMATED TIME: 8-12 hours for intermediate, 6-8 hours for advanced

module Main where

import Data.List (foldl')

{- |
TODO 1: Define the data types for neural network

CONCEPT: Neural Network Structure
A neural network consists of:
- Layers: Each layer has weights, biases, and an activation function
- Network: A sequence of layers that transform input to output
- Weights: Matrices that transform data between layers
- Biases: Vectors added after weight multiplication
- Activations: Non-linear functions (sigmoid, tanh, ReLU)

GUIDELINES:
1. Define a Matrix type as a list of lists: [[Double]]
   This represents a 2D array of floating-point numbers
   Example: [[1.0, 2.0], [3.0, 4.0]] is a 2x2 matrix

2. Define a Vector type as a list: [Double]
   This represents a 1D array
   Example: [1.0, 2.0, 3.0] is a 3-element vector

3. Define an Activation type with constructors for:
   - Sigmoid: Maps values to range (0, 1)
   - Tanh: Maps values to range (-1, 1)  
   - ReLU: Returns max(0, x)
   - Identity: Returns x unchanged

4. Define a Layer type that contains:
   - weights: Matrix of weights
   - biases: Vector of biases
   - activation: Activation function type

5. Define a Network type as a list of layers: [Layer]
   This allows composing multiple layers sequentially

WHY THESE TYPES?
- Type safety: Compiler catches dimension mismatches
- Clarity: Explicit types make code self-documenting
- Composability: Easy to chain layers together

EXAMPLE STRUCTURES:
  A 2-input, 3-hidden, 1-output network:
    Layer 1: weights [[w11, w12], [w21, w22], [w31, w32]] (3x2)
             biases [b1, b2, b3]
             activation Sigmoid
    Layer 2: weights [[w1, w2, w3]] (1x3)
             biases [b]
             activation Sigmoid
-}
type Matrix = [[Double]]
type Vector = [Double]

data Activation = Sigmoid | Tanh | ReLU | Identity
  deriving (Show, Eq)

-- TODO: Define Layer data type
-- Should contain: weights (Matrix), biases (Vector), activation (Activation)
data Layer = LayerStub
  deriving (Show)

-- TODO: Define Network as a list of layers
type Network = [Layer]

{- |
TODO 2: Implement activation functions

CONCEPT: Activation Functions
Activation functions introduce non-linearity, allowing networks to learn complex patterns.
Each function has forward and derivative forms (needed for backpropagation).

GUIDELINES:
Implement two functions for each activation:
1. applyActivation: Applies the activation to a value
2. activationDerivative: Computes derivative (for backpropagation)

FORMULAS:

Sigmoid:
  forward: σ(x) = 1 / (1 + e^(-x))
  derivative: σ'(x) = σ(x) * (1 - σ(x))
  
Tanh:
  forward: tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
  derivative: tanh'(x) = 1 - tanh(x)^2
  
ReLU (Rectified Linear Unit):
  forward: relu(x) = max(0, x)
  derivative: relu'(x) = 1 if x > 0, else 0
  
Identity:
  forward: id(x) = x
  derivative: id'(x) = 1

IMPLEMENTATION TIPS:
- Use Haskell's exp function for e^x
- For Sigmoid, compute sigma = 1 / (1 + exp (-x))
- For derivative, call forward function first to avoid recomputation
- Handle ReLU carefully: derivative is 0 at x=0

EXAMPLES:
  applyActivation Sigmoid 0.0 ≈ 0.5
  applyActivation ReLU (-1.0) = 0.0
  applyActivation ReLU 2.0 = 2.0
  activationDerivative ReLU 2.0 = 1.0
  activationDerivative ReLU (-1.0) = 0.0
-}
applyActivation :: Activation -> Double -> Double
applyActivation Sigmoid x = undefined
-- TODO: Implement for Tanh, ReLU, Identity

activationDerivative :: Activation -> Double -> Double
activationDerivative Sigmoid x = undefined
-- TODO: Implement derivatives for Tanh, ReLU, Identity

{- |
TODO 3: Implement matrix and vector operations

CONCEPT: Linear Algebra for Neural Networks
Neural networks perform linear transformations (matrix-vector multiplication)
followed by non-linear activations.

GUIDELINES:

1. dotProduct: Multiply corresponding elements and sum
   [a, b, c] · [x, y, z] = ax + by + cz
   Implementation: Use zipWith (*) to multiply, then sum

2. matrixVectorMult: Multiply matrix by vector
   Each row of matrix dot-producted with vector gives one output element
   [[a, b], [c, d]] × [x, y] = [ax+by, cx+dy]
   Implementation: Map dotProduct over each row

3. vectorAdd: Element-wise addition
   [a, b, c] + [x, y, z] = [a+x, b+y, c+z]
   Implementation: Use zipWith (+)

4. vectorScale: Multiply vector by scalar
   k * [a, b, c] = [ka, kb, kc]
   Implementation: Map (* k) over elements

5. outerProduct: Create matrix from two vectors
   [a, b] ⊗ [x, y, z] = [[ax, ay, az], [bx, by, bz]]
   Implementation: For each element in v1, multiply by all of v2

6. matrixAdd: Element-wise matrix addition
   [[a, b], [c, d]] + [[w, x], [y, z]] = [[a+w, b+x], [c+y, d+z]]
   Implementation: Use zipWith (zipWith (+))

IMPORTANT: These functions assume correct dimensions.
In production code, you'd add dimension checking.

TESTING:
  dotProduct [1, 2, 3] [4, 5, 6] = 32
  matrixVectorMult [[1, 2], [3, 4]] [5, 6] = [17, 39]
  vectorAdd [1, 2] [3, 4] = [4, 6]
  outerProduct [1, 2] [3, 4] = [[3, 4], [6, 8]]
-}
dotProduct :: Vector -> Vector -> Double
dotProduct v1 v2 = undefined
-- TODO: Multiply corresponding elements and sum

matrixVectorMult :: Matrix -> Vector -> Vector
matrixVectorMult m v = undefined
-- TODO: Multiply each row of matrix with vector

vectorAdd :: Vector -> Vector -> Vector
vectorAdd v1 v2 = undefined
-- TODO: Add corresponding elements

vectorScale :: Double -> Vector -> Vector
vectorScale k v = undefined
-- TODO: Multiply each element by scalar

outerProduct :: Vector -> Vector -> Matrix
outerProduct v1 v2 = undefined
-- TODO: Create matrix from two vectors

matrixAdd :: Matrix -> Matrix -> Matrix
matrixAdd m1 m2 = undefined
-- TODO: Add matrices element-wise

matrixScale :: Double -> Matrix -> Matrix
matrixScale k m = undefined
-- TODO: Multiply each element by scalar

{- |
TODO 4: Implement forward propagation

CONCEPT: Forward Propagation
Transform input through network layers to produce output.
For each layer:
  1. Multiply input by weights: z = W × x
  2. Add biases: z = z + b
  3. Apply activation function: a = σ(z)
  4. Use output as input for next layer

GUIDELINES:

1. forwardLayer: Process one layer
   Input: Layer, input vector
   Output: (output vector, pre-activation values)
   Steps:
   - Multiply weights by input: z = weights × input
   - Add biases: z = z + biases
   - Apply activation: output = activation(z)
   - Return both output and z (z needed for backprop)

2. forward: Process entire network
   Input: Network, input vector
   Output: (final output, list of (activations, pre-activations) for each layer)
   Steps:
   - Use foldl or recursion to process layers sequentially
   - Thread input through each layer
   - Collect intermediate values for backpropagation
   - Return final output and all intermediate values

WHY SAVE INTERMEDIATE VALUES?
Backpropagation needs:
- Pre-activation values (z) to compute gradients
- Activations (a) from each layer for gradient calculation

EXAMPLE:
  Network: [Layer1, Layer2]
  Input: [1.0, 2.0]
  
  Layer 1: [1.0, 2.0] → z1 → a1 = [0.5, 0.3, 0.8]
  Layer 2: [0.5, 0.3, 0.8] → z2 → a2 = [0.7]
  
  Result: ([0.7], [(a1, z1), (a2, z2)])

IMPLEMENTATION PATTERN:
  Use foldl' for efficiency with accumulator:
    (current_output, accumulated_layer_info) = 
      foldl' processLayer (initial_input, []) layers
-}
forwardLayer :: Layer -> Vector -> (Vector, Vector)
forwardLayer layer input = undefined
-- TODO: Process single layer, return (activation, pre-activation)

forward :: Network -> Vector -> (Vector, [(Vector, Vector)])
forward network input = undefined
-- TODO: Process all layers, collect intermediate values

{- |
TODO 5: Implement backpropagation

CONCEPT: Backpropagation
Compute gradients of loss with respect to weights and biases.
Works backwards through network, computing error signals (deltas).

ALGORITHM:
1. Compute output error: δL = (output - target) ⊙ σ'(zL)
   where ⊙ is element-wise multiplication

2. For each layer l (from L-1 to 1):
   δl = (W(l+1)T × δ(l+1)) ⊙ σ'(zl)
   
3. Compute gradients:
   ∂Loss/∂W(l) = δl × a(l-1)T  (outer product)
   ∂Loss/∂b(l) = δl

GUIDELINES:

1. outputDelta: Compute error for output layer
   Input: predicted output, target output, activation type, pre-activation values
   Output: delta (error signal)
   Formula: (predicted - target) ⊙ activation'(z)
   Implementation: 
   - Subtract vectors: error = predicted - target
   - Apply derivative element-wise
   - Multiply corresponding elements

2. backpropLayer: Propagate error to previous layer
   Input: current delta, next layer weights, current pre-activations, activation
   Output: previous layer delta
   Formula: δl = (W(l+1)T × δ(l+1)) ⊙ σ'(zl)
   Implementation:
   - Transpose next layer weights (swap rows/columns)
   - Multiply by current delta
   - Element-wise multiply by activation derivative

3. computeGradients: Calculate weight and bias gradients
   Input: delta, previous layer activations
   Output: (weight gradients, bias gradients)
   Formula: 
   - dW = δ × aT (outer product)
   - db = δ
   Implementation: Use outerProduct

4. backpropagate: Full backpropagation through network
   Input: network, input, target, forward pass results
   Output: list of (weight gradients, bias gradients) for each layer
   Steps:
   - Compute output delta
   - Propagate backwards through layers
   - Compute gradients for each layer
   - Return list of gradients

EXAMPLE:
  If output is [0.8] and target is [1.0]:
  - Error: [0.8 - 1.0] = [-0.2]
  - If σ'(z) = [0.16], then δ = [-0.2 * 0.16] = [-0.032]

IMPLEMENTATION TIPS:
- Process layers in reverse order: use reverse
- Keep track of activations from forward pass
- Gradient list should match network structure
-}
outputDelta :: Vector -> Vector -> Activation -> Vector -> Vector
outputDelta predicted target activation preActivation = undefined
-- TODO: Compute output layer error

backpropLayer :: Vector -> Matrix -> Vector -> Activation -> Vector
backpropLayer delta nextWeights preActivation activation = undefined
-- TODO: Propagate error to previous layer

computeGradients :: Vector -> Vector -> (Matrix, Vector)
computeGradients delta prevActivation = undefined
-- TODO: Compute weight and bias gradients

backpropagate :: Network -> Vector -> Vector -> (Vector, [(Vector, Vector)]) -> [(Matrix, Vector)]
backpropagate network input target (output, layerOutputs) = undefined
-- TODO: Full backpropagation through network

{- |
TODO 6: Implement training functions

CONCEPT: Training
Update network weights to minimize error using gradient descent.
For each training example:
  1. Forward pass: compute predictions
  2. Backward pass: compute gradients
  3. Update: adjust weights and biases

GUIDELINES:

1. updateLayer: Update one layer's parameters
   Input: layer, (weight gradients, bias gradients), learning rate
   Output: updated layer
   Formula: 
   - newWeights = weights - learningRate * gradientWeights
   - newBiases = biases - learningRate * gradientBiases
   Implementation:
   - Use matrixScale for weight update
   - Use vectorScale for bias update
   - Subtract updates from current parameters

2. updateNetwork: Update entire network
   Input: network, list of gradients, learning rate
   Output: updated network
   Implementation:
   - Zip network layers with gradients
   - Map updateLayer over pairs
   - Preserve layer structure

3. trainSingle: Train on one example
   Input: network, input, target, learning rate
   Output: (updated network, loss)
   Steps:
   - Forward pass: get predictions
   - Compute loss (mean squared error)
   - Backpropagate: get gradients
   - Update parameters
   - Return new network and loss

4. trainEpoch: Train on dataset for one epoch
   Input: network, training data, learning rate
   Output: (updated network, average loss)
   Implementation:
   - Fold over training examples
   - Call trainSingle for each
   - Accumulate losses
   - Return final network and average loss

LOSS FUNCTION (Mean Squared Error):
  MSE = (1/n) * Σ(predicted - target)²
  For single prediction: (1/2) * sum((p - t)²) for each output

LEARNING RATE:
  Typical values: 0.01 to 0.1
  Too high: unstable training
  Too low: slow convergence

EXAMPLE TRAINING LOOP:
  Repeat for N epochs:
    1. Train on all examples (one epoch)
    2. Print average loss
    3. If loss is low enough, stop

IMPLEMENTATION TIPS:
- Use strict foldl' for efficiency
- Track loss to monitor training
- Consider momentum or adaptive learning rates (advanced)
-}
meanSquaredError :: Vector -> Vector -> Double
meanSquaredError predicted target = undefined
-- TODO: Compute MSE loss

updateLayer :: Layer -> (Matrix, Vector) -> Double -> Layer
updateLayer layer (gradWeights, gradBiases) learningRate = undefined
-- TODO: Update layer parameters using gradients

updateNetwork :: Network -> [(Matrix, Vector)] -> Double -> Network
updateNetwork network gradients learningRate = undefined
-- TODO: Update all layers in network

trainSingle :: Network -> Vector -> Vector -> Double -> (Network, Double)
trainSingle network input target learningRate = undefined
-- TODO: Train on single example, return (updated network, loss)

trainEpoch :: Network -> [(Vector, Vector)] -> Double -> (Network, Double)
trainEpoch network trainingData learningRate = undefined
-- TODO: Train on all examples, return (network, average loss)

{- |
TODO 7: Implement network initialization and prediction

CONCEPT: Initialization and Usage
- Initialize: Create network with random weights
- Predict: Use trained network for inference

GUIDELINES:

1. initializeLayer: Create layer with dimensions
   Input: input size, output size, activation type
   Output: Layer with initialized weights and biases
   Initialization strategy:
   - Weights: small random values (e.g., -0.5 to 0.5)
   - Biases: zeros or small values
   - For this template: use simple values (0.1, -0.1, etc.)
   
   Better initialization (advanced):
   - Xavier: weights ~ Uniform(-√(6/(nin+nout)), √(6/(nin+nout)))
   - He: weights ~ Normal(0, √(2/nin))

2. createNetwork: Build network from layer specifications
   Input: list of (input size, output size, activation) tuples
   Output: initialized network
   Implementation:
   - Map initializeLayer over specifications
   - Return list of layers

3. predict: Make prediction with trained network
   Input: network, input vector
   Output: output vector
   Implementation:
   - Call forward function
   - Extract final output (ignore intermediate values)
   - Return output

EXAMPLE:
  Create network: [2, 3, 1] (2 inputs, 3 hidden, 1 output)
    Layer 1: 2 → 3 with Sigmoid
    Layer 2: 3 → 1 with Sigmoid
  
  Predict: predict network [1.0, 2.0] → [0.73]

USAGE PATTERN:
  1. Define network structure
  2. Initialize network
  3. Train on data
  4. Make predictions

NOTE: Without random number generation, we use fixed initialization.
For real applications, use System.Random or random package.
-}
initializeLayer :: Int -> Int -> Activation -> Layer
initializeLayer inputSize outputSize activation = undefined
-- TODO: Create layer with given dimensions
-- Use simple initialization (small fixed values or pattern)

createNetwork :: [(Int, Int, Activation)] -> Network
createNetwork specs = undefined
-- TODO: Create network from specifications

predict :: Network -> Vector -> Vector
predict network input = undefined
-- TODO: Make prediction (forward pass only)

{- |
IMPLEMENTATION GUIDE
--------------------

Step-by-step approach:

1. START WITH DATA TYPES (TODO 1)
   - Define Layer with weights, biases, activation
   - Test: Create a layer manually, verify structure

2. IMPLEMENT ACTIVATIONS (TODO 2)
   - Start with Sigmoid (most common)
   - Test each function with known values
   - Verify derivatives match mathematical definitions

3. IMPLEMENT LINEAR ALGEBRA (TODO 3)
   - Start with dotProduct (simplest)
   - Build up to matrix operations
   - Test each function independently
   - Verify dimensions match expectations

4. IMPLEMENT FORWARD PASS (TODO 4)
   - Start with single layer forward
   - Test on simple examples (known weights)
   - Extend to full network forward
   - Verify output shapes are correct

5. IMPLEMENT BACKPROPAGATION (TODO 5)
   - Most complex part - take your time
   - Start with output delta (simplest)
   - Implement gradient computation
   - Work backwards through layers
   - Test with simple network (2→2→1)

6. IMPLEMENT TRAINING (TODO 6)
   - Start with single example training
   - Verify loss decreases
   - Extend to epoch training
   - Test on simple dataset (XOR, AND, OR)

7. IMPLEMENT UTILITIES (TODO 7)
   - Create initialization functions
   - Build complete network
   - Test full training loop
   - Verify network learns simple patterns

TESTING STRATEGY:

1. Unit Tests:
   - Test each function individually
   - Use known inputs with expected outputs
   - Verify edge cases

2. Integration Tests:
   - Test forward pass with fixed weights
   - Verify backprop gradients (numerical gradient check)
   - Test training reduces loss

3. Example Problems:
   - AND gate: [0,0]→0, [0,1]→0, [1,0]→0, [1,1]→1
   - OR gate: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→1
   - XOR gate: [0,0]→0, [0,1]→1, [1,0]→1, [1,1]→0
     (XOR requires hidden layer - good test!)

DEBUGGING TIPS:

1. Print intermediate values
2. Check dimensions at each step
3. Verify activations are in correct range
4. Ensure gradients aren't exploding or vanishing
5. Start with small network (2→2→1)
6. Use low learning rate initially

COMMON PITFALLS:

1. Dimension mismatches: Always verify matrix/vector sizes
2. Forgetting to apply activation derivative in backprop
3. Wrong order in matrix multiplication
4. Not saving intermediate values in forward pass
5. Learning rate too high (causing divergence)

EXTENSIONS (after basic implementation):

1. Add more activation functions (Leaky ReLU, ELU, Softmax)
2. Implement batch training
3. Add momentum or Adam optimizer
4. Implement dropout for regularization
5. Add softmax + cross-entropy for classification
6. Implement different weight initialization strategies
7. Add learning rate scheduling
8. Implement mini-batch gradient descent
9. Add validation set evaluation
10. Save/load trained networks
-}

-- Example usage and tests
main :: IO ()
main = do
  putStrLn "Neural Network in Haskell - Template\n"
  putStrLn "This is a template with detailed implementation guidelines."
  putStrLn "Follow the TODOs above to implement a working neural network.\n"
  
  putStrLn "Example network structure:"
  putStrLn "  Input layer:  2 neurons"
  putStrLn "  Hidden layer: 3 neurons (Sigmoid)"
  putStrLn "  Output layer: 1 neuron (Sigmoid)\n"
  
  putStrLn "Example training data (XOR problem):"
  putStrLn "  [0, 0] → [0]"
  putStrLn "  [0, 1] → [1]"
  putStrLn "  [1, 0] → [1]"
  putStrLn "  [1, 1] → [0]\n"
  
  putStrLn "Once implemented, you can:"
  putStrLn "  1. Create network: createNetwork [(2,3,Sigmoid), (3,1,Sigmoid)]"
  putStrLn "  2. Train: trainEpoch network trainingData 0.1"
  putStrLn "  3. Predict: predict network [1, 0]\n"
  
  putStrLn "Complete the TODOs above to build a working neural network!"
  putStrLn "Start with TODO 1 and work through sequentially."
