# Deepfake Detection - From Scratch Implementation

A from-scratch implementation of Deepfake detection techniques in Python, following key research papers in digital forensics and media authentication.

## Goal

Build deepfake detection systems to understand:
- **Visual Artifacts Detection**: Identifying visual inconsistencies in fake media
- **Temporal Analysis**: Detecting temporal inconsistencies across video frames
- **Physiological Signals**: Analyzing biological signals (blink rate, pulse)
- **Frequency Analysis**: Examining frequency domain artifacts
- **Multi-Modal Detection**: Combining visual, audio, and temporal cues
- **Generalization**: Building detectors that work on unseen manipulation types

## What is Deepfake Detection?

Deepfake detection involves identifying synthetically generated or manipulated media using machine learning and forensic techniques. As generation quality improves, detection must evolve to identify subtle artifacts and inconsistencies.

**Detection Approaches**:
1. **Visual Analysis**: CNN-based artifact detection
2. **Temporal Analysis**: Analyzing motion and temporal patterns
3. **Biological Signals**: Detecting missing or abnormal physiological cues
4. **Frequency Domain**: Examining spectral artifacts
5. **Multi-Modal**: Combining multiple detection signals

## Project Structure

```
deepfake-detection/
├── README.md                              # This file
├── IMPLEMENTATION_GUIDE.md                # Detailed implementation steps
├── requirements.txt                       # Python dependencies
├── phase1_cnn_detection/                  # Basic CNN-based detection
│   ├── template_xceptionnet.py            # XceptionNet architecture
│   ├── template_efficientnet.py           # EfficientNet-based detector
│   ├── template_feature_extraction.py     # Feature extraction
│   └── template_classifier.py             # Binary classification
├── phase2_temporal_analysis/              # Temporal inconsistency detection
│   ├── template_temporal_features.py      # Extract temporal features
│   ├── template_lstm_detector.py          # LSTM-based detection
│   ├── template_optical_flow.py           # Optical flow analysis
│   └── template_3dcnn.py                  # 3D CNN for video
├── phase3_frequency_analysis/             # Frequency domain detection
│   ├── template_fft_analysis.py           # FFT-based artifact detection
│   ├── template_dct_analysis.py           # DCT coefficient analysis
│   └── template_spectral_features.py      # Spectral feature extraction
├── phase4_biological_signals/             # Physiological signal analysis
│   ├── template_blink_detection.py        # Eye blink analysis
│   ├── template_pulse_detection.py        # PPG signal extraction
│   └── template_face_consistency.py       # Facial feature consistency
├── phase5_advanced_methods/               # State-of-the-art methods
│   ├── template_capsule_network.py        # Capsule networks for detection
│   ├── template_attention_mechanism.py    # Attention-based detection
│   ├── template_multi_modal.py            # Multi-modal fusion
│   └── template_adversarial_training.py   # Adversarial robustness
└── solutions/                              # Complete reference implementations
    ├── README.md
    └── [same structure as above]
```

## Key Papers and Techniques

### Phase 1: CNN-Based Detection
Visual artifact detection using deep learning

**Core Papers**:
- [FaceForensics++ (2019)](https://arxiv.org/abs/1901.08971): Comprehensive benchmark
- [XceptionNet for Deepfakes (2018)](https://arxiv.org/abs/1802.06608): Effective CNN architecture
- [EfficientNet-B4 (2019)](https://arxiv.org/abs/1905.11946): High-performance detection

**Technique**:
1. Extract face regions from frames
2. Feed through CNN (XceptionNet, EfficientNet)
3. Extract deep features
4. Binary classification (real/fake)
5. Post-processing and temporal aggregation

**Key Features**:
- Transfer learning from ImageNet
- Data augmentation for robustness
- Class balancing
- Cross-dataset generalization

### Phase 2: Temporal Analysis
Detecting inconsistencies across time

**Core Papers**:
- [Recurrent Neural Networks for Deepfake Detection (2019)](https://arxiv.org/abs/1910.06926)
- [Temporal Inconsistencies (2020)](https://arxiv.org/abs/2004.07676)

**Technique**:
1. Extract frame sequences
2. Compute temporal features (optical flow, frame differences)
3. Use LSTM/GRU to model temporal patterns
4. Detect temporal inconsistencies
5. 3D CNNs for spatiotemporal analysis

**Detected Artifacts**:
- Temporal flickering
- Inconsistent head pose
- Unnatural eye movements
- Lighting inconsistencies

### Phase 3: Frequency Domain Analysis
Examining spectral artifacts invisible to naked eye

**Core Papers**:
- [Frequency Analysis of Deepfakes (2020)](https://arxiv.org/abs/2004.08955)
- [DCT-based Detection (2019)](https://arxiv.org/abs/1906.05856)

**Technique**:
1. Apply FFT or DCT to image patches
2. Analyze frequency spectrum
3. Detect compression artifacts
4. Identify GAN fingerprints
5. Statistical analysis of coefficients

**Key Insights**:
- GANs leave frequency domain fingerprints
- Upsampling creates high-frequency artifacts
- Compression patterns differ from natural images

### Phase 4: Biological Signal Analysis
Detecting missing or abnormal physiological cues

**Core Papers**:
- [Eye Blinking in Deepfakes (2018)](https://arxiv.org/abs/1806.02877)
- [Physiological Signal Detection (2019)](https://arxiv.org/abs/1911.07427)

**Technique**:
1. Detect eye regions and analyze blink patterns
2. Extract pulse signals using photoplethysmography (PPG)
3. Analyze facial blood flow patterns
4. Detect inconsistent head pose sequences
5. Statistical analysis of biological signals

**Detected Anomalies**:
- Abnormal blink rates
- Missing pulse signals
- Inconsistent breathing patterns
- Unnatural facial dynamics

### Phase 5: Advanced Methods
State-of-the-art detection techniques

**Core Papers**:
- [Capsule Networks for Deepfakes (2019)](https://arxiv.org/abs/1910.12467)
- [Multi-task Learning (2020)](https://arxiv.org/abs/2004.11333)
- [Attention Mechanisms (2021)](https://arxiv.org/abs/2103.02406)

**Techniques**:
- **Capsule Networks**: Better generalization via capsule routing
- **Attention Mechanisms**: Focus on discriminative regions
- **Multi-Modal Fusion**: Combine visual, audio, temporal cues
- **Adversarial Training**: Robustness against adaptive attacks
- **Meta-Learning**: Quick adaptation to new manipulation types

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+
- scikit-learn, scipy
- GPU recommended

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install torch torchvision opencv-python scikit-learn scipy numpy matplotlib tqdm
pip install facenet-pytorch  # For face detection
```

### Datasets

**Benchmark Datasets**:
- **FaceForensics++**: 1.8M frames (DeepFakes, Face2Face, FaceSwap, NeuralTextures)
- **Celeb-DF**: High-quality celebrity deepfakes
- **DFDC**: Deepfake Detection Challenge dataset (100K+ videos)
- **DeeperForensics**: 60K videos with various manipulations
- **WildDeepfake**: In-the-wild deepfake videos

### Basic Usage

#### CNN-Based Detection
```bash
# Train XceptionNet detector
python phase1_cnn_detection/train.py \
    --dataset faceforensics \
    --model xceptionnet \
    --epochs 30 \
    --batch-size 32

# Evaluate on test set
python phase1_cnn_detection/evaluate.py \
    --checkpoint xception_best.pt \
    --test-data test_videos/
```

#### Video Analysis
```bash
# Analyze video for deepfakes
python detect_video.py \
    --input suspicious_video.mp4 \
    --model xception_best.pt \
    --output analysis_report.json

# Visualize detection results
python visualize_detection.py \
    --video suspicious_video.mp4 \
    --predictions predictions.json \
    --output annotated_video.mp4
```

#### Multi-Modal Detection
```bash
# Combine multiple detection methods
python phase5_advanced_methods/multi_modal_detect.py \
    --visual-model xception.pt \
    --temporal-model lstm.pt \
    --frequency-model fft.pt \
    --input video.mp4
```

## Learning Path

### Phase 1: CNN-Based Detection (3-4 hours)
**Goal**: Implement visual artifact detection

1. Understand deepfake artifacts
2. Implement face extraction pipeline
3. Build XceptionNet classifier
4. Train on FaceForensics++ dataset
5. Evaluate cross-dataset performance

**Skills learned**:
- Face detection and extraction
- CNN architectures for detection
- Transfer learning
- Data preprocessing
- Evaluation metrics (AUC, accuracy)

### Phase 2: Temporal Analysis (3-4 hours)
**Goal**: Detect temporal inconsistencies

1. Extract frame sequences
2. Compute optical flow
3. Build LSTM temporal detector
4. Implement 3D CNN
5. Aggregate temporal predictions

**Skills learned**:
- Temporal feature extraction
- Sequence modeling (LSTM, GRU)
- 3D convolutions
- Optical flow analysis
- Video processing

### Phase 3: Frequency Analysis (2-3 hours)
**Goal**: Analyze frequency domain artifacts

1. Apply FFT to face patches
2. Extract DCT coefficients
3. Build frequency-based classifier
4. Analyze spectral patterns
5. Detect GAN fingerprints

**Skills learned**:
- Fourier transforms
- DCT analysis
- Frequency domain processing
- Spectral feature engineering
- Compression artifact analysis

### Phase 4: Biological Signals (2-3 hours)
**Goal**: Detect physiological anomalies

1. Implement eye blink detector
2. Extract PPG signals
3. Analyze blink patterns
4. Detect pulse inconsistencies
5. Build biological signal classifier

**Skills learned**:
- Eye tracking
- Signal processing
- Physiological signal extraction
- Statistical pattern analysis
- Biological plausibility checks

### Phase 5: Advanced Methods (4-5 hours)
**Goal**: Implement state-of-the-art techniques

1. Build capsule network detector
2. Implement attention mechanisms
3. Create multi-modal fusion
4. Add adversarial training
5. Benchmark all methods

**Skills learned**:
- Capsule networks
- Attention mechanisms
- Multi-modal learning
- Adversarial robustness
- Ensemble methods

**Total Time**: ~14-19 hours for complete implementation

## Implementation Details

### Face Extraction

```python
import cv2
from facenet_pytorch import MTCNN

# Detect and extract faces
mtcnn = MTCNN(image_size=224, margin=40)
faces = mtcnn(frame)

# Align and crop
aligned_face = align_face(face, landmarks)
```

### XceptionNet Detector

```python
import torch.nn as nn
from torchvision.models import xception

class DeepfakeDetector(nn.Module):
    def __init__(self):
        super().__init__()
        self.xception = xception(pretrained=True)
        self.xception.fc = nn.Linear(2048, 1)  # Binary classification
        
    def forward(self, x):
        return torch.sigmoid(self.xception(x))

# Training
model = DeepfakeDetector()
loss_fn = nn.BCELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
```

### Temporal LSTM Detector

```python
class TemporalDetector(nn.Module):
    def __init__(self, feature_dim=512, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, num_layers=2, batch_first=True)
        self.classifier = nn.Linear(hidden_dim, 1)
        
    def forward(self, features):
        # features: (batch, seq_len, feature_dim)
        lstm_out, _ = self.lstm(features)
        # Use last timestep
        prediction = self.classifier(lstm_out[:, -1, :])
        return torch.sigmoid(prediction)
```

### Frequency Analysis

```python
import numpy as np
from scipy.fft import fft2, fftshift

def extract_frequency_features(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    # Apply 2D FFT
    f_transform = fft2(gray)
    f_shift = fftshift(f_transform)
    magnitude = np.abs(f_shift)
    
    # Extract features from frequency bins
    features = compute_spectral_features(magnitude)
    return features

def compute_spectral_features(magnitude):
    # High frequency energy
    h, w = magnitude.shape
    high_freq_mask = create_high_freq_mask(h, w)
    high_freq_energy = np.sum(magnitude * high_freq_mask)
    
    # Radial frequency profile
    radial_profile = compute_radial_profile(magnitude)
    
    return np.concatenate([
        [high_freq_energy],
        radial_profile
    ])
```

### Eye Blink Detection

```python
from scipy.spatial import distance

def eye_aspect_ratio(eye_landmarks):
    # Compute eye aspect ratio (EAR)
    A = distance.euclidean(eye_landmarks[1], eye_landmarks[5])
    B = distance.euclidean(eye_landmarks[2], eye_landmarks[4])
    C = distance.euclidean(eye_landmarks[0], eye_landmarks[3])
    
    ear = (A + B) / (2.0 * C)
    return ear

def detect_blinks(video_frames):
    blink_count = 0
    ear_threshold = 0.25
    consecutive_frames = 2
    
    for frame in video_frames:
        landmarks = detect_facial_landmarks(frame)
        left_ear = eye_aspect_ratio(landmarks['left_eye'])
        right_ear = eye_aspect_ratio(landmarks['right_eye'])
        
        ear = (left_ear + right_ear) / 2.0
        
        if ear < ear_threshold:
            # Potential blink
            blink_count += 1
            
    return blink_count
```

## Evaluation Metrics

### Binary Classification Metrics

```python
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

# Compute metrics
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)
auc = roc_auc_score(y_true, y_scores)

# Video-level accuracy
video_predictions = aggregate_frame_predictions(frame_predictions)
video_accuracy = accuracy_score(video_labels, video_predictions)
```

### Cross-Dataset Evaluation

```python
# Train on FaceForensics++, test on Celeb-DF
model.train_on_dataset('faceforensics')
celeb_df_accuracy = model.evaluate('celeb-df')

# Measure generalization gap
generalization_gap = train_accuracy - test_accuracy
```

### Detection at Various Thresholds

```python
from sklearn.metrics import roc_curve

fpr, tpr, thresholds = roc_curve(y_true, y_scores)

# Find optimal threshold
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
```

## Challenges in Deepfake Detection

### 1. Cross-Dataset Generalization
- Models overfit to specific manipulation methods
- Poor performance on unseen generation techniques
- Solution: Diverse training data, adversarial training, meta-learning

### 2. Adaptive Attacks
- Deepfake creators can specifically target detection methods
- Adversarial perturbations to fool detectors
- Solution: Adversarial training, ensemble methods, robust features

### 3. Compressed Media
- Social media compression destroys subtle artifacts
- Detection degrades on compressed videos
- Solution: Train on compressed data, robust features

### 4. High-Quality Fakes
- Modern deepfakes are increasingly realistic
- Fewer visual artifacts to detect
- Solution: Multi-modal detection, biological signals, metadata analysis

## Advanced Topics

### Explainable Detection
```python
# Generate attention maps
from pytorch_grad_cam import GradCAM

grad_cam = GradCAM(model, target_layer)
cam = grad_cam(input_image, target_class=1)

# Visualize which regions influence detection
visualize_cam(input_image, cam)
```

### Adversarial Robustness
```python
# Test against adversarial attacks
from foolbox import PyTorchModel, attacks

fmodel = PyTorchModel(model, bounds=(0, 1))
attack = attacks.FGSM()
adversarial = attack(fmodel, images, labels)

# Measure robustness
adversarial_accuracy = evaluate(model, adversarial)
```

### Active Learning
```python
# Select most informative samples
from modAL.models import ActiveLearner

learner = ActiveLearner(
    estimator=model,
    query_strategy=uncertainty_sampling
)

# Query uncertain samples
query_idx = learner.query(unlabeled_pool)
```

## Best Practices

### Data Preprocessing
- Consistent face alignment
- Normalize image statistics
- Handle various video qualities
- Balance real/fake samples

### Model Training
- Use transfer learning
- Apply data augmentation
- Implement early stopping
- Save best checkpoints
- Monitor overfitting

### Evaluation
- Test on multiple datasets
- Measure cross-dataset performance
- Evaluate on compressed media
- Test temporal consistency
- Analyze failure cases

### Deployment
- Optimize inference speed
- Implement batch processing
- Add confidence thresholds
- Provide interpretability
- Monitor model drift

## Troubleshooting

**"Low cross-dataset accuracy"**:
- Increase training data diversity
- Use adversarial training
- Add domain adaptation
- Implement meta-learning

**"Slow inference"**:
- Use model quantization
- Implement TensorRT optimization
- Batch process frames
- Reduce image resolution

**"High false positive rate"**:
- Adjust decision threshold
- Improve face quality filtering
- Add temporal smoothing
- Use ensemble methods

**"Poor compression robustness"**:
- Train on compressed data
- Use robust features
- Add compression augmentation
- Focus on spatial patterns

## Ethics and Responsible Development

### Responsible Detection

1. **Transparency**: Clearly communicate detection confidence
2. **Fairness**: Test across diverse demographics
3. **Privacy**: Protect subjects' privacy
4. **Accuracy**: Minimize false positives and negatives
5. **Context**: Consider detection context and impact

### Societal Impact

- Help combat misinformation
- Protect individuals from malicious deepfakes
- Support media authentication
- Enable platform moderation
- Preserve trust in digital media

## Documentation

### Key Papers

**Benchmarks**:
- [FaceForensics++ (2019)](https://arxiv.org/abs/1901.08971)
- [Celeb-DF (2020)](https://arxiv.org/abs/1909.12962)
- [DFDC (2020)](https://arxiv.org/abs/2006.07397)

**Detection Methods**:
- [XceptionNet (2018)](https://arxiv.org/abs/1802.06608)
- [Capsule Networks (2019)](https://arxiv.org/abs/1910.12467)
- [Temporal Aware Detection (2020)](https://arxiv.org/abs/2004.07676)
- [Frequency Analysis (2020)](https://arxiv.org/abs/2004.08955)

**Surveys**:
- [Deepfakes and Beyond: Survey (2020)](https://arxiv.org/abs/2001.00179)
- [Media Forensics Survey (2020)](https://arxiv.org/abs/2004.09271)

### Resources

- [FaceForensics++ GitHub](https://github.com/ondyari/FaceForensics)
- [Deepfake Detection Challenge](https://www.kaggle.com/c/deepfake-detection-challenge)
- [Sensity AI Resources](https://sensity.ai/resources/)
- [Detection Papers Collection](https://github.com/aerophile/awesome-deepfakes)

### Video Courses

- [Computer Vision Courses](https://github.com/Developer-Y/cs-video-courses#computer-vision)
- [Deep Learning Courses](https://github.com/Developer-Y/cs-video-courses#deep-learning)
- [Machine Learning Courses](https://github.com/Developer-Y/cs-video-courses#machine-learning)

## Related Projects

- **deepfake-creation/**: Understanding generation techniques
- **diffusion-models/**: Modern generative models
- **ml-in-production/**: Deploying detection systems

## License

Educational purposes only. Use responsibly and ethically.

## Acknowledgments

Inspired by:
- FaceForensics++ by Rössler et al.
- DFDC competition and participants
- Academic research in media forensics
- Industry efforts in deepfake detection
- Open-source detection tools and datasets
