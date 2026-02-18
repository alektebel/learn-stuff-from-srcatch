# Deepfake Creation - From Scratch Implementation

A from-scratch implementation of Deepfake generation techniques in Python, following key research papers.

## Goal

Build deepfake generation systems to understand:
- **Face Swapping**: Replace faces in videos with other faces
- **Face Reenactment**: Transfer facial expressions and movements
- **Generative Models**: GANs and autoencoders for face synthesis
- **Video Generation**: Temporal consistency and frame interpolation
- **Lip Synchronization**: Audio-driven facial animation
- **Advanced Techniques**: First Order Motion Model, neural rendering

## What are Deepfakes?

Deepfakes are synthetic media created using deep learning techniques that can replace or manipulate faces in images and videos. The technology has evolved through several key innovations:

1. **Face Swapping**: Replace identity while preserving expression
2. **Face Reenactment**: Transfer expressions from source to target
3. **Audio-Driven**: Generate facial movements from audio
4. **Full Body**: Extend to body and pose manipulation

## Project Structure

```
deepfake-creation/
├── README.md                          # This file
├── IMPLEMENTATION_GUIDE.md            # Detailed implementation steps
├── requirements.txt                   # Python dependencies
├── phase1_face_swap/                  # Classic face swapping
│   ├── template_face_detector.py      # Face detection (MTCNN, dlib)
│   ├── template_face_encoder.py       # Face encoding/embedding
│   ├── template_autoencoder.py        # Encoder-decoder architecture
│   └── template_face_swap.py          # Face swapping pipeline
├── phase2_face_reenactment/           # Expression transfer
│   ├── template_landmark_detector.py  # Facial landmark detection
│   ├── template_expression_transfer.py # Transfer expressions
│   └── template_video_generator.py    # Generate video frames
├── phase3_first_order_motion/         # First Order Motion Model
│   ├── template_keypoint_detector.py  # Sparse keypoint detection
│   ├── template_motion_transfer.py    # Motion field transfer
│   └── template_generator.py          # Image animation
├── phase4_audio_driven/               # Audio to face animation
│   ├── template_audio_encoder.py      # Audio feature extraction
│   ├── template_lip_sync.py           # Lip synchronization
│   └── template_wav2lip.py            # Wav2Lip implementation
└── solutions/                          # Complete reference implementations
    ├── README.md
    └── [same structure as above]
```

## Key Papers and Techniques

### Phase 1: Classic Face Swapping
Based on **DeepFakes** and **FaceSwap** approaches

**Core Papers**:
- [DeepFakes (2017)](https://arxiv.org/abs/1909.11573): Original autoencoder approach
- [FaceSwap-GAN](https://github.com/shaoanlu/faceswap-GAN): GAN-based face swapping

**Technique**:
1. Detect and align faces in source and target videos
2. Train shared encoder with separate decoders for each identity
3. Encode target face, decode with source decoder
4. Blend swapped face back into original frame

**Architecture**:
```
Source Face → Encoder → Decoder_A → Reconstructed Source
Target Face → Encoder → Decoder_B → Reconstructed Target
Target Face → Encoder → Decoder_A → Swapped Face
```

### Phase 2: Face Reenactment
Based on **Face2Face** and expression transfer techniques

**Core Papers**:
- [Face2Face (2016)](https://arxiv.org/abs/1904.03251): Real-time facial reenactment
- [X2Face (2018)](https://arxiv.org/abs/1809.03815): Self-supervised face reenactment

**Technique**:
1. Extract facial landmarks and expressions from source
2. Transfer expressions to target face model
3. Render target face with source expressions
4. Ensure temporal consistency across frames

### Phase 3: First Order Motion Model
Based on **FOMM** - state-of-the-art image animation

**Core Paper**:
- [First Order Motion Model (2019)](https://arxiv.org/abs/2003.00196): Image animation using sparse keypoints

**Technique**:
1. Extract sparse keypoints and local affine transformations
2. Learn motion representation from single image
3. Transfer motion to animate static images
4. Works on various objects (faces, bodies, objects)

**Key Innovation**: 
- No 3D model required
- Works with single driving video
- Generalizes to unseen objects
- Self-supervised learning

### Phase 4: Audio-Driven Animation
Based on **Wav2Lip** and audio-to-visual synthesis

**Core Papers**:
- [Wav2Lip (2020)](https://arxiv.org/abs/2008.10010): Accurate lip-sync for any identity
- [MakeItTalk (2020)](https://arxiv.org/abs/2004.12992): Audio-driven facial animation

**Technique**:
1. Extract audio features (MFCC, mel-spectrogram)
2. Generate lip movements synchronized with audio
3. Preserve facial identity and expressions
4. Ensure high visual quality

## Quick Start

### Prerequisites

- Python 3.8+
- PyTorch 1.12+
- OpenCV 4.5+
- dlib or MTCNN for face detection
- GPU recommended (CUDA-enabled)

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Or manually:
pip install torch torchvision opencv-python dlib face_recognition numpy matplotlib tqdm
```

### Datasets

Common datasets for training:
- **CelebA**: Celebrity faces dataset (200K images)
- **VGGFace2**: Large-scale face dataset (3.3M images)
- **FaceForensics++**: Video dataset for deepfakes
- **VoxCeleb**: Audio-visual dataset

### Basic Usage

#### Face Swapping
```bash
# Train face swap model
python phase1_face_swap/train.py --source data/personA --target data/personB --epochs 100

# Generate swapped video
python phase1_face_swap/swap_video.py --model checkpoint.pt --input video.mp4 --output swapped.mp4
```

#### First Order Motion
```bash
# Animate image with driving video
python phase3_first_order_motion/animate.py \
    --source image.jpg \
    --driving driving_video.mp4 \
    --checkpoint fomm_checkpoint.pt \
    --output animated.mp4
```

#### Wav2Lip
```bash
# Sync lips with audio
python phase4_audio_driven/wav2lip.py \
    --face video.mp4 \
    --audio speech.wav \
    --checkpoint wav2lip.pt \
    --output synced.mp4
```

## Learning Path

### Phase 1: Face Swapping Basics (4-6 hours)
**Goal**: Implement classic face swapping with autoencoders

1. Understand face detection and alignment
2. Implement face encoder-decoder architecture
3. Train separate decoders for different identities
4. Implement face blending and post-processing
5. Generate swapped images and videos

**Skills learned**:
- Face detection and alignment
- Autoencoder architecture
- Identity preservation
- Image blending techniques
- Video processing

### Phase 2: Face Reenactment (3-4 hours)
**Goal**: Transfer expressions between faces

1. Extract facial landmarks
2. Compute expression parameters
3. Transfer expressions to target face
4. Render animated face
5. Ensure temporal smoothness

**Skills learned**:
- Facial landmark detection
- Expression parameterization
- Facial animation
- Temporal consistency
- 3D face models (optional)

### Phase 3: First Order Motion Model (4-5 hours)
**Goal**: Implement state-of-the-art image animation

1. Understand keypoint detection network
2. Implement dense motion prediction
3. Build generator network
4. Train with self-supervised losses
5. Animate images with driving videos

**Skills learned**:
- Keypoint detection
- Motion representation
- Dense motion fields
- Self-supervised learning
- Advanced GAN architectures

### Phase 4: Audio-Driven Animation (3-4 hours)
**Goal**: Generate lip-synced talking faces

1. Extract audio features
2. Implement audio-to-visual mapping
3. Build lip-sync discriminator
4. Train Wav2Lip model
5. Generate synchronized videos

**Skills learned**:
- Audio processing
- Audio-visual synchronization
- Discriminative training
- Lip-sync metrics
- Multi-modal learning

**Total Time**: ~14-19 hours for complete implementation

## Implementation Details

### Face Detection and Alignment

```python
# Using MTCNN or dlib
import cv2
import dlib

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Detect face
faces = detector(image, 1)
landmarks = predictor(image, faces[0])

# Align face
aligned_face = align_face(image, landmarks)
```

### Autoencoder Architecture

```python
class FaceEncoder(nn.Module):
    def __init__(self):
        # Conv layers to encode face to latent vector
        
class FaceDecoder(nn.Module):
    def __init__(self):
        # Deconv layers to decode latent to face
        
# Training
encoded = encoder(face)
reconstructed = decoder(encoded)
loss = mse_loss(reconstructed, face)
```

### First Order Motion Model

```python
# Keypoint detector
keypoints_source = keypoint_detector(source_image)
keypoints_driving = keypoint_detector(driving_frame)

# Motion estimation
motion_params = estimate_motion(keypoints_source, keypoints_driving)

# Generate frame
generated_frame = generator(source_image, motion_params)
```

### Wav2Lip Pipeline

```python
# Extract audio features
audio_features = audio_encoder(audio_clip)

# Generate lip movements
lip_frames = lip_generator(face_frames, audio_features)

# Discriminator for sync
sync_score = sync_discriminator(lip_frames, audio_features)
```

## Ethics and Responsible Use

⚠️ **Important Ethical Considerations**:

1. **Consent**: Never create deepfakes of people without their explicit consent
2. **Disclosure**: Always disclose when media is synthetically generated
3. **Legal**: Be aware of laws regarding deepfakes in your jurisdiction
4. **Harm Prevention**: Don't create deepfakes for harassment, misinformation, or fraud
5. **Education**: Use this knowledge to understand and detect deepfakes

**This implementation is for educational purposes only.**

### Responsible AI Practices

- Implement watermarking in generated content
- Add metadata indicating synthetic origin
- Study detection methods alongside creation
- Understand societal implications
- Follow ethical AI guidelines

## Advanced Topics

### Quality Improvements
- Higher resolution generation (512x512, 1024x1024)
- Better temporal consistency
- Improved blending techniques
- Artifact reduction

### Advanced Architectures
- StyleGAN-based face swapping
- Neural radiance fields (NeRF)
- 3D-aware generation
- Transformer-based models

### Additional Features
- Multi-face swapping
- Facial attribute editing
- Age progression/regression
- Expression amplification

## Testing and Evaluation

### Visual Quality Metrics
```python
# Structural Similarity (SSIM)
from skimage.metrics import structural_similarity
ssim_score = structural_similarity(original, generated)

# Peak Signal-to-Noise Ratio (PSNR)
psnr_score = peak_signal_noise_ratio(original, generated)

# Fréchet Inception Distance (FID)
fid_score = calculate_fid(real_images, generated_images)
```

### Identity Preservation
```python
# Face recognition similarity
from face_recognition import face_encodings, face_distance

encoding_original = face_encodings(original_face)[0]
encoding_swapped = face_encodings(swapped_face)[0]
distance = face_distance([encoding_original], encoding_swapped)
```

### Temporal Consistency
```python
# Optical flow consistency
flow = cv2.calcOpticalFlowFarneback(frame1, frame2, ...)
consistency_score = evaluate_flow_consistency(flow)
```

## Troubleshooting

### Common Issues

**"Face not detected"**:
- Improve lighting conditions
- Use higher resolution images
- Try different face detectors (MTCNN, dlib, RetinaFace)
- Check face angle and occlusion

**"Poor blending quality"**:
- Implement Poisson blending
- Use alpha masks
- Add color correction
- Smooth boundary transitions

**"Temporal flickering"**:
- Use optical flow for stabilization
- Implement temporal smoothing
- Use frame interpolation
- Apply post-processing filters

**"Identity leakage"**:
- Train longer
- Use stronger encoder
- Implement adversarial loss
- Increase model capacity

## Performance Tips

### Training
- Use mixed precision training
- Implement gradient accumulation
- Use data augmentation
- Enable distributed training

### Inference
- Batch process frames
- Use TensorRT optimization
- Implement frame caching
- Optimize face detection

## Documentation

### Key Papers

**Foundation**:
- [DeepFakes (2017)](https://arxiv.org/abs/1909.11573)
- [Face2Face (2016)](https://arxiv.org/abs/1904.03251)
- [FaceSwap](https://github.com/MarekKowalski/FaceSwap)

**Modern Approaches**:
- [First Order Motion Model (2019)](https://arxiv.org/abs/2003.00196)
- [Wav2Lip (2020)](https://arxiv.org/abs/2008.10010)
- [SimSwap (2020)](https://arxiv.org/abs/2106.06340)
- [HyperReenact (2022)](https://arxiv.org/abs/2203.06814)

**Detection**:
- [FaceForensics++ (2019)](https://arxiv.org/abs/1901.08971)
- See `deepfake-detection/` directory

### Resources

- [FaceForensics++ Dataset](https://github.com/ondyari/FaceForensics)
- [First Order Motion Model Code](https://github.com/AliaksandrSiarohin/first-order-model)
- [Wav2Lip Official Implementation](https://github.com/Rudrabha/Wav2Lip)
- [Deep Learning for Deepfakes Survey](https://arxiv.org/abs/2004.11138)

### Video Courses

- [Computer Vision Courses](https://github.com/Developer-Y/cs-video-courses#computer-vision)
- [Deep Learning Courses](https://github.com/Developer-Y/cs-video-courses#deep-learning)
- [Generative AI and LLMs](https://github.com/Developer-Y/cs-video-courses#generative-ai-and-llms)

## Related Projects

- **deepfake-detection/**: Detection and forensic analysis
- **diffusion-models/**: Modern generative models
- **distributed-training/**: Training large models

## License

Educational purposes only. Follow ethical guidelines and local laws.

## Acknowledgments

Inspired by:
- Original DeepFakes community
- First Order Motion Model by Aliaksandr Siarohin
- Wav2Lip by Prajwal KR
- Face2Face by Thies et al.
- Academic research community working on face synthesis and analysis
