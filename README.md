# DeepFake Detection Research — FYP

## 0. Project Status

**Project type:** Final-Year Project (AIML / Deep Learning / Computer Vision)  
**Research domain:** Deepfake video detection  
**Primary objective:** Design, implement, evaluate, and document an independently developed deepfake detector suitable for an academic research paper.

This repository is **not a continuation, fork, renamed version, or direct modification of the senior project**.

The senior project is treated only as:
1. a reference implementation,
2. a reproducibility target / baseline where feasible, and
3. a comparison point in the final research paper.

---

# 1. Research Problem

Deepfake detection systems often achieve high performance when training and testing conditions are similar, but performance can degrade when the manipulation method, compression, identity, video source, or generation pipeline changes.

The central research problem is therefore:

> **Can a deepfake detector learn manipulation evidence that remains useful across videos and manipulation distributions, rather than relying primarily on dataset-specific visual artifacts?**

This project will investigate that problem through a detector that explicitly combines:

- spatial appearance information,
- frequency-domain information,
- short-term temporal change,
- and temporal consistency across multiple frames.

The proposed system will **not** use the senior project's ResNet50 + BiLSTM architecture.

---

# 2. Reference / Senior Project

The senior project paper is:

> **“High-Fidelity Deepfake Detection Using a CNN and Bidirectional LSTM Pipeline”**

Its reported pipeline is:

```text
Video
  ↓
Frame Sampling
  ↓
YOLOv8 Face Detection
  ↓
Face Cropping
  ↓
ResNet50
  ↓
BiLSTM
  ↓
Real / Fake
```

The reported implementation uses:

- 3,000 videos
- 1,500 real / 1,500 fake
- 30 frames per video
- 224×224 face crops
- ImageNet-pretrained ResNet50
- 2,048-dimensional frame features
- one-layer BiLSTM
- 256 hidden units per direction
- dropout 0.5
- AdamW
- learning rate 3e-4
- batch size 4
- 12 epochs
- BCEWithLogitsLoss
- 70/15/15 train/validation/test split

The paper reports 86.00% test accuracy and 86.79% F1 on its test setup.

**Important:** the senior paper does not clearly identify the exact source/subset corresponding to its 3,000-video experimental dataset. Therefore, this project must NOT assume that dataset is reproducible.

The senior implementation must not be copied and renamed.

---

# 3. Proposed Research Architecture

## 3.1 Working Name

The proposed model will be called:

# **CDTC-Net**
### Cross-Domain Temporal Consistency Network

The name is a working project name. The final paper title/name may change after the literature review.

---

# 4. Core Research Hypothesis

### Hypothesis

> **Deepfake videos contain complementary manipulation evidence in spatial appearance, frequency-domain statistics, and temporal inconsistencies. A detector that explicitly models these complementary signals and their temporal consistency should generalize better to unseen manipulation distributions than a detector based primarily on spatial CNN features followed by recurrent sequence modeling.**

The research is NOT simply:

> “Use three branches because three branches are better.”

The experiments must determine whether each information source contributes useful, complementary evidence.

---

# 5. Proposed Architecture — Exact Specification

The architecture must be implemented as follows unless a later documented literature review demonstrates that a component is technically invalid or redundant.

```text
                         INPUT VIDEO
                              │
                              ▼
                    Temporal Frame Sampler
                              │
                     N = 16 frames/video
                              │
              ┌───────────────┴────────────────┐
              │                                │
              ▼                                ▼
        Face Crop / Align                Full-frame metadata
              │
              ▼
      ┌──────────────────────────────────────────────┐
      │                                              │
      │              TWO-DOMAIN ENCODER              │
      │                                              │
      │   ┌─────────────────┐   ┌─────────────────┐ │
      │   │ RGB Branch      │   │ Frequency       │ │
      │   │                 │   │ Branch          │ │
      │   │ ConvNeXt-Tiny   │   │ FFT/DCT         │ │
      │   │ backbone        │   │ representation  │ │
      │   └────────┬────────┘   └────────┬────────┘ │
      │            │                     │          │
      │            ▼                     ▼          │
      │        RGB token            Frequency token │
      │            │                     │          │
      │            └──────────┬──────────┘          │
      │                       ▼                     │
      │               Cross-Domain Fusion           │
      │                       │                     │
      └───────────────────────┼─────────────────────┘
                              ▼
                    Frame Representation
                              │
                ┌─────────────┴──────────────┐
                │                            │
                ▼                            ▼
        Temporal Difference             Frame Tokens
            Features                        │
                │                            │
                └─────────────┬──────────────┘
                              ▼
                   Temporal Consistency
                         Transformer
                              │
                              ▼
                    Video Representation
                              │
                              ▼
                     Classification Head
                              │
                    ┌─────────┴─────────┐
                    ▼                   ▼
                Real/Fake          Probability
```

---

# 6. Component Specifications

## 6.1 Video Sampling

Each video must be converted into a fixed-length sequence.

Default:

```text
Frames per video: 16
Sampling: uniformly distributed across the usable video duration
```

Do NOT simply take the first 16 frames.

The sampler must support:

- uniform sampling,
- deterministic sampling for validation/test,
- optional randomized temporal sampling during training.

The exact selected frame indices must be reproducible from a random seed.

---

# 7. Face Processing

The model is a facial manipulation detector.

Each sampled frame should undergo:

1. face detection,
2. largest / primary face selection,
3. bounding-box expansion by a small configurable margin,
4. square crop,
5. resize to 224×224,
6. normalization.

### Face detector

**Do not use YOLOv8**, because the senior system uses YOLOv8.

Use a modern dedicated face detector such as **RetinaFace** or another well-supported pretrained face detector.

The detector is a preprocessing component, not the primary research contribution.

The implementation must record:

- detection success/failure,
- bounding box,
- frame index,
- confidence.

If no face is detected, the system must have a deterministic fallback rather than silently dropping the frame.

---

# 8. RGB Branch

The RGB branch extracts spatial facial appearance features.

### Backbone

Use:

**ConvNeXt-Tiny**

with ImageNet pretrained weights where available.

The final classification head must be removed.

The backbone output must be projected to a compact feature dimension:

```text
RGB feature dimension = 256
```

The backbone should initially be partially frozen to stabilize training.

A configuration must allow:

```text
freeze_backbone = true
```

and later:

```text
freeze_backbone = false
```

for controlled experiments.

### Important

Do NOT replace this with:

- ResNet50
- ResNet18
- ResNet101
- BiLSTM

as the proposed model.

Those architectures may be implemented separately as baselines.

---

# 9. Frequency Branch

The frequency branch is intended to capture manipulation traces that may not be obvious in RGB appearance.

For every cropped face frame:

1. convert to grayscale,
2. normalize pixel values,
3. compute a 2-D frequency representation,
4. retain the magnitude information,
5. apply logarithmic scaling,
6. normalize,
7. resize/project into the frequency encoder input representation.

The default frequency representation is:

```text
log(1 + |FFT(x)|)
```

with the DC component shifted away from the center for visualization/processing consistency.

The implementation must preserve the ability to switch between:

```text
FFT
DCT
```

through configuration.

### Frequency encoder

Use a lightweight CNN encoder.

Do NOT simply feed the FFT image into the same RGB backbone.

The frequency branch must produce:

```text
Frequency feature dimension = 256
```

---

# 10. Cross-Domain Fusion

For each frame we have:

```text
RGB feature       = 256-D
Frequency feature = 256-D
```

Concatenate them:

```text
512-D
```

Then use a learned projection:

```text
Linear(512 → 256)
LayerNorm
GELU
Dropout(0.2)
```

Output:

```text
Frame representation = 256-D
```

This creates one representation for each sampled frame.

For 16 frames:

```text
16 × 256
```

---

# 11. Temporal Difference Features

The model must explicitly model changes between consecutive frames.

For frame representations:

```text
F1, F2, ..., F16
```

calculate:

```text
D_t = F_(t+1) - F_t
```

for:

```text
t = 1 ... 15
```

Also calculate the absolute difference:

```text
A_t = |F_(t+1) - F_t|
```

Concatenate:

```text
[D_t ; A_t]
```

and project:

```text
512 → 256
```

using:

```text
Linear
LayerNorm
GELU
Dropout
```

This creates explicit temporal-change features.

---

# 12. Temporal Consistency Transformer

The proposed model must NOT use an LSTM or BiLSTM.

Instead, construct temporal tokens containing:

- frame representation,
- temporal difference representation.

The temporal module should use a lightweight Transformer encoder.

Default:

```text
Embedding dimension: 256
Number of heads: 8
Transformer layers: 2
Feed-forward dimension: 512
Dropout: 0.1
```

Add learned or sinusoidal temporal positional encoding.

A learnable `[CLS]` token must be used to aggregate the video-level representation.

Output:

```text
Video embedding = 256-D
```

---

# 13. Classification Head

The video representation is passed through:

```text
Linear(256 → 128)
GELU
Dropout(0.3)
Linear(128 → 1)
```

The final output is a logit.

Training:

```text
BCEWithLogitsLoss
```

Inference:

```text
sigmoid(logit)
```

Output:

```text
P(fake) ∈ [0,1]
```

Classification threshold:

```text
0.5
```

The threshold must be configurable.

---

# 14. Complete Proposed Model

The complete pipeline is:

```text
Video
 ↓
Uniform 16-frame sampling
 ↓
Face detection + crop + alignment
 ↓
 ┌───────────────────────────────┐
 │                               │
 ▼                               ▼
RGB face                      Frequency map
 │                               │
ConvNeXt-Tiny                  FFT/DCT CNN
 │                               │
256-D                           256-D
 └───────────────┬───────────────┘
                 ▼
       Cross-domain fusion
                 │
               256-D
                 │
       ┌─────────┴─────────┐
       │                   │
       ▼                   ▼
 Frame features       Frame-to-frame
                       differences
       │                   │
       └─────────┬─────────┘
                 ▼
      Temporal Transformer
                 │
               256-D
                 │
          Classification
                 │
          Real / Deepfake
```

This architecture is intentionally different from:

```text
YOLOv8 → ResNet50 → BiLSTM
```

used in the senior project.

---

# 15. Why This Architecture?

The architecture is based on four assumptions that must be experimentally tested:

### A. RGB appearance

Deepfakes can contain spatial artifacts in:

- facial texture,
- boundaries,
- skin appearance,
- blending,
- local structure.

### B. Frequency information

Manipulation and synthesis pipelines can alter high-frequency statistics and local frequency distributions.

### C. Temporal differences

A fake can appear convincing in an individual frame while exhibiting subtle inconsistencies across adjacent frames.

### D. Cross-domain fusion

RGB and frequency representations contain potentially complementary information.

The research question is whether combining these signals improves:

1. in-domain performance,
2. robustness,
3. cross-dataset generalization.

---

# 16. DATASET — FINAL PROJECT REQUIREMENT

## Primary Dataset: FaceForensics++

The primary development/training dataset will be:

# FaceForensics++ (FF++)

FaceForensics++ contains 1,000 original video sequences and manipulated versions generated using multiple manipulation methods, including:

- DeepFakes
- Face2Face
- FaceSwap
- NeuralTextures

The official dataset repository requires completing the dataset access process before the download script/link is provided.

Official repository:

https://github.com/ondyari/FaceForensics

The dataset must be obtained through the official access mechanism.

Do NOT download a random third-party mirror and treat it as the canonical dataset.

---

# 17. FaceForensics++ Version / Compression

The experiments must explicitly record which FF++ version and compression level are used.

The initial primary experiment should use:

```text
FaceForensics++
Compression: c23
```

The project must preserve the ability to evaluate:

```text
c23
c40
```

where available.

### Reason

Compression can substantially affect forensic artifacts.

A detector that only works on pristine/high-quality data is less useful.

---

# 18. Dataset Selection

The first experimental dataset should contain:

### Real

Original FF++ videos.

### Fake

Manipulated FF++ videos.

The four manipulation methods should be retained as separate metadata categories:

```text
DeepFakes
Face2Face
FaceSwap
NeuralTextures
```

Do NOT collapse all metadata into only `real/fake`.

The labels should be:

```text
REAL = 0
FAKE = 1
```

---

# 19. Dataset Split

Splitting must happen at the **source-video / identity level**, not at the frame level.

This is critical.

Never do:

```text
random frames → train/test
```

because frames from the same video can appear in both sets and cause severe leakage.

The split must be:

```text
Train: 70%
Validation: 15%
Test: 15%
```

or use an official published split where appropriate.

Once selected, the exact split must be saved permanently:

```text
data/splits/train.csv
data/splits/val.csv
data/splits/test.csv
```

Every row should contain at minimum:

```text
video_path
label
dataset
manipulation_method
split
source_video_id
```

The split generation script must use a fixed random seed.

---

# 20. Cross-Dataset Generalization Dataset

The secondary evaluation dataset will be:

# Celeb-DF v2

Celeb-DF was designed as a challenging DeepFake dataset with higher-quality synthesized videos and reduced obvious artifacts compared with earlier datasets.

The official repository describes Celeb-DF v2 and provides access through its dataset request process.

Official repository:

https://github.com/yuezunli/celeb-deepfakeforensics

The project must use Celeb-DF as an **unseen external evaluation dataset**, not simply merge it into FF++ training.

---

# 21. Cross-Dataset Experiment

The strongest generalization experiment is:

```text
TRAIN:
FaceForensics++

        ↓

MODEL:
CDTC-Net

        ↓

TEST:
Celeb-DF
```

The model must NOT be fine-tuned on Celeb-DF before reporting the primary cross-dataset result.

This experiment measures whether the learned representation transfers to a different deepfake generation/data distribution.

---

# 22. Optional Modern Generalization Benchmark

If computational resources and dataset access permit, investigate:

**Celeb-DF++**

Celeb-DF++ is a newer benchmark designed specifically around generalizable DeepFake detection and contains multiple unseen DeepFake methods.

It may be used as an additional external evaluation benchmark.

However:

> Celeb-DF++ must not replace the primary FF++ → Celeb-DF experiment unless the research plan is explicitly revised and documented.

---

# 23. Dataset Storage

The datasets themselves must NOT be committed to GitHub.

Recommended structure:

```text
data/
├── raw/
│   ├── faceforensics/
│   └── celebdf/
│
├── processed/
│   ├── face_crops/
│   └── frequency/
│
├── metadata/
│   ├── ffpp_metadata.csv
│   └── celebdf_metadata.csv
│
└── splits/
    ├── train.csv
    ├── val.csv
    └── test.csv
```

Add all large dataset directories to `.gitignore`.

---

# 24. Dataset Manifest

Create:

```text
configs/dataset.yaml
```

It must define:

```yaml
primary_dataset: faceforensics++
compression: c23

frames_per_video: 16

image_size: 224

label_mapping:
  real: 0
  fake: 1

train_ratio: 0.70
val_ratio: 0.15
test_ratio: 0.15

external_dataset: celeb_df
```

The actual local dataset paths must NOT be hard-coded inside Python source files.

---

# 25. Baseline Models

The project must implement and evaluate multiple baselines.

## Baseline 1 — Senior Architecture

Reproduce the senior architecture as closely as possible:

```text
Face detection
→ ResNet50
→ BiLSTM
→ Classifier
```

This is the primary internal comparison.

It must be implemented independently.

Do NOT copy the senior repository's source code.

---

## Baseline 2 — Spatial-only

```text
Face
→ ConvNeXt-Tiny
→ temporal average pooling
→ classifier
```

Purpose:

Determine how much benefit comes from temporal modeling and frequency information.

---

## Baseline 3 — RGB + Temporal Transformer

```text
Face
→ ConvNeXt-Tiny
→ Transformer
→ classifier
```

Purpose:

Determine whether the proposed frequency branch provides additional value.

---

## Baseline 4 — Frequency-only

```text
Face
→ FFT/DCT representation
→ frequency CNN
→ temporal pooling
→ classifier
```

Purpose:

Measure whether frequency information is independently useful.

---

## Proposed Model

```text
RGB
+
Frequency
+
Temporal Differences
+
Temporal Transformer
```

---

# 26. Required Ablation Study

The paper must contain an ablation study.

At minimum:

### A0 — RGB only

```text
RGB
→ ConvNeXt
→ Temporal Transformer
```

### A1 — RGB + Frequency

```text
RGB
+
Frequency
→ Fusion
→ Temporal Transformer
```

### A2 — RGB + Temporal Difference

```text
RGB
+
Temporal Difference
→ Temporal Transformer
```

### A3 — RGB + Frequency + Temporal Difference

```text
RGB
+
Frequency
+
Temporal Difference
→ Temporal Transformer
```

### A4 — Full CDTC-Net

```text
RGB
+
Frequency
+
Temporal Difference
+
Cross-domain fusion
+
Temporal Transformer
```

The purpose is to determine which components actually matter.

---

# 27. Metrics

Every model must report:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC

Also generate:

- confusion matrix,
- ROC curve,
- precision-recall curve,
- per-manipulation performance where possible.

For external evaluation, report the same metrics.

---

# 28. Additional Robustness Tests

If computational resources permit, test:

### Compression robustness

```text
c23 → c40
```

### Frame-count sensitivity

```text
8 frames
16 frames
32 frames
```

### Frequency representation

```text
FFT
DCT
```

### Temporal module

```text
Transformer
LSTM
BiLSTM
```

The latter should be treated as an experimental comparison, NOT as part of the proposed architecture.

---

# 29. Training Configuration

Initial configuration:

```yaml
optimizer: AdamW

learning_rate: 0.0001

weight_decay: 0.01

batch_size: automatically determined by available VRAM

epochs: 30

loss: BCEWithLogitsLoss

early_stopping:
  enabled: true
  patience: 5

mixed_precision: true

seed: 42
```

Batch size must NOT be hard-coded to 4 simply because the senior project used 4.

The batch size should be determined based on GPU memory and documented.

---

# 30. Reproducibility

Every experiment must record:

```text
random seed
dataset version
dataset split
compression level
frame count
image resolution
face detector
backbone
frequency method
optimizer
learning rate
batch size
epochs
early stopping
hardware
software versions
model checkpoint
```

Create:

```text
experiments/
```

with one directory per experiment.

Example:

```text
experiments/
└── cdtc_ffpp_c23_seed42/
    ├── config.yaml
    ├── metrics.json
    ├── training_log.csv
    ├── confusion_matrix.png
    ├── roc_curve.png
    └── best_model.pt
```

---

# 31. Required Project Structure

The final repository should follow approximately:

```text
deepfake-detection-research/
│
├── README.md
├── LICENSE
├── .gitignore
├── requirements.txt
│
├── configs/
│   ├── dataset.yaml
│   ├── model.yaml
│   └── training.yaml
│
├── src/
│   ├── data/
│   │   ├── dataset.py
│   │   ├── sampler.py
│   │   ├── face_detection.py
│   │   ├── preprocessing.py
│   │   └── splits.py
│   │
│   ├── models/
│   │   ├── rgb_encoder.py
│   │   ├── frequency_encoder.py
│   │   ├── fusion.py
│   │   ├── temporal_difference.py
│   │   ├── temporal_transformer.py
│   │   ├── cdtc_net.py
│   │   └── baselines/
│   │       ├── senior_resnet_bilstm.py
│   │       ├── rgb_only.py
│   │       ├── rgb_transformer.py
│   │       └── frequency_only.py
│   │
│   ├── training/
│   │   ├── train.py
│   │   ├── validate.py
│   │   └── losses.py
│   │
│   ├── evaluation/
│   │   ├── evaluate.py
│   │   ├── metrics.py
│   │   ├── plots.py
│   │   └── cross_dataset.py
│   │
│   └── utils/
│       ├── seed.py
│       ├── logging.py
│       └── checkpointing.py
│
├── scripts/
│   ├── build_metadata.py
│   ├── create_splits.py
│   ├── preprocess_dataset.py
│   ├── train_baseline.py
│   ├── train_cdtc.py
│   └── evaluate.py
│
├── experiments/
│
├── notebooks/
│
├── docs/
│   ├── research_notes.md
│   ├── experiment_log.md
│   └── literature_review.md
│
└── data/
    ├── raw/
    ├── processed/
    ├── metadata/
    └── splits/
```

---

# 32. Software Requirements

Use Python and PyTorch.

Preferred stack:

```text
Python 3.11+
PyTorch
torchvision
OpenCV
NumPy
Pandas
scikit-learn
Pillow
PyYAML
Matplotlib
tqdm
```

Additional libraries may be added only when justified.

The project must support CUDA when available.

The code must automatically detect:

```text
CUDA
CPU
```

and print the selected device.

---

# 33. No Hidden Dataset Assumptions

The code must NEVER assume:

```text
D:\...
C:\...
/home/...
```

or any specific machine path.

All paths must come from configuration or command-line arguments.

---

# 34. No Dataset Leakage

This is a hard requirement.

Never allow:

- frames from the same video in multiple splits,
- manipulated versions derived from the same source video across train/test,
- external test videos to enter training,
- preprocessing statistics calculated using test data,
- test-set tuning,
- threshold tuning on the final test set.

All split generation must happen before training.

---

# 35. Research Integrity

The project must never:

- fabricate results,
- invent accuracy values,
- invent dataset sizes,
- claim a model is novel without literature verification,
- claim state-of-the-art performance without proper benchmarking,
- copy source code and present it as original,
- copy a paper's architecture and rename it,
- tune repeatedly on the test set,
- hide failed experiments.

Failed experiments are valuable research evidence and should be logged.

---

# 36. Literature Review Requirement

Before making a final novelty claim, perform a literature review covering:

- CNN-based deepfake detection,
- frequency-domain detection,
- spatial-frequency dual-stream methods,
- temporal deepfake detection,
- temporal transformers,
- frame-difference modeling,
- cross-dataset generalization,
- domain generalization,
- FaceForensics++,
- Celeb-DF,
- modern generalization benchmarks.

The literature review must answer:

1. What has already been done?
2. What architectures are common?
3. What are their weaknesses?
4. Which combinations already exist?
5. What is genuinely different about CDTC-Net?
6. Is the proposed contribution actually novel enough for an FYP paper?

If literature shows that the exact architecture already exists, **do not falsely claim novelty**.

Instead, revise the research contribution while preserving the overall research objective.

---

# 37. Research Contribution

The intended contribution is NOT:

> “We used ConvNeXt.”

It is NOT:

> “We used FFT.”

It is NOT:

> “We used a Transformer.”

Those are existing techniques.

The intended contribution is the **systematic investigation of complementary spatial, frequency, and temporal-consistency signals for deepfake video detection, with explicit cross-dataset evaluation.**

The paper should demonstrate:

```text
Component
    ↓
Hypothesis
    ↓
Controlled experiment
    ↓
Measured result
    ↓
Interpretation
```

---

# 38. Expected Research Questions

The project should answer:

### RQ1
Does adding frequency-domain information improve deepfake detection over RGB-only representations?

### RQ2
Does explicit frame-to-frame difference modeling improve detection?

### RQ3
Does cross-domain fusion provide complementary information?

### RQ4
Does the proposed temporal Transformer improve over recurrent temporal modeling?

### RQ5
Does the proposed representation generalize better to Celeb-DF than the senior ResNet50-BiLSTM baseline?

### RQ6
Which components contribute most to robustness under compression and dataset shift?

---

# 39. Paper Structure

The final paper should approximately follow:

## 1. Abstract

Problem → method → experiments → results → contribution.

## 2. Introduction

- deepfakes,
- detection challenge,
- generalization problem,
- research gap,
- proposed solution,
- contributions.

## 3. Related Work

- spatial detectors,
- frequency-based detectors,
- temporal detectors,
- transformer approaches,
- cross-dataset generalization.

## 4. Dataset

- FF++,
- Celeb-DF,
- preprocessing,
- splits,
- compression.

## 5. Proposed Method

Detailed architecture.

## 6. Experimental Setup

- hardware,
- training,
- metrics,
- baselines.

## 7. Results

- primary results,
- ablation,
- cross-dataset,
- robustness.

## 8. Discussion

Explain why the model succeeds/fails.

## 9. Limitations

Be honest.

## 10. Conclusion

Summarize findings.

---

# 40. What Antigravity Must Do First

Antigravity must NOT immediately start training.

Follow these phases.

## Phase 1 — Repository setup

Create:

- directory structure,
- configuration system,
- requirements,
- `.gitignore`,
- logging,
- experiment system,
- dataset abstraction.

## Phase 2 — Research verification

Create:

```text
docs/literature_review.md
```

Review relevant recent literature and determine whether the proposed CDTC-Net architecture contains an actual research gap.

If an exact or near-identical architecture is found, document it and propose a defensible modification.

Do not fabricate novelty.

## Phase 3 — Dataset preparation

Wait for the user to obtain official dataset access.

Do not download datasets automatically without user instruction.

Once the user provides the dataset location:

1. inspect it,
2. identify videos,
3. generate metadata,
4. verify labels,
5. generate leakage-safe splits,
6. run dataset integrity checks.

## Phase 4 — Data pipeline test

Before training:

- load one real video,
- load one fake video,
- sample 16 frames,
- detect faces,
- create crops,
- calculate frequency maps,
- verify tensors,
- verify labels.

Create visualization samples.

## Phase 5 — Baselines

Implement and validate:

1. senior ResNet50-BiLSTM baseline,
2. RGB-only baseline,
3. RGB + Transformer baseline,
4. frequency-only baseline.

## Phase 6 — Proposed model

Implement:

```text
CDTC-Net
```

and verify tensor dimensions at every stage.

## Phase 7 — Small-scale sanity training

Train on a tiny subset first.

Purpose:

- catch bugs,
- verify gradients,
- verify loss decreases,
- verify overfitting on a tiny sample.

Do NOT report this as a research result.

## Phase 8 — Full experiments

Run the predefined experiments.

## Phase 9 — Ablation

Run all required ablations.

## Phase 10 — Cross-dataset evaluation

Train on FF++.

Evaluate on Celeb-DF.

Do not train on Celeb-DF before the primary external test.

## Phase 11 — Analysis

Generate:

- tables,
- plots,
- confusion matrices,
- ROC curves,
- per-manipulation results,
- failure examples.

## Phase 12 — Paper support

Generate research notes and result summaries.

Do not fabricate prose claiming results before the experiments actually exist.

---

# 41. Antigravity Operating Rules

You are an AI coding/research agent working on this repository.

### Rule 1

Read this README completely before modifying the repository.

### Rule 2

Do not silently change the research architecture.

### Rule 3

Do not silently change the dataset.

### Rule 4

Do not download large datasets without explicit user approval.

### Rule 5

Do not invent experimental results.

### Rule 6

Do not use the senior repository as a code dependency.

### Rule 7

Do not copy senior source code.

### Rule 8

Do not claim CDTC-Net is novel until the literature review supports that claim.

### Rule 9

Every experiment must have a reproducible configuration.

### Rule 10

Every result must be traceable to an experiment configuration and checkpoint.

### Rule 11

Never train/test on individual frames as independent samples for the primary video-level experiment.

### Rule 12

Prevent identity/source-video leakage.

### Rule 13

Prefer simple, readable code over unnecessary abstraction.

### Rule 14

Do not add components merely because they sound impressive.

### Rule 15

If a proposed change alters the scientific question, stop and ask the user.

---

# 42. Definition of Done

The project is considered complete only when:

- [ ] Dataset acquisition is documented.
- [ ] Dataset integrity is verified.
- [ ] Leakage-safe splits exist.
- [ ] Data pipeline works.
- [ ] Face preprocessing works.
- [ ] RGB branch works.
- [ ] Frequency branch works.
- [ ] Cross-domain fusion works.
- [ ] Temporal difference module works.
- [ ] Temporal Transformer works.
- [ ] CDTC-Net trains successfully.
- [ ] Senior ResNet50-BiLSTM baseline is implemented independently.
- [ ] Other required baselines are implemented.
- [ ] Ablation study is completed.
- [ ] FF++ test evaluation is completed.
- [ ] Celeb-DF external evaluation is completed.
- [ ] Metrics are reported.
- [ ] Confusion matrices are generated.
- [ ] ROC-AUC is reported.
- [ ] Compression robustness is investigated where feasible.
- [ ] Results are reproducible.
- [ ] Failed experiments are documented.
- [ ] Literature review is complete.
- [ ] Novelty claims are supported.
- [ ] Research paper draft is based only on actual experimental results.

---

# 43. Final Research Philosophy

This project is not an attempt to make the biggest model possible.

It is an attempt to answer a meaningful research question with controlled experiments.

The most important principle is:

```text
Research question
      ↓
Hypothesis
      ↓
Architecture
      ↓
Controlled experiment
      ↓
Evidence
      ↓
Conclusion
```

Not:

```text
Cool model
      ↓
Train it
      ↓
Get accuracy
      ↓
Invent explanation
```

The final system should be technically strong, independently implemented, reproducible, experimentally justified, and defensible as an academic Final-Year Project.

---

# 44. Reference Material

### Senior paper

The senior paper supplied with this project is the reference for the ResNet50 + BiLSTM baseline and must be cited appropriately in the final paper.

### FaceForensics++

Official repository:

https://github.com/ondyari/FaceForensics

### Celeb-DF

Official repository:

https://github.com/yuezunli/celeb-deepfakeforensics

### Celeb-DF++

Official repository:

https://github.com/OUC-VAS/Celeb-DF-PP

---

# 45. Immediate Next Action

When this README is first loaded by Antigravity:

**DO NOT START MODEL TRAINING.**

First:

1. read this entire README,
2. inspect the repository,
3. create the project structure,
4. create the configuration system,
5. create the dataset abstraction,
6. create the experiment/logging framework,
7. perform the literature verification,
8. report whether CDTC-Net is sufficiently distinct from existing published methods,
9. wait for official dataset files/access before preprocessing or training.

The first milestone is therefore:

> **A clean, reproducible research codebase ready for the officially obtained FaceForensics++ dataset, with the proposed CDTC-Net architecture specified and the research gap verified.**
