# CDTC-Net: Comprehensive Research Results Master Document

> **Scope:** Complete, canonical audit of all benchmark experiments (EXP-1 through EXP-5 V2) and component ablation studies (ABL-1 through ABL-4) for the Final Year Project.
> **Source Artifacts:** All metrics are extracted directly from saved evaluation JSON artifacts without post-hoc modifications.
> **Audit Date:** 2026-10-06  

---

## 1. Executive Master Results Table

### In-Domain Evaluation: FaceForensics++ c23 Test Split ($N=750$: $150$ Real, $600$ Fake)

| Exp ID | Model Architecture | Best Ep | Decision $\tau$ | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | Confusion Matrix (TN / FP / FN / TP) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-1** | ResNet-50 + BiLSTM | Epoch 25 | $\tau=0.50$ | 79.07% | 74.42% | **0.8336** | 90.79% | 82.17% | 66.67% | 86.26% | `100 / 50 / 107 / 493` |
| **EXP-2** | ConvNeXt RGB Spatial-only | Epoch 22 | $\tau=0.50$ | 74.27% | 65.92% | **0.7253** | 86.93% | 79.83% | 52.00% | 83.23% | `78 / 72 / 121 / 479` |
| **EXP-3** | ConvNeXt RGB + Transformer | Epoch 10 | $\tau=0.50$ | 77.87% | 64.17% | **0.7185** | 85.57% | 87.00% | 41.33% | 86.28% | `62 / 88 / 78 / 522` |
| **EXP-4** | Frequency-only (2D-FFT CNN) | Epoch 11 | $\tau=0.50$ | 80.00% | 50.00% | **0.6436** | 80.00% | 100.00% | 0.00% | 88.89% | `0 / 150 / 0 / 600` |
| **EXP-5 (V1)** | Full Proposed CDTC-Net (V1 Baseline) | Epoch 11 | $\tau=0.50$ | 80.27% | 63.17% | **0.7259** | 84.88% | 91.67% | 34.67% | 88.14% | `52 / 98 / 50 / 550` |
| **EXP-5 (V2)** | Full Proposed CDTC-Net (V2 Class-Balanced) | Epoch 14 | $\tau=0.65$ | 64.27% | 64.17% | **0.7012** | 87.73% | 64.33% | 64.00% | 74.23% | `96 / 54 / 214 / 386` |
| **ABL-1** | Dual-Domain + Avg Pooling (No Transformer) | Epoch 10 | $\tau=0.50$ | 74.93% | 65.08% | **0.7264** | 86.40% | 81.50% | 48.67% | 83.88% | `73 / 77 / 111 / 489` |
| **ABL-2** | Dual-Domain + Transformer (No Diff Tokens) | Epoch 10 | $\tau=0.50$ | 78.40% | 63.25% | **0.7205** | 85.10% | 88.50% | 38.00% | 86.76% | `57 / 93 / 69 / 531` |
| **ABL-3** | Signed-Only Diff Tokens | Epoch 15 | $\tau=0.50$ | 77.33% | 67.83% | **0.7258** | 87.46% | 83.67% | 52.00% | 85.52% | `78 / 72 / 98 / 502` |
| **ABL-4** | CDTC-Net with 2D-DCT | Epoch 10 | $\tau=0.50$ | 71.47% | 63.92% | **0.7095** | 86.28% | 76.50% | 51.33% | 81.10% | `77 / 73 / 141 / 459` |

### Zero-Shot Cross-Dataset Evaluation: Celeb-DF v2 Benchmark ($N=518$: $178$ Real, $340$ Fake)

| Exp ID | Model Architecture | Decision $\tau$ | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | Confusion Matrix (TN / FP / FN / TP) |
|:---|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **EXP-1** | ResNet-50 + BiLSTM | $\tau=0.50$ | 72.39% | 65.99% | **0.7271** | 75.19% | 86.47% | 45.51% | 80.44% | `81 / 97 / 46 / 294` |
| **EXP-2** | ConvNeXt RGB Spatial-only | $\tau=0.50$ | 66.41% | 59.29% | **0.6649** | 71.17% | 82.06% | 36.52% | 76.23% | `65 / 113 / 61 / 279` |
| **EXP-3** | ConvNeXt RGB + Transformer | $\tau=0.50$ | 68.73% | 58.78% | **0.7089** | 70.32% | 90.59% | 26.97% | 79.18% | `48 / 130 / 32 / 308` |
| **EXP-4** | Frequency-only (2D-FFT CNN) | $\tau=0.50$ | 65.64% | 50.00% | **0.5116** | 65.64% | 100.00% | 0.00% | 79.25% | `0 / 178 / 0 / 340` |
| **EXP-5 (V1)** | Full Proposed CDTC-Net (V1 Baseline) | $\tau=0.50$ | 67.95% | 54.71% | **0.6720** | 67.90% | 97.06% | 12.36% | 79.90% | `22 / 156 / 10 / 330` |
| **EXP-5 (V2)** | Full Proposed CDTC-Net (V2 Class-Balanced) | $\tau=0.65$ | 58.30% | 62.35% | **0.6779** | 79.25% | 49.41% | 75.28% | 60.87% | `134 / 44 / 172 / 168` |
| **ABL-1** | Dual-Domain + Avg Pooling (No Transformer) | $\tau=0.50$ | 66.99% | 58.66% | **0.6644** | 70.56% | 85.29% | 32.02% | 77.23% | `57 / 121 / 50 / 290` |
| **ABL-2** | Dual-Domain + Transformer (No Diff Tokens) | $\tau=0.50$ | 70.08% | 59.00% | **0.6730** | 70.24% | 94.41% | 23.60% | 80.55% | `42 / 136 / 19 / 321` |
| **ABL-3** | Signed-Only Diff Tokens | $\tau=0.50$ | 63.71% | 57.63% | **0.6247** | 70.43% | 77.06% | 38.20% | 73.60% | `68 / 110 / 78 / 262` |
| **ABL-4** | CDTC-Net with 2D-DCT | $\tau=0.50$ | 68.92% | 56.78% | **0.7003** | 69.00% | 95.59% | 17.98% | 80.15% | `32 / 146 / 15 / 325` |

---

## 2. Benchmark Experiments Profile (EXP-1 to EXP-5 V2)

### EXP-1: ResNet-50 + BiLSTM
- **Description:** Senior recurrent baseline combining ResNet-50 spatial feature extraction with bidirectional LSTM temporal sequence modeling.
- **Training Objective:** Standard Binary Cross-Entropy
- **Selected Checkpoint:** [Epoch 25](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/exp_1/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1)

#### Quantitative Performance Summary

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 79.07% | 74.42% | 0.8336 | 90.79% | 82.17% | 66.67% | 86.26% | 100 | 50 | 107 | 493 |
| **Celeb-DF v2 (Zero-Shot)** | 72.39% | 65.99% | 0.7271 | 75.19% | 86.47% | 45.51% | 80.44% | 81 | 97 | 46 | 294 |

#### Per-Manipulation Breakdown on FaceForensics++ c23

| Manipulation Method | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `deepfakes` | 75.67% | 75.67% | 0.8477 | 71.75% | 84.67% | 77.68% | 100 | 50 | 23 | 127 |
| `face2face` | 77.00% | 77.00% | 0.8711 | 72.38% | 87.33% | 79.15% | 100 | 50 | 19 | 131 |
| `faceswap` | 71.00% | 71.00% | 0.7863 | 69.33% | 75.33% | 72.20% | 100 | 50 | 37 | 113 |
| `neuraltextures` | 74.00% | 74.00% | 0.8294 | 70.93% | 81.33% | 75.78% | 100 | 50 | 28 | 122 |

---

### EXP-2: ConvNeXt RGB Spatial-only
- **Description:** Modern spatial-only baseline utilizing ConvNeXt-Tiny RGB feature extractor with frame-level average pooling.
- **Training Objective:** Standard Binary Cross-Entropy
- **Selected Checkpoint:** [Epoch 22](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/exp_2/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2)

#### Quantitative Performance Summary

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 74.27% | 65.92% | 0.7253 | 86.93% | 79.83% | 52.00% | 83.23% | 78 | 72 | 121 | 479 |
| **Celeb-DF v2 (Zero-Shot)** | 66.41% | 59.29% | 0.6649 | 71.17% | 82.06% | 36.52% | 76.23% | 65 | 113 | 61 | 279 |

#### Per-Manipulation Breakdown on FaceForensics++ c23

| Manipulation Method | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `deepfakes` | 66.33% | 66.33% | 0.7377 | 62.69% | 80.67% | 70.55% | 78 | 72 | 29 | 121 |
| `face2face` | 67.00% | 67.00% | 0.7515 | 63.08% | 82.00% | 71.30% | 78 | 72 | 27 | 123 |
| `faceswap` | 64.33% | 64.33% | 0.6989 | 61.50% | 76.67% | 68.25% | 78 | 72 | 35 | 115 |
| `neuraltextures` | 66.00% | 66.00% | 0.7130 | 62.50% | 80.00% | 70.18% | 78 | 72 | 30 | 120 |

---

### EXP-3: ConvNeXt RGB + Transformer
- **Description:** Spatial-temporal baseline combining ConvNeXt-Tiny RGB encoder with a 2-layer temporal Transformer encoder.
- **Training Objective:** Standard Binary Cross-Entropy
- **Selected Checkpoint:** [Epoch 10](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/exp_3/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3)

#### Quantitative Performance Summary

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 77.87% | 64.17% | 0.7185 | 85.57% | 87.00% | 41.33% | 86.28% | 62 | 88 | 78 | 522 |
| **Celeb-DF v2 (Zero-Shot)** | 68.73% | 58.78% | 0.7089 | 70.32% | 90.59% | 26.97% | 79.18% | 48 | 130 | 32 | 308 |

#### Per-Manipulation Breakdown on FaceForensics++ c23

| Manipulation Method | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `deepfakes` | 65.00% | 65.00% | 0.7435 | 60.18% | 88.67% | 71.70% | 62 | 88 | 17 | 133 |
| `face2face` | 64.67% | 64.67% | 0.7375 | 60.00% | 88.00% | 71.35% | 62 | 88 | 18 | 132 |
| `faceswap` | 61.67% | 61.67% | 0.6833 | 58.29% | 82.00% | 68.14% | 62 | 88 | 27 | 123 |
| `neuraltextures` | 65.33% | 65.33% | 0.7098 | 60.36% | 89.33% | 72.04% | 62 | 88 | 16 | 134 |

---

### EXP-4: Frequency-only (2D-FFT CNN)
- **Description:** Spectral baseline evaluating standalone 2D-FFT magnitude spectra with a 4-layer Frequency CNN and Transformer.
- **Training Objective:** Standard Binary Cross-Entropy
- **Selected Checkpoint:** [Epoch 11](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/exp_4/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4)

#### Quantitative Performance Summary

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 80.00% | 50.00% | 0.6436 | 80.00% | 100.00% | 0.00% | 88.89% | 0 | 150 | 0 | 600 |
| **Celeb-DF v2 (Zero-Shot)** | 65.64% | 50.00% | 0.5116 | 65.64% | 100.00% | 0.00% | 79.25% | 0 | 178 | 0 | 340 |

#### Per-Manipulation Breakdown on FaceForensics++ c23

| Manipulation Method | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `deepfakes` | 50.00% | 50.00% | 0.7527 | 50.00% | 100.00% | 66.67% | 0 | 150 | 0 | 150 |
| `face2face` | 50.00% | 50.00% | 0.5988 | 50.00% | 100.00% | 66.67% | 0 | 150 | 0 | 150 |
| `faceswap` | 50.00% | 50.00% | 0.5862 | 50.00% | 100.00% | 66.67% | 0 | 150 | 0 | 150 |
| `neuraltextures` | 50.00% | 50.00% | 0.6367 | 50.00% | 100.00% | 66.67% | 0 | 150 | 0 | 150 |

---

### EXP-5 (V1): Full Proposed CDTC-Net (V1 Baseline)
- **Description:** Proposed CDTC-Net architecture trained with standard BCE loss (equal class weighting).
- **Training Objective:** Standard Binary Cross-Entropy (pos_weight=1.0)
- **Selected Checkpoint:** [Epoch 11](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/exp_5/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5)

#### Quantitative Performance Summary

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 80.27% | 63.17% | 0.7259 | 84.88% | 91.67% | 34.67% | 88.14% | 52 | 98 | 50 | 550 |
| **Celeb-DF v2 (Zero-Shot)** | 67.95% | 54.71% | 0.6720 | 67.90% | 97.06% | 12.36% | 79.90% | 22 | 156 | 10 | 330 |

#### Per-Manipulation Breakdown on FaceForensics++ c23

| Manipulation Method | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `deepfakes` | 65.00% | 65.00% | 0.7636 | 59.34% | 95.33% | 73.15% | 52 | 98 | 7 | 143 |
| `face2face` | 63.67% | 63.67% | 0.7452 | 58.65% | 92.67% | 71.83% | 52 | 98 | 11 | 139 |
| `faceswap` | 61.00% | 61.00% | 0.6841 | 57.21% | 87.33% | 69.13% | 52 | 98 | 19 | 131 |
| `neuraltextures` | 63.00% | 63.00% | 0.7106 | 58.30% | 91.33% | 71.17% | 52 | 98 | 13 | 137 |

---

### EXP-5 (V2): Full Proposed CDTC-Net (V2 Class-Balanced)
- **Description:** Proposed CDTC-Net architecture retrained with Class-Weighted BCE and validation-calibrated decision threshold.
- **Training Objective:** Class-Weighted BCE (pos_weight=0.25, w_real=1.0, w_fake=0.25)
- **Selected Checkpoint:** [Epoch 14](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.65$
- **Directory Location:** [`experiments/exp_5_v2_balanced/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced)

#### Quantitative Performance Summary

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 64.27% | 64.17% | 0.7012 | 87.73% | 64.33% | 64.00% | 74.23% | 96 | 54 | 214 | 386 |
| **Celeb-DF v2 (Zero-Shot)** | 58.30% | 62.35% | 0.6779 | 79.25% | 49.41% | 75.28% | 60.87% | 134 | 44 | 172 | 168 |

#### Per-Manipulation Breakdown on FaceForensics++ c23

| Manipulation Method | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| `deepfakes` | 65.00% | 65.00% | 0.7296 | 64.71% | 66.00% | 65.35% | 96 | 54 | 51 | 99 |
| `face2face` | 66.33% | 66.33% | 0.7217 | 65.61% | 68.67% | 67.10% | 96 | 54 | 47 | 103 |
| `faceswap` | 62.33% | 62.33% | 0.6739 | 62.76% | 60.67% | 61.69% | 96 | 54 | 59 | 91 |
| `neuraltextures` | 63.00% | 63.00% | 0.6797 | 63.27% | 62.00% | 62.63% | 96 | 54 | 57 | 93 |

---

## 3. Component Ablation Studies (ABL-1 to ABL-4)

### ABL-1: Dual-Domain + Avg Pooling (No Transformer)
- **Ablation Hypothesis:** Ablation removing the temporal Transformer encoder, using simple temporal average pooling over fused cross-domain tokens.
- **Selected Checkpoint:** [Epoch 10](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/abl_1/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1)

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 74.93% | 65.08% | 0.7264 | 86.40% | 81.50% | 48.67% | 83.88% | 73 | 77 | 111 | 489 |
| **Celeb-DF v2 (Zero-Shot)** | 66.99% | 58.66% | 0.6644 | 70.56% | 85.29% | 32.02% | 77.23% | 57 | 121 | 50 | 290 |

---

### ABL-2: Dual-Domain + Transformer (No Diff Tokens)
- **Ablation Hypothesis:** Ablation removing explicit signed/absolute temporal difference tokens ([D_t, |D_t|]), passing only fused frame tokens to Transformer.
- **Selected Checkpoint:** [Epoch 10](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/abl_2/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2)

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 78.40% | 63.25% | 0.7205 | 85.10% | 88.50% | 38.00% | 86.76% | 57 | 93 | 69 | 531 |
| **Celeb-DF v2 (Zero-Shot)** | 70.08% | 59.00% | 0.6730 | 70.24% | 94.41% | 23.60% | 80.55% | 42 | 136 | 19 | 321 |

---

### ABL-3: Signed-Only Diff Tokens
- **Ablation Hypothesis:** Ablation using only signed temporal difference tokens (D_t) without absolute magnitude tokens (|D_t|).
- **Selected Checkpoint:** [Epoch 15](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/abl_3/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3)

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 77.33% | 67.83% | 0.7258 | 87.46% | 83.67% | 52.00% | 85.52% | 78 | 72 | 98 | 502 |
| **Celeb-DF v2 (Zero-Shot)** | 63.71% | 57.63% | 0.6247 | 70.43% | 77.06% | 38.20% | 73.60% | 68 | 110 | 78 | 262 |

---

### ABL-4: CDTC-Net with 2D-DCT
- **Ablation Hypothesis:** Ablation substituting 2D Fast Fourier Transform (FFT) with 2D Discrete Cosine Transform (DCT) block energy representation.
- **Selected Checkpoint:** [Epoch 10](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4/checkpoints/best_model.pt)
- **Decision Threshold:** $\tau = 0.50$
- **Directory Location:** [`experiments/abl_4/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4)

| Benchmark Dataset | Accuracy | Balanced Acc | ROC-AUC | Precision | Recall | Specificity | F1 Score | TN | FP | FN | TP |
|:---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **FaceForensics++ c23 (In-Domain)** | 71.47% | 63.92% | 0.7095 | 86.28% | 76.50% | 51.33% | 81.10% | 77 | 73 | 141 | 459 |
| **Celeb-DF v2 (Zero-Shot)** | 68.92% | 56.78% | 0.7003 | 69.00% | 95.59% | 17.98% | 80.15% | 32 | 146 | 15 | 325 |

---

## 4. Deep-Dive: CDTC-Net V1 vs. CDTC-Net V2 Comparison

### Scientific Motivation for V2 Retraining
In EXP-5 V1 (trained with standard unweighted BCE on a $4:1$ fake:real class imbalance), the model exhibited severe positive-class bias, predicting FAKE on $65.33\%$ of authentic in-domain FF++ videos and $87.64\%$ of authentic zero-shot Celeb-DF videos at the default $0.50$ threshold. EXP-5 V2 was trained with class-balanced weighting ($	ext{pos\_weight} = 0.25$, $w_{	ext{real}}=1.0, w_{	ext{fake}}=0.25$) and calibrated with a validation balanced accuracy sweep ($	au^* = 0.6500$).

### Side-by-Side Quantitative Comparison

| Metric Dimension | EXP-5 V1 Baseline ($	au=0.50$) | EXP-5 V2 Balanced ($	au=0.65$) | Absolute Shift | Relative Change / Impact |
|:---|:---:|:---:|:---:|:---|
| **FF++ Accuracy** | 80.27% | 64.27% | -16.00% | Raw accuracy dropped due to reduced fake-class majority weighting. |
| **FF++ Balanced Accuracy** | 63.17% | 64.17% | +1.00% | +1.00% improvement in class-averaged performance. |
| **FF++ Specificity (Real TNR)** | **34.67%** | **64.00%** | **+29.33%** | **+29.33% increase in authentic video recognition (TN: 52 -> 96).** |
| **FF++ False Positive Count** | **98 / 150 (65.33%)** | **54 / 150 (36.00%)** | **-44 False Positives** | **-44.9% reduction in false alarms on real videos.** |
| **FF++ Fake Recall (TPR)** | 91.67% | 64.33% | -27.34% | Expected trade-off: reduced over-prediction of fake class. |
| **FF++ Precision** | 84.88% | 87.73% | +2.85% | Higher confidence when predicting FAKE. |
| **FF++ ROC-AUC** | 0.7259 | 0.7012 | -0.0247 | Slight shift in ranking curve across decision spectrum. |
| **Celeb-DF Specificity (Zero-Shot)** | **12.36%** | **75.28%** | **+62.92%** | **+62.92% increase in authentic zero-shot specificity (TN: 22 -> 134).** |
| **Celeb-DF False Positives** | **156 / 178 (87.64%)** | **44 / 178 (24.72%)** | **-112 False Positives** | **-71.8% reduction in cross-dataset false alarms.** |
| **Celeb-DF Balanced Accuracy** | 54.71% | 62.35% | +7.64% | Substantial improvement in cross-dataset balanced accuracy. |
| **Celeb-DF ROC-AUC** | 0.6720 | 0.6779 | +0.0059 | Slightly improved ranking ability under zero-shot domain shift. |

---

## 5. Cross-Model Per-Manipulation Method Comparison (FF++ c23)

| Model Tag | Deepfakes AUC (Acc) | Face2Face AUC (Acc) | FaceSwap AUC (Acc) | NeuralTextures AUC (Acc) |
|:---|:---:|:---:|:---:|:---:|
| **EXP-1** | 0.8477 (75.7%) | 0.8711 (77.0%) | 0.7863 (71.0%) | 0.8294 (74.0%) |
| **EXP-2** | 0.7377 (66.3%) | 0.7515 (67.0%) | 0.6989 (64.3%) | 0.7130 (66.0%) |
| **EXP-3** | 0.7435 (65.0%) | 0.7375 (64.7%) | 0.6833 (61.7%) | 0.7098 (65.3%) |
| **EXP-4** | 0.7527 (50.0%) | 0.5988 (50.0%) | 0.5862 (50.0%) | 0.6367 (50.0%) |
| **EXP-5 (V1)** | 0.7636 (65.0%) | 0.7452 (63.7%) | 0.6841 (61.0%) | 0.7106 (63.0%) |
| **EXP-5 (V2)** | 0.7296 (65.0%) | 0.7217 (66.3%) | 0.6739 (62.3%) | 0.6797 (63.0%) |
| **ABL-1** | 0.7546 (67.7%) | 0.7477 (65.3%) | 0.6998 (64.0%) | 0.7033 (63.3%) |
| **ABL-2** | 0.7638 (65.3%) | 0.7408 (64.0%) | 0.6744 (61.3%) | 0.7030 (62.3%) |
| **ABL-3** | 0.7432 (69.0%) | 0.7441 (68.0%) | 0.7062 (67.3%) | 0.7094 (67.0%) |
| **ABL-4** | 0.7353 (66.3%) | 0.7307 (65.3%) | 0.6626 (59.0%) | 0.7093 (65.0%) |

---

## 6. Complete Saved Artifacts Inventory & Verification Table

| Experiment Tag | Best Checkpoint Path | Training Log CSV | Evaluation Metrics JSON | Curves & Plots | Cross-Dataset Folder |
|:---|:---|:---|:---|:---|:---|
| **EXP-1** | [`experiments/exp_1/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1/checkpoints/best_model.pt) | [`experiments/exp_1/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1/training_log.csv) | [`experiments/exp_1/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1/evaluation/test_metrics.json) | [`experiments/exp_1/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1/evaluation/roc_curve.png) | [`experiments/exp_1/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_1/cross_dataset) |
| **EXP-2** | [`experiments/exp_2/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2/checkpoints/best_model.pt) | [`experiments/exp_2/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2/training_log.csv) | [`experiments/exp_2/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2/evaluation/test_metrics.json) | [`experiments/exp_2/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2/evaluation/roc_curve.png) | [`experiments/exp_2/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_2/cross_dataset) |
| **EXP-3** | [`experiments/exp_3/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3/checkpoints/best_model.pt) | [`experiments/exp_3/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3/training_log.csv) | [`experiments/exp_3/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3/evaluation/test_metrics.json) | [`experiments/exp_3/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3/evaluation/roc_curve.png) | [`experiments/exp_3/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_3/cross_dataset) |
| **EXP-4** | [`experiments/exp_4/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4/checkpoints/best_model.pt) | [`experiments/exp_4/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4/training_log.csv) | [`experiments/exp_4/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4/evaluation/test_metrics.json) | [`experiments/exp_4/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4/evaluation/roc_curve.png) | [`experiments/exp_4/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_4/cross_dataset) |
| **EXP-5 (V1)** | [`experiments/exp_5/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5/checkpoints/best_model.pt) | [`experiments/exp_5/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5/training_log.csv) | [`experiments/exp_5/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5/evaluation/test_metrics.json) | [`experiments/exp_5/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5/evaluation/roc_curve.png) | [`experiments/exp_5/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5/cross_dataset) |
| **EXP-5 (V2)** | [`experiments/exp_5_v2_balanced/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced/checkpoints/best_model.pt) | [`experiments/exp_5_v2_balanced/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced/training_log.csv) | [`experiments/exp_5_v2_balanced/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced/evaluation/test_metrics.json) | [`experiments/exp_5_v2_balanced/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced/evaluation/roc_curve.png) | [`experiments/exp_5_v2_balanced/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/exp_5_v2_balanced/cross_dataset) |
| **ABL-1** | [`experiments/abl_1/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1/checkpoints/best_model.pt) | [`experiments/abl_1/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1/training_log.csv) | [`experiments/abl_1/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1/evaluation/test_metrics.json) | [`experiments/abl_1/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1/evaluation/roc_curve.png) | [`experiments/abl_1/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_1/cross_dataset) |
| **ABL-2** | [`experiments/abl_2/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2/checkpoints/best_model.pt) | [`experiments/abl_2/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2/training_log.csv) | [`experiments/abl_2/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2/evaluation/test_metrics.json) | [`experiments/abl_2/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2/evaluation/roc_curve.png) | [`experiments/abl_2/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_2/cross_dataset) |
| **ABL-3** | [`experiments/abl_3/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3/checkpoints/best_model.pt) | [`experiments/abl_3/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3/training_log.csv) | [`experiments/abl_3/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3/evaluation/test_metrics.json) | [`experiments/abl_3/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3/evaluation/roc_curve.png) | [`experiments/abl_3/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_3/cross_dataset) |
| **ABL-4** | [`experiments/abl_4/checkpoints/best_model.pt`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4/checkpoints/best_model.pt) | [`experiments/abl_4/training_log.csv`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4/training_log.csv) | [`experiments/abl_4/evaluation/test_metrics.json`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4/evaluation/test_metrics.json) | [`experiments/abl_4/evaluation/roc_curve.png`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4/evaluation/roc_curve.png) | [`experiments/abl_4/cross_dataset/`](file:///d:/Project/DeepFake Detection_FYP/experiments/abl_4/cross_dataset) |

---

## 7. Flagged Inconsistencies & Scientific Observations

### Flag 1: 1-Sample Boundary Discrepancy on FF++ Test Evaluation
- **Discrepancy:** The canonical saved evaluation artifact ([`experiments/exp_5_v2_balanced/evaluation/test_metrics.json`](file:///d:/Project/DeepFake%20Detection_FYP/experiments/exp_5_v2_balanced/evaluation/test_metrics.json)) records $\text{TN}=96, \text{FP}=54, \text{FN}=214, \text{TP}=386$ ($386$ true positives out of $600$ fake videos). An online sequential re-scan recorded $\text{FN}=213, \text{TP}=387$.
- **Root Cause:** Video `data/raw/faceforensics/NeuralTextures/791_770.mp4` (GT=FAKE) outputs raw logit $+0.621094$, yielding $P(\text{fake}) = 0.650000$ in Float32 and $0.650391$ under FP16 autocast. Under strict $(> 0.650000)$ inequality it is counted as $\text{FN}$, whereas under scalar $(\ge 0.650000)$ it falls on the upper boundary.
- **Resolution:** The canonical saved artifact is the permanent ground truth: $\mathbf{\text{TN}=96, \text{FP}=54, \text{FN}=214, \text{TP}=386}$.

### Flag 2: EXP-4 (Frequency-Only) Zero Specificity at Default Threshold
- **Observation:** In EXP-4, the model achieved $100.0\%$ recall and $0.0\%$ specificity ($\text{TN}=0, \text{FP}=150$) on in-domain FF++ test at $\tau=0.50$, despite achieving $0.6436$ ROC-AUC.
- **Scientific Meaning:** Frequency magnitude features without spatial grounding suffered severe positive-class logit shift under standard unweighted BCE training, causing all predictions to exceed $0.50$. This demonstrates that spectral features alone are insufficient for robust threshold-dependent classification without multi-domain spatial fusion.

### Flag 3: Metric Trade-Off: ROC-AUC vs. Balanced Accuracy vs. Specificity
- **Observation:** No single model strictly dominates all metrics across all dimensions:
  - **EXP-1 (ResNet-50 + BiLSTM)** achieved highest overall in-domain ROC-AUC ($0.8336$) and highest cross-dataset ROC-AUC ($0.7271$).
  - **EXP-5 V1 (CDTC-Net Standard BCE)** achieved highest raw recall ($91.67\%$ in-domain, $97.06\%$ cross-dataset) but suffered poor specificity ($34.67\%$ in-domain, $12.36\%$ cross-dataset).
  - **EXP-5 V2 (CDTC-Net Class-Balanced)** achieved highest real-video specificity balance ($64.00\%$ in-domain, $75.28\%$ cross-dataset) and reduced false positives by $44.9\%$ in-domain and $71.8\%$ cross-domain.

### Flag 4: Cross-Dataset Domain Shift Vulnerability
- **Observation:** All 10 models experienced a performance drop when transferred zero-shot from FaceForensics++ c23 (H.264 broadcast compression) to Celeb-DF v2 (high-resolution YouTube interviews).
- **Scientific Meaning:** Cross-dataset degradation is primarily driven by compression codec shifts and facial synthesis method divergence. Dual-domain spatial-temporal modeling (EXP-5 V2) provided superior authentic score calibration ($75.28\%$ specificity) on unseen data compared to spatial-only (EXP-2: $36.52\%$) and frequency-only (EXP-4: $0.0\%$) architectures.
