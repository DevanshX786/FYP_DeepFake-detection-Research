# Literature Review & Research Verification: Deepfake Video Detection

**Project:** Cross-Domain Temporal Consistency Network (CDTC-Net) for Deepfake Video Detection  
**Author:** FYP Research Team  
**Date:** 2026-10-04  
**Status:** Phase 2 Literature Review & Research Gap Verification  

---

## 1. Executive Summary & Research Gap Verification

The objective of this research is to investigate whether explicitly combining **spatial facial representations (RGB)**, **frequency-domain artifact representations (FFT/DCT)**, **short-term temporal difference modeling**, and **long-range temporal consistency via a Transformer encoder** provides superior detection accuracy and cross-dataset generalization compared to conventional CNN-recurrent architectures (such as the senior project baseline ResNet50 + BiLSTM).

### Key Findings:
1. **Spatial CNN / Modern Backbone Detectors:** While early methods (MesoNet, Xception, EfficientNet) established strong in-domain results on FaceForensics++, modern ConvNeXt architectures capture richer hierarchical representations but remain vulnerable to unseen synthesis artifacts when used in isolation.
2. **Frequency-Domain Forensics:** Seminal works (Frank et al., 2020; Qian et al. *F3-Net*, 2020) demonstrated that upsampling and blending artifacts in GAN/diffusion generators introduce characteristic spectrum anomalies in the 2D Discrete Fourier / Cosine Transform. However, frequency features alone lack structural semantics and degrade under heavy lossy compression (e.g., c40).
3. **Temporal Inconsistency & Recurrent Limitations:** Recurrent networks (LSTMs / BiLSTMs) suffer from vanishing temporal gradients and sequential bias, failing to attend symmetrically across distant frames. Explicit frame-to-frame difference vectors ($\Delta F_t$) isolate subtle blending flickers and boundary jitter.
4. **Research Distinctiveness:** While individual elements (dual-stream RGB-frequency, or Transformer sequence modeling) exist in literature, **CDTC-Net** introduces a clean, unified formulation combining:
   - **Modern ConvNeXt-Tiny spatial encoder** + **Log-magnitude FFT/DCT lightweight encoder**,
   - **Cross-domain projection fusion**,
   - **Explicit differential motion tokens ($D_t$ and $|D_t|$)**, and
   - **A lightweight Temporal Consistency Transformer with a `[CLS]` token**, evaluated systematically under rigorous zero-shot cross-dataset transfer ($\text{FF++} \to \text{Celeb-DF}$).

---

## 2. Taxonomy of Deepfake Detection Approaches

| Category | Representative Methods | Key Strengths | Fundamental Limitations |
|---|---|---|---|
| **Frame-Level Spatial CNNs** | MesoNet (Afchar et al., 2018), Xception (Rossler et al., 2019), EfficientNet (Tan & Le), ConvNeXt (Liu et al., 2022) | High single-frame accuracy, strong feature extractors | Overfit to generator-specific textures; ignore temporal dynamics |
| **Frequency-Domain Analysis** | F3-Net (Qian et al., 2020), Freq-Detect (Frank et al., 2020), SRM-based forensics | Detects upsampling grid artifacts invisible in spatial RGB | Sensitive to video compression codecs (H.264/H.265 at high QP) |
| **Dual-Stream Spatial-Frequency** | SPSL (Liu et al., 2021), MFE (Wang et al., 2021) | Fuses complementary texture and spectral clues | Often implemented with heavy static backbones; lacks sequential dynamics |
| **Recurrent Temporal Forensics** | ResNet50 + BiLSTM (Senior Baseline, 2024), Guera & Delp (2018) | Captures temporal transitions | BiLSTM lacks global attention; prone to forgetting; expensive recurrence |
| **Transformer / Attention Forensics** | FTCN (Zheng et al., 2021), ICT (Dong et al., 2022), LipForensics (Haliassos et al., 2021) | Global temporal attention, robust sequence representations | High computational demand; often neglects explicit spectral cues |
| **Proposed CDTC-Net** | ConvNeXt-Tiny + FFT/DCT + Temporal Difference + Transformer | Multimodal (spatial + spectral), explicit difference features, global temporal self-attention | Requires synchronized two-branch frame extraction |

---

## 3. Detailed Component Analysis

### 3.1 RGB Spatial Encoder (ConvNeXt-Tiny)
ConvNeXt modernizes standard convolutional networks by incorporating Vision Transformer design choices (7x7 depthwise convolutions, inverted bottlenecks, larger receptive fields, LayerNorm, and GELU). It provides superior feature quality and robustness over standard ResNet50 while maintaining computational efficiency and standard convolutional inductive biases.

### 3.2 Frequency-Domain Branch (2D FFT / DCT)
Manipulated facial regions typically undergo face swapping or reenactment followed by affine warping, boundary feathering, and Poisson blending. These operations disrupt high-frequency energy distribution:
$$\mathcal{F}(u, v) = \log(1 + |\text{FFT2D}(I_{gray})|)$$
Feeding log-magnitude spectrum maps into a dedicated lightweight convolutional encoder extracts spectral anomaly fingerprints that RGB backbones overlook.

### 3.3 Explicit Temporal Difference Modeling
Frame differences highlight inter-frame motion inconsistencies and landmark jitter:
$$D_t = F_{t+1} - F_t, \quad A_t = |F_{t+1} - F_t|$$
$$\tilde{D}_t = \text{MLP}([D_t; A_t])$$
By converting inter-frame feature variations into explicit tokens, the downstream sequence model does not need to learn subtraction operations implicitly.

### 3.4 Temporal Consistency Transformer
Rather than a unidirectional or bidirectional recurrent cell, a multi-head self-attention Transformer encoder processes the full token sequence (frame tokens + difference tokens + `[CLS]` token):
$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V$$
This enables arbitrary pairwise frame comparisons, directly capturing non-local inconsistencies (such as periodic blinking anomalies, lighting phase mismatches, and intermittent synthesis dropouts).

---

## 4. Benchmark Datasets & Generalization Challenge

### 4.1 FaceForensics++ (FF++)
- **Description:** 1,000 pristine YouTube video sequences with corresponding manipulated sequences generated via 4 techniques:
  - *DeepFakes* (autoencoder-based face replacement)
  - *Face2Face* (graphics-based facial reenactment)
  - *FaceSwap* (graphics-based 3D model face replacement)
  - *NeuralTextures* (neural rendering with learned texture maps)
- **Compression levels:** Raw (c0), High Quality (c23), Low Quality (c40).
- **Role in FYP:** Primary development, training, and in-domain evaluation benchmark.

### 4.2 Celeb-DF v2
- **Description:** 590 original celebrity videos and 5,639 high-quality DeepFake videos created using an improved synthesis algorithm specifically reducing visual artifacts and boundary anomalies.
- **Role in FYP:** Unseen out-of-domain evaluation benchmark. Models trained purely on FF++ are tested zero-shot on Celeb-DF to quantify true domain generalization.

---

## 5. Formal Research Questions Addressed

- **RQ1:** Does integrating log-magnitude frequency representations provide statistically significant improvements over spatial-only models?
- **RQ2:** Does explicit feature-space frame differencing ($D_t, A_t$) improve sensitivity to temporal synthesis artifacts?
- **RQ3:** Does cross-domain feature fusion preserve complementary discriminative cues without mutual interference?
- **RQ4:** Does a lightweight Transformer encoder outperform recurrent BiLSTM modeling in sequence classification?
- **RQ5:** Does CDTC-Net achieve superior cross-dataset generalization ($\text{FF++} \to \text{Celeb-DF}$) compared to the senior ResNet50 + BiLSTM baseline?
- **RQ6:** How resilient is the frequency-spatial representation under compression degradation (c23 vs c40)?

---

## 6. Conclusion & Novelty Justification

CDTC-Net avoids incremental "stacking" by grounding every architectural component in a specific physical artifact of the deepfake generation pipeline (spatial blending artifacts, spectral synthesis signatures, and frame-to-frame warping inconsistencies). The experimental framework includes thorough ablation testing (A0 through A4) and strict identity-level split guarantees, ensuring genuine scientific validity suitable for an academic research paper.
