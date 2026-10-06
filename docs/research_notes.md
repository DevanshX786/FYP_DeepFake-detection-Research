# Research Notes: DeepFake Detection FYP

## Architecture Notes
- **Working Title:** CDTC-Net (Cross-Domain Temporal Consistency Network)
- **Input:** $N=16$ uniformly sampled frames per video (reproducible seed).
- **Face Processing:** Standardized face crop (expanded bounding box, $224 \times 224$).
- **RGB Backbone:** ConvNeXt-Tiny (ImageNet pretrained, feature projected to 256-D).
- **Frequency Branch:** 2D FFT / DCT log-magnitude spectrum map ($\log(1 + |\mathcal{F}(x)|)$) passed to a 4-stage convolutional encoder (256-D).
- **Cross-Domain Fusion:** Concatenation (512-D) $\to$ Linear(512, 256) $\to$ LayerNorm $\to$ GELU $\to$ Dropout(0.2).
- **Temporal Difference Module:** Computes signed difference $D_t = F_{t+1} - F_t$ and absolute difference $A_t = |F_{t+1} - F_t|$, projected to 256-D.
- **Temporal Transformer:** 2-layer Transformer encoder (8 heads, 512 feedforward dim, 256 embed dim) with learnable `[CLS]` token and positional embeddings.
- **Classification Head:** Linear(256 $\to$ 128) $\to$ GELU $\to$ Dropout(0.3) $\to$ Linear(128 $\to$ 1) with BCEWithLogitsLoss.

## Data Leakage Prevention Guarantees
1. All splits are generated strictly at the source video / identity level.
2. Manipulations derived from the same original sequence (DeepFakes, Face2Face, FaceSwap, NeuralTextures) reside entirely in the same split as their source sequence.
3. Celeb-DF is retained purely for out-of-domain evaluation without any fine-tuning in primary benchmark reporting.
4. Preprocessing normalization and statistics are computed independently per sample or derived solely from training sets.

## Baseline Specifications
1. **Reference Project Baseline:** Independent implementation of YOLOv8 / face detector + ResNet50 (2048-D) + 1-layer BiLSTM (256 units/dir) + Linear classifier.
2. **Spatial-Only Baseline:** ConvNeXt-Tiny + Temporal Average Pooling + Linear classifier.
3. **RGB + Temporal Transformer Baseline:** ConvNeXt-Tiny + Temporal Transformer + Linear classifier.
4. **Frequency-Only Baseline:** FFT/DCT + Frequency CNN + Temporal Average Pooling + Linear classifier.
