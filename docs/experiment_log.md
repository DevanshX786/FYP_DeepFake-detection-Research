# Experiment Log: DeepFake Detection FYP

| Experiment ID | Date | Model | Dataset / Split | Epochs | Batch Size | LR | In-Domain AUC | In-Domain Acc | In-Domain F1 | Celeb-DF AUC | Status / Notes |
|---|---|---|---|---|---|---|---|---|---|---|---|
| *exp_000_pipeline_sanity* | 2026-10-04 | CDTC-Net (Dummy) | Synthetic Sanity Check | 1 | 2 | 1e-4 | N/A | N/A | N/A | N/A | Pipeline & tensor shape verification |
| *exp_000b_phase7_sanity* | 2026-10-04 | CDTC-Net (Full) | FF++ Subset (50 train, 25 val) | 4 | 4 | 1e-4 | 0.3400 | 80.00% | 0.8889 | Untouched | Gradient stability & VRAM audit |
| *exp_001_senior_baseline* | TBD | ResNet50-BiLSTM | FF++ (c23) | 30 | Auto | 1e-4 | - | - | - | - | Pending full training launch |
| *exp_002_spatial_convnext* | TBD | RGB ConvNeXt-Tiny | FF++ (c23) | 30 | Auto | 1e-4 | - | - | - | - | Pending full training launch |
| *exp_003_rgb_transformer* | TBD | RGB + Transformer | FF++ (c23) | 30 | Auto | 1e-4 | - | - | - | - | Pending full training launch |
| *exp_004_freq_only* | TBD | Frequency CNN | FF++ (c23) | 30 | Auto | 1e-4 | - | - | - | - | Pending full training launch |
| *exp_005_cdtc_full* | TBD | CDTC-Net (Full) | FF++ (c23) | 30 | Auto | 1e-4 | - | - | - | - | Primary proposed model experiment |
| *exp_006_cross_celebdf* | TBD | CDTC-Net vs Baselines | Celeb-DF v2 | - | - | - | - | - | - | - | Cross-dataset generalization test |
| *exp_007_c40_robustness* | TBD | CDTC-Net vs Baselines | FF++ (c40) | - | - | - | - | - | - | - | Robustness to heavy compression |

