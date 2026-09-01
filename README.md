# ECG-MIT BIH: Sub-50 KB Inter-Patient Arrhythmia Classification

A lightweight ECG arrhythmia classification study on the **MIT-BIH Arrhythmia Database**, using a strict inter-patient evaluation protocol, frequency-domain feature selection, sample-adaptive spectral attention, and INT8 ONNX deployment.

The main evaluation follows the **AAMI 5-class formulation** with a patient-independent **DS1 → DS2** split. The work also includes 3-class and 4-class analyses, cross-database evaluation, model-size comparisons, quantization analysis, robustness experiments, and ablation studies.

---

## Overview

Automated ECG beat classification is challenging not only because of the differences between arrhythmia classes, but also because of severe class imbalance and patient-to-patient variation.

This repository contains an end-to-end implementation and experimental results covering:

* ECG signal preprocessing and beat segmentation
* AAMI 5-class arrhythmia classification
* DCT-based spectral representation
* Frequency Discovery Network (FDN) for selecting 40 DCT coefficients
* Lightweight 1D-CNN classification
* Sample-Adaptive Spectral Attention (SASA)
* FP32 and INT8 ONNX deployment
* Strict inter-patient evaluation
* Cross-database evaluation
* 3-class and 4-class analyses
* Ablation and robustness experiments

The primary target is a deployable model with a footprint below **50 KB** while maintaining the evaluation protocol consistently across experiments.

---

## Dataset

The primary dataset is the **MIT-BIH Arrhythmia Database**.

### Signal configuration

* **Lead:** MLII
* **Sampling frequency:** 360 Hz
* **Beat-centered window:** 250 samples
* **Window:** 90 samples before and 160 samples after the annotated R-peak
* **Filtering:** 3rd-order Butterworth bandpass, 0.5–40 Hz
* **Power-line filtering:** 60 Hz notch filter, Q = 30

### AAMI 5-class mapping

| Class | Description                            |
| ----- | -------------------------------------- |
| **N** | Normal and related beats               |
| **S** | Supraventricular ectopic beats         |
| **V** | Ventricular ectopic beats              |
| **F** | Fusion beats                           |
| **Q** | Unknown / paced / unclassifiable beats |

The repository includes visualizations of individual N/S/V/F/Q beats and ECG preprocessing stages.

![AAMI 5-Class Beat Visualization](figures/signal_processing/single_beat_NSVFQ_visualization.png)

---

## Signal Processing

The ECG recordings are processed before classification to obtain beat-centered representations.

A representative signal-processing visualization is provided below.

![ECG Signal Processing](figures/signal_processing/raw_ecg_vs_dct_bandpass.png)

Additional signal-processing figures are available in [`figures/signal_processing`](figures/signal_processing/), including:

* Full patient ECG signal
* Heartbeat segmentation
* Raw ECG vs. filtered/DCT representation
* Individual N/S/V/F/Q beat visualization

![Heartbeat Segmentation](figures/signal_processing/best_heartbeat_segmentation.png)

---

## Evaluation Protocol

The primary experiment follows a strict **inter-patient** evaluation protocol.

```text
MIT-BIH Arrhythmia Database
            │
            ├── DS1
            │    └── Training / Validation
            │
            └── DS2
                 └── Held-out Testing
```

The split is based on the de Chazal inter-patient protocol, keeping the test patients separated from the training/validation patients.

### Primary evaluation

* **Training/validation:** DS1
* **Testing:** DS2
* **Classes:** AAMI 5 classes
* **Metric:** Accuracy and Macro-F1
* **Repeated experiments:** 5 seeds for the primary reported results

Macro-F1 is particularly important here because the AAMI classes are highly imbalanced. Accuracy alone can obscure poor performance on minority classes.

---

## Class Imbalance and Augmentation

The training procedure accounts for the severe imbalance between AAMI classes.

### Class weighting

Inverse-square-root-of-frequency weighting is used to reduce the dominance of majority classes.

### Loss

The classification models use **Focal Loss** together with class weighting.

### Data augmentation

No SMOTE or synthetic interpolation is used.

Instead, augmentation is performed by creating noisy duplicates of real minority beats using Gaussian noise.

This keeps the augmented samples tied to observed ECG beats rather than generating interpolated samples between different beats.

### Leakage control

Patient-dependent information is handled within individual records. Features involving patient-specific normalization, such as RR-related features, are computed without using information from other patients.

---

# Frequency-Domain Representation

Instead of feeding the complete time-domain beat directly into a larger network, the experiments use the **Discrete Cosine Transform (DCT)** as a compact spectral representation.

The resulting DCT representation contains 61 coefficients.

The project then investigates whether a smaller subset can retain useful classification information.

---



## Frequency Discovery Network — 40 DCT Coefficients

The **Frequency Discovery Network (FDN)** is used to identify a compact set of DCT coefficients.

The selected representation contains **40 DCT coefficients** rather than using all available coefficients.

The selected indices are:

```text
[0, 1, 2, 3, 4, 5, 6, 7, 8, 9,
 10, 12, 13, 15, 16, 17, 18, 19, 21, 22,
 23, 26, 27, 28, 29, 30, 31, 32, 33, 34,
 35, 40, 41, 42, 45, 50, 52, 55, 58, 60]
```

The FDN-selected coefficients are compared against a naive fixed selection of the first 40 DCT bins.

![FDN vs Naive 40 DCT Selection](figures/feature_discovery/fdn_discovery_vs_naive_40.png)

The resulting 40-coefficient representation is used by the lightweight models described below.

---

## Derived Feature Dataset

For convenient access to the extracted 40-DCT representation, the features
selected through FDN were exported as a derived dataset on Kaggle.

📊 **Kaggle Dataset:**  
[MIT-BIH FDN-40DCT](https://www.kaggle.com/datasets/mahmudul28/mit-bih-fdn-40dct)

The dataset contains three exported CSV files:

| File | Description |
|---|---|
| `ds1_frozen_40dct.csv` | 40-DCT features exported from DS1 |
| `ds2_frozen_40dct.csv` | 40-DCT features exported from DS2 |
| `ds1_ds2_frozen_40dct.csv` | Combined DS1 and DS2 40-DCT features |

These files are provided as **derived feature data for analysis and reuse**.
They are not the original MIT-BIH recordings.

# Models

Three main configurations are evaluated.

## 1. Baseline

The baseline uses a 1D-CNN classification backbone with a fixed selection of the first 40 DCT bins.

This provides a reference for evaluating the effect of the frequency-selection procedure.

---

## 2. LDN — Lightweight Deployment Network

LDN uses the 40 DCT coefficients selected through FDN.

The model is designed around the compact spectral representation and is subsequently exported and quantized for small-footprint deployment.

---

## 3. SASA-Net

SASA-Net extends the compact spectral representation with **Sample-Adaptive Spectral Attention**.

Instead of applying exactly the same spectral weighting to every beat, the attention gate produces a beat-dependent weighting of the selected spectral features.

RR-related information is used to condition the attention mechanism.

Conceptually:

```text
40 FDN-selected DCT coefficients
                │
                ├───────────────┐
                │               │
                │          RR features
                │               │
                │               ▼
                │       Spectral attention
                │               │
                └───────► Adaptive weighting
                                │
                                ▼
                           Classifier
```

The model therefore retains the compact 40-coefficient representation while allowing the spectral weighting to vary between beats.

---

# Main Results — AAMI 5-Class

The primary results use the strict **DS1 → DS2 inter-patient protocol**.

### Five-seed results

| Model                |        Accuracy |            Macro-F1 |
| -------------------- | --------------: | ------------------: |
| Baseline             |          0.8700 |     0.3760 ± 0.0446 |
| LDN (FP32)           | 0.8572 ± 0.0156 | **0.4069 ± 0.0287** |
| SASA-Net (FP32)      | 0.8210 ± 0.0504 |     0.3910 ± 0.0251 |
| LDN (INT8 ONNX)      | 0.8592 ± 0.0159 | **0.4079 ± 0.0290** |
| SASA-Net (INT8 ONNX) | 0.8199 ± 0.0519 |     0.3928 ± 0.0259 |

For the primary 5-class task, **LDN obtains the highest Macro-F1 among the listed configurations**, while SASA-Net provides the sample-adaptive spectral attention configuration.

The INT8 LDN result is close to its FP32 counterpart in the reported evaluation.

---

## 5-Class Confusion Matrix

The representative confusion matrices show the class-wise behavior of the models, including the difficulty of distinguishing minority classes under the imbalanced AAMI setting.

![5-Class Confusion Matrix](figures/classification/confusion_matrix_5class.png)

A direct LDN/SASA-Net comparison is also provided:

![LDN vs SASA-Net Confusion Matrices](figures/model_comparison/ldn_vs_sasa_confusion_matrices.png)

The repository also contains a combined comparison of multiple model configurations:

![Four Model Confusion Matrices](figures/model_comparison/four_model_confusion_matrices.png)

---

# Model Size and Deployment

A major part of the experiments focuses on reducing the deployed model footprint.

The trained models are exported to **ONNX** and quantized to **INT8**.

### Reported deployment sizes

| Model    | Deployment Format |         Size |
| -------- | ----------------- | -----------: |
| LDN      | INT8 ONNX         | **44.30 KB** |
| SASA-Net | INT8 ONNX         | **48.26 KB** |

Both configurations remain below the **50 KB** target.

![Model Size Comparison](figures/model_comparison/model_size_comparison.png)

The quantization directory also contains the FP32/INT8 model-size comparison.

![Quantized Model Size Comparison](figures/quantization/model_size_comparison.png)

---

## FP32 vs INT8

Quantization is evaluated beyond file size by comparing the classification behavior of the FP32 and INT8 models.

![FP32 vs INT8 Confusion Matrix](figures/quantization/fp32_vs_int8_confusion_matrix.png)

An additional INT8 confusion matrix is available here:

![INT8 Confusion Matrix](figures/quantization/int8_confusion_matrix_counts.png)

The preprocessing normalization is also folded directly into the inference graph, avoiding an additional runtime normalization operation.

---

# 3-Class Analysis

Additional experiments collapse the classification problem into three classes.

| Model    | Accuracy |   Macro-F1 |
| -------- | -------: | ---------: |
| Baseline |   0.9010 |     0.5990 |
| LDN      |   0.9080 |     0.6486 |
| SASA-Net |   0.9000 | **0.6487** |

![3-Class Confusion Matrix](figures/classification/confusion_matrix_3class.png)

The 3-class experiment provides a less granular classification setting in which the models can be compared separately from the primary 5-class evaluation.

---

# 4-Class Analysis

The 4-class configuration provides an intermediate classification setting.

| Model    |   Accuracy |   Macro-F1 |
| -------- | ---------: | ---------: |
| Baseline |     0.8950 |     0.4248 |
| LDN      |     0.9070 |     0.4736 |
| SASA-Net | **0.9180** | **0.4951** |

![4-Class Confusion Matrix](figures/classification/confusion_matrix_4class.png)

The repository stores the corresponding 3-class, 4-class, and 5-class numerical results under [`results/`](results/).

---

# Cross-Database Evaluation

The models are also evaluated outside the original MIT-BIH test database.

The cross-database experiments include:

* **INCART**
* **European ST-T Database (EDB)**

These experiments are intended to examine how the models behave when the evaluation data come from a different database distribution.

## INCART

| Model                |   Accuracy | Macro-F1 |
| -------------------- | ---------: | -------: |
| LDN (FP32)           | **94.26%** |   0.4334 |
| SASA-Net (FP32)      |     88.02% |   0.4284 |
| SASA-Net (INT8 ONNX) |     80.25% |   0.3802 |

![Cross-Database Macro-F1](figures/cross_database/cross_database_macro_f1.png)

![Per-Class INCART F1](figures/cross_database/per_class_f1_incart.png)

![Classwise Cross-Database Generalization](figures/cross_database/classwise_cross_database_generalization.png)

## European ST-T Database

| Model                | Accuracy | Macro-F1 |
| -------------------- | -------: | -------: |
| LDN (FP32)           |   72.60% |   0.2055 |
| SASA-Net (FP32)      |   80.55% |   0.1949 |
| SASA-Net (INT8 ONNX) |   80.65% |   0.1980 |

The European ST-T experiment was performed using inferior limb leads. Minority-class performance is strongly affected by the extreme class imbalance in this evaluation.

---

# Multi-Seed Evaluation

Because a single training seed can give an incomplete picture of model behavior, the primary experiments include multiple seeds.

![Multi-Seed Performance](figures/model_comparison/multi_seed_performance.png)

The reported primary 5-class values are presented as mean ± standard deviation where applicable.

---

# Robustness Analysis

Additional evaluation examines model behavior beyond overall accuracy and Macro-F1.

The repository includes:

![ROC and Precision-Recall Curves](figures/robustness/ROC_PR.png)

and:

![Per-Patient Performance](figures/robustness/per_patient_performance.png)

These analyses provide additional views of classification behavior across patients and operating conditions.

---

# Ablation and Previous Configurations

Several configurations were evaluated during development.

| Configuration           | Best Macro-F1 / Result | Observation                                                                                      |
| ----------------------- | ---------------------: | ------------------------------------------------------------------------------------------------ |
| VDN (Variance-Ranked)   |                 0.4160 | Selected top-40 bins by variance; only 18/40 coefficients overlapped with FDN                    |
| FDN                     |                 0.3847 | Learnable frequency selection discovered a compact 40-bin representation                         |
| Extended Budget         |                 0.3852 | Increasing the representation to 49 bins did not provide a statistically significant improvement |
| Two-Stage Cascade       |                 0.3827 | Binary N-vs-Ectopic followed by a 4-way split provided no additive gain                          |
| SASA-Cascade            |                 0.3881 | Combining the cascade with SASA did not improve over the individual components                   |
| Contrastive Pretraining |                 0.1830 | Embedding space did not improve under the tested configuration                                   |
| PhySASA-Net             |                ~0.4200 | Over-regularized RR features affected F-class representation                                     |
| Deep SASA (SE-ResNet)   |                 0.3500 | Larger network overfit augmented minority samples                                                |
| Multi-Domain SASA       |                 0.3500 | Additional WPD features were redundant with DCT and caused ONNX issues                           |
| ResGate                 |                <0.4000 | Residual MLP gate smoothed the attention output and weakened threshold-like behavior             |
| Random Split            |                    N/A | Evaluated to demonstrate the effect of patient-level data leakage                                |

These experiments are retained to document what was tested and how the final configuration was selected.

---

# Comparison with Reported Results

The following table places the main 5-class results alongside selected reported configurations.

| Reference / Model                             | Accuracy | Macro-F1 | Model Size |
| --------------------------------------------- | -------: | -------: | ---------: |
| de Chazal et al. (2004) — Linear Discriminant |   86.24% |       NR |        N/A |
| Zhang et al. (2021) — Adversarial CNN         |       NR |       NR |         NR |
| Wu et al. (2022) — DNN Ensemble + Focal Loss  |   91.89% |   49.95% |         NR |
| Elgazzar (2026) — Lightweight 1D-CNN          |   82.81% |   38.23% |     1.5 MB |
| LDN                                           |   85.40% |   0.4069 |   44.26 KB |
| SASA-Net                                      |   85.00% |   0.3910 |   48.26 KB |

The comparison should be interpreted together with differences in dataset protocol, class definitions, evaluation splits, preprocessing, and model configuration. The repository's primary evaluation is based on the strict DS1 → DS2 inter-patient setting.

---

# Repository Structure

```text
SASA-Net-ECG/
│
├── data/
│   └── Processed datasets and feature representations
│
├── figures/
│   ├── classification/
│   │   ├── confusion_matrix_3class.png
│   │   ├── confusion_matrix_4class.png
│   │   └── confusion_matrix_5class.png
│   │
│   ├── cross_database/
│   │   ├── classwise_cross_database_generalization.png
│   │   ├── cross_database_macro_f1.png
│   │   └── per_class_f1_incart.png
│   │
│   ├── feature_discovery/
│   │   └── fdn_discovery_vs_naive_40.png
│   │
│   ├── model_comparison/
│   │   ├── four_model_confusion_matrices.png
│   │   ├── ldn_vs_sasa_confusion_matrices.png
│   │   ├── model_size_comparison.png
│   │   └── multi_seed_performance.png
│   │
│   ├── quantization/
│   │   ├── fp32_vs_int8_confusion_matrix.png
│   │   ├── int8_confusion_matrix_counts.png
│   │   └── model_size_comparison.png
│   │
│   ├── robustness/
│   │   ├── ROC_PR.png
│   │   └── per_patient_performance.png
│   │
│   └── signal_processing/
│       ├── best_heartbeat_segmentation.png
│       ├── one_patient_ecg_signal.png
│       ├── raw_ecg_vs_dct_bandpass.png
│       └── single_beat_NSVFQ_visualization.png
│
├── models/
│   └── Trained and exported model artifacts
│
├── results/
│   ├── 3_classes/
│   │   └── results.csv
│   ├── 4_classes/
│   │   └── results.csv
│   ├── 5_classes/
│   │   └── results.csv
│   └── cross_dataset/
│       ├── results.csv
│       └── New Microsoft Excel Worksheet.xlsx
│
├── LICENSE
└── README.md
```

---

# Key Experimental Settings

| Component               | Configuration                 |
| ----------------------- | ----------------------------- |
| Dataset                 | MIT-BIH Arrhythmia Database   |
| Lead                    | MLII                          |
| Sampling rate           | 360 Hz                        |
| Beat window             | 250 samples                   |
| Bandpass                | 0.5–40 Hz                     |
| Bandpass order          | 3                             |
| Notch                   | 60 Hz, Q=30                   |
| Classification          | AAMI 5-class                  |
| Primary split           | DS1 → DS2                     |
| Evaluation              | Inter-patient                 |
| Spectral representation | DCT                           |
| Selected coefficients   | 40                            |
| Frequency selection     | FDN                           |
| Loss                    | Focal Loss                    |
| Class weighting         | Inverse square-root frequency |
| SMOTE                   | Not used                      |
| Minority augmentation   | Gaussian-noise duplication    |
| Deployment              | ONNX                          |
| Quantization            | INT8                          |
| LDN size                | 44.30 KB                      |
| SASA-Net size           | 48.26 KB                      |

---

# Main Takeaways

* The primary evaluation uses a **strict patient-independent DS1 → DS2 split** rather than a random beat-level split.
* The main classification setting uses the **five AAMI classes: N, S, V, F, and Q**.
* A compact representation based on **40 DCT coefficients** is used for the lightweight models.
* FDN is used to select the 40 coefficients instead of simply taking the first 40 DCT bins.
* LDN achieves a reported **Macro-F1 of 0.4069 ± 0.0287** in FP32 on the primary 5-class evaluation.
* The INT8 LDN model achieves **0.4079 ± 0.0290 Macro-F1** in the reported evaluation.
* The INT8 models remain below the **50 KB** deployment target:

  * LDN: **44.30 KB**
  * SASA-Net: **48.26 KB**
* Additional experiments cover 3-class and 4-class classification, cross-database evaluation, robustness, quantization, and multiple ablations.
* Cross-database results show that performance varies substantially across databases and class distributions.

---

# Reproducibility

The repository is organized to keep the processed data, trained models, figures, and numerical results separated.

```text
data/      → processed data / feature representations
models/    → trained and exported models
results/   → numerical experiment results
figures/   → generated visualizations
```

The CSV result files under `results/` contain the numerical outputs for the corresponding classification settings, while the figures provide visual summaries of the experiments.

For exact training and evaluation commands, refer to the implementation and experiment files included in the repository.

---

# Limitations

The reported results should be interpreted within the scope of the evaluated datasets and protocols.

In particular:

* Performance varies between AAMI classes because of severe class imbalance.
* Macro-F1 remains substantially lower than overall Accuracy in the 5-class setting.
* Cross-database performance is not uniform across databases.
* The European ST-T evaluation shows reduced minority-class performance under extreme class imbalance and different lead characteristics.
* A model footprint below 50 KB does not by itself establish clinical suitability.
* The experiments are research evaluations and do not constitute clinical validation.

---

# Citation

If you use this repository or the associated experimental setup in your work, please cite the corresponding publication when available.

```bibtex
@misc{sasa_net_ecg,
  title  = {ECG-MIT BIH: Sub-50 KB Inter-Patient Arrhythmia Classification},
  author = {Mahmudul Hasan},
  year   = {2026},
  url    = {https://github.com/mahmudul28/SASA-Net-ECG}
}
```

---

# License

This repository is distributed under the **MIT License**. See [`LICENSE`](LICENSE) for details.
