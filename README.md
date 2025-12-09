# Higgs Boson Classification: DNN vs XGBoost Benchmark & SHAP Analysis

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange)
![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green)
![SHAP](https://img.shields.io/badge/SHAP-Explainability-red)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

## 🔬 Project Overview
This project aims to reproduce and validate the application of Deep Learning in High Energy Physics (HEP), specifically for the classification of Higgs Boson decay events vs. Background noise.

Inspired by the seminal paper **"Searching for Exotic Particles in High-Energy Physics with Deep Learning" (Baldi et al., 2014)**, this repository benchmarks a **Deep Neural Network (DNN)** against the industry-standard **XGBoost (BDT)**.

Beyond raw performance metrics, this project focuses on **Explainable AI (XAI)**. Using SHAP values, we open the "black box" to verify if the model's decisions align with the Standard Model of particle physics.

## 📂 Dataset
The data is sourced from the **UCI Machine Learning Repository**:
* **Dataset:** HIGGS
* **Source:** [UCI Repository Link](https://archive.ics.uci.edu/ml/datasets/HIGGS)
* **Size:** 11 million events (sampled 1M for this study).
* **Features:** 28 kinematic features (21 low-level detector hits + 7 high-level physics derived features).

## 🛠️ Methodology

### 1. Preprocessing
* Stratified sampling to maintain signal/background ratio.
* Standard Scaling ($z = (x - \mu)/\sigma$) applied to all kinematic variables to ensure DNN convergence.

### 2. Models Architecture
* **Baseline (XGBoost):** Gradient Boosted Decision Trees, traditionally the gold standard in HEP analysis.
* **Deep Neural Network (DNN):**
    * Input Layer: 28 features.
    * Hidden Layers: 4 dense layers (300 neurons each) with ReLU activation.
    * Regularization: Batch Normalization + Dropout (0.3).
    * Optimizer: Adam with Learning Rate Decay.

## 🏆 Performance Results

| Model | AUC-ROC Score |
| :--- | :--- |
| XGBoost (Baseline) | 0.8213 |
| **Deep Neural Network (Ours)** | **0.8505** |

<p float="left">
  <img src="images/roc_curve.png" width="45%" />
  <img src="images/score_dist.png" width="45%" />
</p>

The DNN demonstrates superior signal efficiency for a given background rejection rate compared to the BDT baseline.

## ⚛️ Physical Interpretation & Discussion
A critical aspect of applying AI to physics is ensuring the model relies on physical phenomena rather than artifacts. We employed **SHAP (SHapley Additive exPlanations)** to rank feature importance.

<div align="center">
  <img src="images/shap_summary.png" width="80%" />
</div>

### Analysis
The analysis confirms the model's physical consistency:
1.  **Invariant Mass Dominance:** The features $m_{bb}$ (mass of bottom quark pair) and $m_{wwbb}$ emerged as the top predictors. This aligns with theoretical expectations, as invariant mass is the primary kinematic handle for identifying resonance decays ($H \to b\bar{b}$).
2.  **Model Validation:** The network autonomously "rediscovered" high-level physics concepts from the training data without being explicitly programmed with the Standard Model equations.

## 🚀 How to Run
1.  Clone this repository:
    ```bash
    git clone [https://github.com/SEU_USUARIO/higgs-boson-ml.git](https://github.com/SEU_USUARIO/higgs-boson-ml.git)
    ```
2.  Install dependencies:
    ```bash
    pip install -r requirements.txt
    ```
3.  Run the Jupyter Notebook in `notebooks/`.

## 📚 References & Inspiration
This work is a reproduction study based on the dataset and concepts presented in:

> **[1] Baldi, P., Sadowski, P., & Whiteson, D. (2014). "Searching for exotic particles in high-energy physics with deep learning." *Nature Communications*, 5, 4308.**

Additional references used for XAI validation:
> [2] Lundberg, S. M., & Lee, S. (2017). "A unified approach to interpreting model predictions." *NeurIPS*.

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
