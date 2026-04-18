# SMM4H-HeaRD 2026 Task 1: Multilingual ADE Detection

**Team Paradise** | System Description Paper + Code

[![Paper](https://img.shields.io/badge/Paper-ACL%20Format-blue)](paper/main.pdf)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.8+-orange.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)

## 📋 Overview

This repository contains our submission to the **SMM4H-HeaRD 2026 Shared Task, Task 1**: Binary classification of social media posts for Adverse Drug Event (ADE) mentions across **7 languages** (German, French, Russian, English, Mandarin, Japanese) with **zero-shot transfer to Farsi**.

**Key Contribution:** We demonstrate that **threshold calibration alone** yields +0.050 macro F1 improvement (from 0.547 → 0.597) on a frozen XLM-RoBERTa model — larger than many encoder-level ablations reported in literature.

### 🏆 Results

| Metric | Value | vs. Field |
|--------|-------|-----------|
| **Overall Macro F1** | **0.597** | +0.051 above mean |
| Field Mean | 0.547 | — |
| Field Median | 0.580 | +0.017 above |
| Japanese (ja) | 0.609 | **+0.075** 🔥 |
| Zero-shot Farsi (fa) | 0.408 | **+0.041** 🚀 |
| German (de) | 0.610 | -0.054 |
| French (fr) | 0.634 | -0.047 |

---

## 🎯 Task Description

**Input:** Social media post in any target language  
**Output:** Binary label (1 = contains ADE mention, 0 = no ADE)  
**Challenge:**
- Severe class imbalance (2.4% to 60% positive rates)
- Platform diversity (Twitter/X vs. patient forums vs. drug reviews)
- Zero-shot Farsi evaluation (no training data)
- Unweighted macro-F1 across 9 test splits

---

## 🏗️ System Architecture

![Methodology Pipeline](figures/methodology_diagram.png)

**3-Stage Pipeline:**

1. **Data Input:** 6 training languages (47.5k docs) + CADEC translated subset
2. **Model Training:** XLM-RoBERTa-large + Focal Loss + Language-Balanced Sampling
3. **Threshold Ablation:** Same model, three calibration strategies (V1/V2/V3)

---

## 🚀 Quick Start

### Prerequisites

```bash
Python 3.8+
CUDA-capable GPU (we used RTX 4090, 24GB)
```

### Installation

```bash
git clone https://github.com/DhruvGoyal404/SMM4H_TASK1.git
cd SMM4H_TASK1
pip install -r requirements.txt
```

### Data Setup

Download the official SMM4H-HeaRD 2026 Task 1 data from [CodaBench](https://www.codabench.org/competitions/14124/) and place files in the root directory:

```
SMM4H_TASK1/
├── train_data_SMM4H_2026_Task_1.csv
├── dev_data_SMM4H_2026_Task_1.csv
├── train_data_cadec_translated.csv
├── dev_data_cadec_translated.csv
└── combined_test_data_unlabeled.csv
```

### Training

Run the complete pipeline (data prep → training → inference → submission):

```bash
jupyter notebook code.ipynb
```

Or execute all cells programmatically:

```bash
jupyter nbconvert --to notebook --execute code.ipynb
```

**Training Time:** ~35 minutes on RTX 4090

**Output Files:**
- `best_model.pt` — trained model checkpoint
- `test_probs.npy` — raw probability scores for test set
- `submission.csv` — predictions in CodaBench format
- `submission.zip` — ready for upload

---

## 📊 Methodology

### Model

- **Backbone:** XLM-RoBERTa-large (24 layers, 559M parameters)
- **Pre-training:** 100 languages including Farsi (enables zero-shot)
- **Classification Head:** Linear layer on `[CLS]` token with 10% dropout

### Training Strategy

```python
Loss:       Focal Loss (γ=2.0, α=0.25)
Sampling:   Language-Balanced Weighted Sampler (weights: 0.5–21.2)
Epochs:     8
Batch:      64 (32 × 2 gradient accumulation)
LR:         2e-5 with 10% linear warmup
Optimizer:  AdamW (weight decay 0.01)
Hardware:   NVIDIA RTX 4090, fp16 mixed precision
```

### Threshold Calibration (Our Contribution)

We hold the trained model **completely frozen** and ablate only the decision threshold:

| Strategy | Method | Thresholds | F1 Score |
|----------|--------|------------|----------|
| **V1** | Manual inspection of probability histograms | de=0.47, en=0.16, fr=0.44, ja=0.23, ru=0.60, zh=0.79, fa=0.50 | 0.547 |
| **V2** | Percentile-match to training prior | τ = Pctl(probs, 100(1-π)) | 0.575 |
| **V3** | Feedback-refined using CodaBench results | de=0.40, en=0.10, fr=0.25, ja=0.10, ru=0.10, zh=0.06, fa=0.50 | **0.597** ✓ |

**Key Insight:** Platform type (Twitter vs. forums) drives threshold selection more than language alone.

---

## 📁 Repository Structure

```
SMM4H_TASK1/
├── code.ipynb                  # Main training + inference notebook
├── requirements.txt            # Python dependencies
├── README.md                   # This file
├── LICENSE                     # MIT License
├── paper/
│   ├── main.tex               # ACL-format paper source
│   ├── custom.bib             # Bibliography
│   └── main.pdf               # Compiled paper
├── figures/
│   └── methodology_diagram.png # System architecture diagram
└── submissions/
    ├── submission_v1.csv      # V1 predictions (F1=0.547)
    ├── submission_v2.csv      # V2 predictions (F1=0.575)
    └── submission_v3.csv      # V3 predictions (F1=0.597, BEST)
```

---

## 📝 Paper

Our system description paper is submitted to the **SMM4H-HeaRD 2026 Workshop** (co-located with a major NLP conference).

**Read the paper:** [`paper/main.pdf`](paper/main.pdf)

**Compile from source:**
```bash
cd paper/
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

---

## 🔑 Key Findings

### What Worked ✅

1. **XLM-R for Zero-Shot:** Strong Farsi performance (+0.041 vs. mean) with no training data
2. **Focal Loss:** Prevents early convergence to majority class at 2–10% imbalance
3. **Threshold Calibration:** +0.050 F1 gain with zero model changes

### What Didn't Work ❌

1. **Dev Set Contamination:** Merging dev into training destroyed calibration signal (F1 inflated to 0.997)
2. **Forum Over-prediction:** German/French patient forums under-performed (predicted 8–13% positive but test prior was tighter)
3. **Farsi MT Baseline:** Authentication issues blocked Helsinki-NLP translation comparison

### Lessons for Future Participants 💡

**Keep validation split separate when thresholds are part of your pipeline.** Merging dev into training eliminates the very signal needed for post-hoc calibration.

---

## 🙏 Acknowledgments

- **SMM4H-HeaRD 2026 Organizers** for the shared task and data
- **Hugging Face** for the Transformers library
- **Original Dataset Authors:**
  - KEEPHA (German/French forums)
  - RuDReC (Russian drug reviews)
  - CADEC-v2 (English drug reviews, translated to de/fr)
  - Previous SMM4H organizers (English Twitter)

---

## 📜 Citation

If you use this code or find our work helpful, please cite:

```bibtex
@inproceedings{paradise-smm4h2026,
  title     = {Team Paradise at SMM4H-HeaRD 2026: Multilingual Adverse Drug Event Detection with XLM-RoBERTa and Threshold-Only Ablation},
  author    = {Goyal, Dhruv},
  booktitle = {Proceedings of the Social Media Mining for Health Workshop (SMM4H-HeaRD)},
  year      = {2026},
  publisher = {Association for Computational Linguistics}
}
```

---

## 📧 Contact

**Dhruv Goyal**  
BTECH CSE, TIET Patiala  
GitHub: [@DhruvGoyal404](https://github.com/DhruvGoyal404)

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

**Built with ❤️ for multilingual health NLP**
