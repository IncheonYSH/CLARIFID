<h1 align="center">CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings</h1>
<p align="center">
<a href="https://doi.org/10.1016/j.eswa.2025.130633"><img alt="Static Badge" src="https://img.shields.io/badge/Published%20in-ESWA-blue?style=flat-square&&link=10.1016%2Fj.eswa.2025.130633"></a>&nbsp;&nbsp;&nbsp;&nbsp;
    <a href="https://arxiv.org/abs/2507.17234"><img alt="Static Badge" src="https://img.shields.io/badge/Preprint-arXiv-%23B31B1B?style=flat-square&link=https%3A%2F%2Farxiv.org%2Fabs%2F2507.17234"></a></p>



---
Official implementation for our paper:  
**"CLARIFID: Improving Radiology Report Generation by Reinforcing Clinically Accurate Impressions and Enforcing Detailed Findings"**

> The paper is currently in pre-publication proofreading. We will update the paper link once the final version is officially released.

CLARIFID enhances radiology report generation by explicitly modeling the expert workflow from Findings to Impression. The framework employs section-aware pretraining, CheXbert-guided PPO fine-tuning, and multi-view fusion to produce clinically accurate and coherent reports. 

DOI: [10.1016/j.eswa.2025.130633](https://doi.org/10.1016/j.eswa.2025.130633)

---

## Overview

This repository provides the implementation for our radiology report generation model.  
The workflow consists of:

1. **Dataset preprocessing** (`dataset_preprocessing.py`)
2. **Pretraining** (`pretrain.py`)
3. **Post-training** (`posttrain.py`)
4. **Evaluation** (`evaluate.py`)

---

## Requirements

You can install the required packages using:

```bash
pip install -r requirements.txt
```

### Datasets

Please download the following datasets:

- MIMIC-CXR
- IU X-Ray

### Preprocessed Data File

The preprocessed CSV file is listed below.  
Due to the MIMIC-CXR data usage license, we cannot release the processed files for MIMIC-CXR.

* [IU X-Ray](https://drive.google.com/file/d/1OL11Y2HBjuQZmZE7hqH3pecK1aeLHjkB/view?usp=sharing)

### Checkpoints

* [Pretrained](https://drive.google.com/file/d/1uDa0rjD3ZFNKJZ4WA8AejkpLADxQVKTa/view?usp=sharing)
* [+Post-trained](https://drive.google.com/file/d/1Y-_oezAjwPLUh2IC4T72iREKqBc0GmlO/view?usp=sharing)

---

## Additional Setup

### Evaluation Tool

* Clone and install `pycocoevalcap` for metric evaluation.

* To compute the CE metric, please download the [CheXbert](https://github.com/stanfordmlgroup/CheXbert) checkpoint.

---

## Cite

If you find our work useful in your projects, please consider citing our paper:

```json
The paper is currently in pre-publication proofreading. We will update the BibTeX once the final version is officially released.
```



