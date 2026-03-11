
# tsAMP: A Multi-level Framework for Antimicrobial Peptide Prediction

tsAMP is a novel predictive framework based on the **ESM-1v** protein language model. It is designed to provide comprehensive analysis across three hierarchical levels: peptide identification, species-specific potency, and strain-level sensitivity.

## Overview

The framework is comprised of three integrated modules:

* **tsAMP-I (Identification)**: Used for the classification of sequences as AMPs or non-AMPs. A curated dataset encompassing comprehensive AMP-related information is utilized to enhance the generalization capability of the model.
* **tsAMP-C (Species-level)**: Developed to predict the inhibitory potency of AMPs against **33 bacterial species**. Features are extracted from both AMPs and bacterial species using ESM-1v.
* **tsAMP-CS (Strain-level)**: Designed to calculate precise **MIC values** against specific strains from 10 prevalent target species. Strain-specific features are leveraged to allow for extensions to other pathogenic bacteria.

---

## Preparation

Feature extraction in tsAMP is powered by the ESM-1v model.

1. The ESM-1v model parameters should be obtained from the [ESM GitHub Repository](https://github.com/facebookresearch/esm).
2. Feature extraction is performed using the following command:

```bash
python scripts/extract.py
```
---
## Installation
tsAMP using python3.11 can install the environment using the following instructions, which contain the configuration of the esm2 environment

   conda env create -f environment.yml


## Training and Inference

Detailed instructions for operating each module are provided below.

### 1. tsAMP-I (Identification)
To perform AMP screening or classification on test sequences, the following script is utilized:

```bash
python model/tsAMP-I/predict.py
```

### 2. tsAMP-C (Species-level Prediction)
The tsAMP-C module supports both GAN-based training and direct inference.

* **Training**: Data augmentation and model training are executed via:
    ```bash
    python trainGAN.py
    ```
    *Note: Training and testing samples for different MIC thresholds are stored in `/tsAMP/data/tsAMP-C/MIC/`. Target species mean representations are located in `/tsAMP/data/tsAMP-C/species/`.*

* **Inference**: To predict MIC labels for a specific target (e.g., *Candida albicans*), the following command is executed:
    ```bash
    python predict.py \
      --output_excel test.xlsx \
      --model_path "/tsAMP/model/tsAMP-C/mic16_Candida_albicans.pt" \
      --test_dir "/tsAMP/data/tsAMP-C/MIC/test16/16_Candida_albicans_2.xlsx"
    ```

### 3. tsAMP-CS (Strain-level Prediction)
Comprehensive workflows for training and testing at the strain level are encapsulated in the provided shell script:

```bash
bash run.sh
```

---

## Directory Structure

To ensure the framework functions correctly, the following data organization is recommended:

* `/data/tsAMP-C/MIC/`: Contains training and testing samples categorized by MIC thresholds.
* `/tsAMP/data/tsAMP-C/species/`: Contains the mean representation vectors for target species.
