# tsAMP: A Multi-level Framework for Antimicrobial Peptide Prediction

tsAMP is a novel predictive framework based on the **ESM-1v** protein language model. It provides comprehensive antimicrobial peptide (AMP) analysis across three hierarchical levels: peptide identification, species-specific potency, and strain-level sensitivity.

---

## Overview

The framework comprises three integrated modules:

- **tsAMP-I (Identification)**: Classifies input sequences as AMPs or non-AMPs. A curated dataset encompassing comprehensive AMP-related information is utilized to enhance the generalization capability of the model.
- **tsAMP-C (Species-level)**: Predicts the inhibitory potency of AMPs against **33 bacterial species**. Features are extracted from both AMPs and bacterial species using ESM-1v.
- **tsAMP-CS (Strain-level)**: Calculates precise **MIC values** against specific strains from 10 prevalent target species. Strain-specific features are leveraged to allow for extensions to other pathogenic bacteria.

---

## Table of Contents

1. [System Requirements](#1-system-requirements)
2. [Installation (Step-by-Step)](#2-installation-step-by-step)
3. [Feature Extraction with ESM-1v](#3-feature-extraction-with-esm-1v)
4. [Module Usage](#4-module-usage)
   - [4.1 tsAMP-I (Identification)](#41-tsamp-i-identification)
   - [4.2 tsAMP-C (Species-level Prediction)](#42-tsamp-c-species-level-prediction)
   - [4.3 tsAMP-CS (Strain-level Prediction)](#43-tsamp-cs-strain-level-prediction)
5. [Running Baseline / Comparison Models (testmodel)](#5-running-baseline--comparison-models-testmodel)
6. [Directory Structure](#6-directory-structure)
7. [Expected Outputs](#7-expected-outputs)
8. [Troubleshooting](#8-troubleshooting)

---

## 1. System Requirements

Before beginning, please ensure that your system meets the following requirements:

| Requirement | Details |
|---|---|
| Operating System | Linux (Ubuntu 18.04+ recommended), macOS, or Windows with WSL |
| Python | 3.11 (other versions have not been tested) |
| Conda | Miniconda or Anaconda (for environment management) |
| GPU | NVIDIA GPU with CUDA support is **strongly recommended** for feature extraction and training. CPU-only execution is possible but significantly slower. |
| Disk Space | At least **10 GB** of free space (the ESM-1v model weights are approximately 3 GB) |
| RAM | 16 GB minimum; 32 GB recommended |

**If you do not have Conda installed**, please follow the official guide:
- Miniconda (lightweight, recommended): https://docs.conda.io/en/latest/miniconda.html
- Download the installer for your OS, then run:
  ```bash
  # Example for Linux:
  bash Miniconda3-latest-Linux-x86_64.sh
  ```
- After installation, close and reopen your terminal, then verify:
  ```bash
  conda --version
  # Expected output (example): conda 24.x.x
  ```

---

## 2. Installation (Step-by-Step)

### Step 1: Clone the repository

```bash
git clone https://github.com/YangLab-BUPT/tsAMP.git
cd tsAMP
```

### Step 2: Create the Conda environment

The file `environment.yml` contains all required Python packages (including ESM-1v dependencies). Run:

```bash
conda env create -f environment.yml
```

This process may take 5–15 minutes depending on your internet speed. When it finishes, you should see a message similar to:

```
done
#
# To activate this environment, use
#
#     $ conda activate tsAMP
```

### Step 3: Activate the environment

**You must activate this environment every time you open a new terminal before running any tsAMP commands.**

```bash
conda activate tsAMP
```

After activation, your terminal prompt should change to show the environment name, for example:

```
(tsAMP) user@machine:~/tsAMP$
```

### Step 4: Verify the installation

Run the following commands to confirm everything is properly installed:

```bash
python --version
# Expected output: Python 3.11.x

python -c "import torch; print('PyTorch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
# Expected output (example):
# PyTorch version: 2.x.x
# CUDA available: True    (if you have a GPU; False is acceptable for CPU-only use)

python -c "import esm; print('ESM imported successfully')"
# Expected output: ESM imported successfully
```

If any of these commands produce an error, please refer to the [Troubleshooting](#8-troubleshooting) section.

---

## 3. Feature Extraction with ESM-1v

tsAMP uses the **ESM-1v** protein language model to generate sequence representations. Before running any prediction module, you must first extract features from your peptide sequences.

### Step 1: Download the ESM-1v model weights

The model weights can be obtained from the official ESM repository:
- Repository: https://github.com/facebookresearch/esm
- The required model is **ESM-1v** (`esm1v_t33_650M_UR90S`).

If the weights are not downloaded automatically by the script, you may manually download them:

```bash
# The model will be cached to ~/.cache/torch/hub/checkpoints/ by default.
# Alternatively, you can download manually:
wget https://dl.fbaipublicfiles.com/fair-esm/models/esm1v_t33_650M_UR90S_1.pt -P ~/.cache/torch/hub/checkpoints/
```

### Step 2: Prepare your input sequences

Place your peptide sequences in **FASTA format** in the designated input directory. A sample input file is provided at `data/sample_input.fasta`. The format should be:

```
>peptide_1
GLFDIVKKVVGALLAG
>peptide_2
KWKLFKKIEKVGQNIRDGIIK
```

### Step 3: Run feature extraction

```bash
python scripts/extract.py
```

Extracted features will be saved to the `data/` directory (the exact output path is printed to the terminal upon completion). These feature files are used as input for all three prediction modules.

---

## 4. Module Usage

> **Important**: Make sure you have activated the Conda environment (`conda activate tsAMP`) and completed feature extraction (Section 3) before proceeding.

### 4.1 tsAMP-I (Identification)

**Purpose**: Classify input peptide sequences as AMPs (antimicrobial peptides) or non-AMPs.

**Command**:

```bash
cd model/tsAMP-I
python predict.py
```

**Input**: The script reads pre-extracted ESM-1v features. Ensure that feature extraction (Section 3) has been completed.

**Output**: A results file will be generated in the current directory, containing the predicted label (AMP or non-AMP)  for each input sequence.

---

### 4.2 tsAMP-C (Species-level Prediction)

**Purpose**: Predict whether a given AMP can inhibit a specific bacterial species (at a defined MIC threshold).

#### 4.2.1 Training (optional — pre-trained models are provided)

If you wish to retrain the model with GAN-based data augmentation:

```bash
cd model/tsAMP-C
python trainGAN.py
```

Training data and testing samples for different MIC thresholds are located in:

```
data/tsAMP-C/MIC/
├── train16/          
│   ├── 16_Escherichia_coli_1.xlsx
│   ├── 16_Candida_albicans_1.xlsx
│   └── ...
├── test16/            
│   ├── 16_Escherichia_coli_2.xlsx
│   ├── 16_Candida_albicans_2.xlsx
│   └── ...
└── ...                # Other MIC thresholds (e.g., train32/, test32/)
```

Target species mean representations are located in:

```
data/tsAMP-C/species/
├── Escherichia_coli.pt
├── Candida_albicans.pt
└── ...                # 33 species in total
```

#### 4.2.2 Inference (prediction using pre-trained models)

To predict whether AMPs can inhibit a specific species, run the following command. Here we use *Candida albicans* at MIC threshold 16 µg/mL as an example:

```bash
cd model/tsAMP-C

python predict.py \
  --output_excel test.xlsx \
  --model_path "/path/to/tsAMP/model/tsAMP-C/mic16_Candida_albicans.pt" \
  --test_dir "/path/to/tsAMP/data/tsAMP-C/MIC/test16/16_Candida_albicans_2.xlsx"
```

**Parameter explanation**:

| Parameter | Description | Example |
|---|---|---|
| `--output_excel` | Filename for the output results | `test.xlsx` |
| `--model_path` | Path to the pre-trained model file for a specific species and MIC threshold | `model/tsAMP-C/mic16_Candida_albicans.pt` |
| `--test_dir` | Path to the test data file | `data/tsAMP-C/MIC/test16/16_Candida_albicans_2.xlsx` |

> **Note**: Replace `/path/to/tsAMP/` with the actual absolute path to your tsAMP installation directory. For example, if you cloned the repository to your home directory, it would be `/home/username/tsAMP/`.

**Available pre-trained models**: All pre-trained model files are stored in `model/tsAMP-C/` and follow the naming convention `mic{threshold}_{Species_name}.pt`. To predict for a different species, simply change the model and test file paths accordingly.

**Output**: The output Excel file (`test.xlsx`) contains the predicted inhibitory labels and confidence scores for each AMP in the test set.

---

### 4.3 tsAMP-CS (Strain-level Prediction)

**Purpose**: Predict precise MIC values against specific bacterial strains from the 10 most prevalent target species.

**Command**:

```bash
cd model/tsAMP-CS
bash run.sh
```

The `run.sh` script executes the complete training and testing workflow. To view what the script does before running it, you can inspect it with:

```bash
cat run.sh
```

**Output**: Strain-level MIC prediction results will be saved in the designated output directory (as specified inside `run.sh`).

---

## 5. Running Baseline / Comparison Models (testmodel)

The `testmodel/` directory contains scripts and configurations for running baseline and comparison models referenced in our paper. This allows for independent verification and benchmarking against tsAMP.

For detailed instructions, please refer to:

```
testmodel/README.md
```

To get started quickly:

```bash
cd testmodel
# Follow the instructions in testmodel/README.md
```

> **Note**: Some baseline models may require additional dependencies. Please consult `testmodel/README.md` for specific environment setup instructions.

---

## 6. Directory Structure

```
tsAMP/
├── environment.yml                  # Conda environment configuration
├── README.md                        # This file
│
├── scripts/
│   └── extract.py                   # ESM-1v feature extraction script
│
├── data/
│   ├── sample_input.fasta           # Example input sequences
│   ├── tsAMP-C/
│   │   ├── MIC/
│   │   │   ├── train16/             # Training data (MIC threshold = 16)
│   │   │   ├── test16/              # Testing data  (MIC threshold = 16)
│   │   │   └── ...                  # Other thresholds
│   │   └── species/                 # Species mean representation vectors (.pt)
│   └── tsAMP-CS/                    # Strain-level data
│
├── model/
│   ├── tsAMP-I/
│   │   └── predict.py               # AMP identification script
│   ├── tsAMP-C/
│   │   ├── trainGAN.py              # GAN-based training script
│   │   ├── predict.py               # Species-level prediction script
│   │   └── mic16_*.pt               # Pre-trained model weights
│   └── tsAMP-CS/
│       └── run.sh                   # Strain-level training & testing workflow
│
└── testmodel/
    ├── README.md                    # Instructions for baseline models
    └── ...                          # Baseline model scripts and configs
```

---

## 7. Expected Outputs

To help verify that your setup is working correctly, the table below summarizes what each module should produce:

| Module | Command | Output Location | Output Format | What to Expect |
|---|---|---|---|---|
| Feature Extraction | `python scripts/extract.py` | `data/` | `.pt` files | One feature file per input sequence |
| tsAMP-I | `python predict.py` | `model/tsAMP-I/` | `.csv` or `.xlsx` | Columns: sequence ID, predicted label (AMP/non-AMP), confidence score |
| tsAMP-C | `python predict.py --output_excel test.xlsx ...` | Current directory | `.xlsx` | Columns: sequence ID, predicted inhibitory label, confidence score |
| tsAMP-CS | `bash run.sh` | As specified in `run.sh` | `.xlsx` or `.csv` | Columns: sequence ID, predicted MIC value, strain information |

---

## 8. Troubleshooting

**Q: `conda: command not found`**
Conda is not installed or not in your system PATH. Please install Miniconda (see Section 1) and restart your terminal.

**Q: `conda env create` fails with "ResolvePackageNotFound"**
Some packages may not be available for your operating system. Try updating Conda first:
```bash
conda update conda
conda env create -f environment.yml
```

**Q: `ModuleNotFoundError: No module named 'esm'`**
The Conda environment is not activated. Run:
```bash
conda activate tsAMP
```

**Q: `CUDA out of memory` error during feature extraction**
The ESM-1v model requires significant GPU memory. You can reduce the batch size in `scripts/extract.py`, or use CPU mode by adding `--nogpu` (if supported), though this will be substantially slower.

**Q: `FileNotFoundError` when running prediction scripts**
Ensure that you have completed feature extraction (Section 3) before running any prediction module, and that the file paths in your commands match your actual directory structure.

**Q: `Permission denied` when running `bash run.sh`**
Grant execute permission to the script:
```bash
chmod +x run.sh
bash run.sh
```

**Q: How do I use my own peptide sequences?**
Prepare your sequences in FASTA format (see the example in Section 3, Step 2), place the file in the `data/` directory, update the input path in `scripts/extract.py` if necessary, then follow the workflow from Section 3 onward.

---

## Citation


---

## License


## Contact

For questions or issues, please open an issue on this repository or contact the corresponding author.
