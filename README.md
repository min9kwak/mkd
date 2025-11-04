# A Cross-Modal Mutual Knowledge Distillation Framework for Alzheimer's Disease Diagnosis: Addressing Incomplete Modalities

This repository provides the official implementation of the following paper published in **IEEE Transactions on Automation Science and Engineering**:

**Kwak, M. G., Mao, L., Zheng, Z., Su, Y., Lure, F., & Li, J. (2025).** A Cross-Modal Mutual Knowledge Distillation Framework for Alzheimer's Disease Diagnosis: Addressing Incomplete Modalities. *IEEE Transactions on Automation Science and Engineering*. [https://doi.org/10.1109/TASE.2025.3556290](https://doi.org/10.1109/TASE.2025.3556290)

---

## 📖 Overview

Alzheimer's Disease (AD) detection benefits significantly from multi-modal imaging, particularly combining **MRI** and **PET** scans. However, PET imaging is:
- Expensive and requires radiotracer injection
- Limited in availability and infrastructure
- Often missing in clinical settings

**The Challenge**: Many patients have MRI scans but lack PET scans, creating an **incomplete multi-modal data problem**.

**Our Solution**: We propose a **Cross-Modal Mutual Knowledge Distillation (MKD)** framework that:
1. Trains a multi-modal teacher model to extract common representations from MRI+PET data
2. Distills this knowledge to a student model that uses only MRI
3. Transfers the student's richer MRI information back to enhance the teacher model

This mutual learning approach enables effective AD diagnosis even when PET scans are unavailable.

---

## 🎯 Method

Our framework consists of three key training stages:

### Stage 1: Student-oriented Multi-modal Teacher (SMT)
- **Input**: Paired MRI + PET scans (complete multi-modal data)
- **Architecture**: Dual-branch network with modality-specific extractors and shared encoder
- **Goal**: Learn to disentangle common and modality-specific representations through:
  - Similarity loss: Encourages common features across modalities
  - Difference loss: Separates modality-specific features
  - Reconstruction loss: Ensures information preservation
  - Classification loss: Maintains diagnostic accuracy

**Key Innovation**: The teacher is designed to be student-oriented, extracting representations that are learnable from MRI alone.

### Stage 2: Knowledge Distillation to MRI-only Student (SMT-Student)
- **Input**: MRI-only scans (both complete and incomplete cases)
- **Teacher**: Frozen SMT model from Stage 1
- **Goal**: Train student model to mimic teacher's common representations using only MRI
- **Advantage**: Student has access to more training samples (patients with MRI but no PET)

### Stage 3: Mutual Transfer to Enhanced Teacher (SMT⁺)
- **Input**: Paired MRI + PET scans
- **Initialization**: Load pre-trained SMT model
- **Knowledge Transfer**: Use SMT-Student's enriched MRI feature extractor
- **Goal**: Enhance multi-modal teacher with student's richer MRI knowledge
- **Result**: SMT⁺ model with improved performance on complete multi-modal data

---

## 🚀 Quick Start

### Requirements
- Python 3.10+
- PyTorch 2.0.0+
- Additional dependencies: `numpy`, `pandas`, `scikit-learn`, `rich`, `wandb` (optional)

### Installation
```bash
pip install torch torchvision numpy pandas scikit-learn rich wandb
```

### Training Pipeline

#### Step 1: Train SMT (Multi-modal Teacher)
```bash
python run_smt.py \
    --gpus 0 \
    --server main \
    --epochs 100 \
    --batch_size 32
```

#### Step 2: Train SMT-Student (Knowledge Distillation)
```bash
python run_smt_student.py \
    --teacher_dir checkpoints/SMT-FBP/YYYY-MM-DD_HH-MM-SS \
    --teacher_position best \
    --gpus 0 \
    --epochs 100 \
    --batch_size 32
```

#### Step 3: Train SMT⁺ (Enhanced Multi-modal Model)
```bash
python run_smt_plus.py \
    --student_dir checkpoints/SMT-Student-FBP/YYYY-MM-DD_HH-MM-SS \
    --student_position best \
    --gpus 0 \
    --epochs 100 \
    --batch_size 32
```

> **Note**: `run_smt_plus.py` automatically tracks the teacher checkpoint used for student training.

### Baseline Models

Train single-modality baseline:
```bash
python run_single.py --modality mri --gpus 0
```

Train multi-modality baseline (without MKD):
```bash
python run_multi.py --gpus 0
```

---

## 📁 Project Structure

```
.
├── configs/                        # Configuration files for different tasks
│   ├── base.py                    # Base configuration class
│   └── slice/                     # Task-specific configs
│       ├── smt.py                 # SMT configuration
│       ├── smt_student.py         # SMT-Student configuration
│       ├── smt_plus.py            # SMT⁺ configuration
│       ├── single.py              # Single-modality baseline config
│       └── multi.py               # Multi-modality baseline config
│
├── datasets/                      # Dataset loading and preprocessing
│   ├── brain.py                   # Main dataset classes (BrainProcessor, BrainMulti, BrainMRI)
│   ├── samplers.py                # Custom samplers for imbalanced data
│   └── slice/
│       └── transforms.py          # Image augmentation and transforms
│
├── models/                        # Neural network architectures
│   └── slice/
│       ├── backbone.py            # DenseNet backbone implementations
│       ├── resnet.py              # ResNet backbone implementations
│       ├── head.py                # Classification heads
│       ├── demo.py                # Demographic information encoder
│       └── build.py               # Model building utilities
│
├── tasks/                         # Training logic for different stages
│   └── slice/
│       ├── smt.py                 # SMT training task
│       ├── smt_student.py         # SMT-Student training task
│       ├── smt_plus.py            # SMT⁺ training task
│       ├── single.py              # Single-modality training
│       └── multi.py               # Multi-modality baseline training
│
├── utils/                         # Utility functions
│   ├── gpu.py                     # GPU setup and distributed training
│   ├── logging.py                 # Logging utilities
│   ├── loss.py                    # Custom loss functions (similarity, difference, etc.)
│   ├── metrics.py                 # Evaluation metrics
│   └── optimization.py            # Optimizers and schedulers
│
├── shell/                         # Shell scripts for batch experiments
│   ├── run_smt.sh
│   ├── run_smt_student.sh
│   └── run_smt_plus.sh
│
├── run_smt.py                     # Main script: Train SMT
├── run_smt_student.py             # Main script: Train SMT-Student
├── run_smt_plus.py                # Main script: Train SMT⁺
├── run_single.py                  # Main script: Single-modality baseline
└── run_multi.py                   # Main script: Multi-modality baseline
```

---

## 💾 Data Preparation

### Required Data Structure

Your data directory should be organized as follows:

```
{DATA_ROOT}/
├── labels/
│   ├── data_info_multi.csv        # Main metadata file
│   ├── mri_abnormal.pkl           # List of abnormal MRI scans to exclude
│   ├── AV45_FBP_SUVR.xlsx        # PET meta-ROI composite scores
│   └── MRI_BAI_features.csv      # MRI hippocampus volume features
│
├── template/FS7/                  # Preprocessed MRI images
│   ├── {subject_id}.pkl
│   └── ...
│
└── template/PUP_FBP/              # Preprocessed PET images (FBP tracer)
    ├── {subject_id}.pkl
    └── ...
```

### CSV Format (`data_info_multi.csv`)

Required columns:
- **RID**: Patient ID (string)
- **Conv**: Conversion label (0=non-converter, 1=converter to AD, -1=unlabeled)
- **MRI**: MRI filename (without extension)
- **FBP**: PET filename (without extension) - can be NaN for incomplete cases
- **PTGENDER**: Gender (1=male, 2=female)
- **Age**: Age at scan
- **CDGLOBAL**: Clinical Dementia Rating global score
- **MMSCORE**: Mini-Mental State Examination score
- **APOE**: APOE genotype status
- **IS_FILE**: Boolean indicating file availability

### Image Format

- Images should be saved as `.pkl` files containing 3D numpy arrays
- MRI: T1-weighted structural MRI, preprocessed and registered
- PET: Amyloid PET (FBP tracer), preprocessed and registered to MRI space

### Additional Metadata Files

1. **AV45_FBP_SUVR.xlsx**: 
   - Column `ID`: PET filename
   - Column `MC`: Meta-ROI composite SUVR value

2. **MRI_BAI_features.csv**:
   - Column `Filename`: Path containing MRI filename
   - Columns `Left-Hippocampus`, `Right-Hippocampus`: Volume measurements

---

## 🔧 Configuration

Key hyperparameters can be set via command-line arguments or modified in config files:

- `--gpus`: GPU device IDs (e.g., `0` or `0,1`)
- `--batch_size`: Batch size for training
- `--epochs`: Number of training epochs
- `--lr`: Learning rate
- `--missing_rate`: Simulated PET missing rate (for experiments)
- `--enable_wandb`: Enable Weights & Biases logging
- `--balance`: Use class balancing for imbalanced data

See individual config files in `configs/slice/` for full options.

---

## 📊 Evaluation

Models are evaluated on:
- **Accuracy**: Overall classification accuracy
- **AUC**: Area under ROC curve
- **Sensitivity & Specificity**: For clinical relevance

Evaluation is performed on test sets with complete multi-modal data (MRI + PET).

---

## 📝 Citation

If you use this project in your research, please cite:

```bibtex
@article{kwak2025cross,
  title={A Cross-Modal Mutual Knowledge Distillation Framework for Alzheimer's Disease Diagnosis: Addressing Incomplete Modalities},
  author={Kwak, Min Gu and Mao, Lingchao and Zheng, Zhiyang and Su, Yi and Lure, Fleming and Li, Jing},
  journal={IEEE Transactions on Automation Science and Engineering},
  year={2025},
  publisher={IEEE}
}
```

---

## 📧 Contact

For questions or issues, please open an issue in this repository or contact the authors.

---

## 📄 License

This project is released for academic research use. Please refer to the paper for details on dataset access and usage restrictions.
