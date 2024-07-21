# A Mutual Knowledge Distillation-Empowered AI Framework for Early Detection of Alzheimer's Disease Using Incomplete Multi-Modal Images
This repository provides the code to implement the following [paper](https://pubmed.ncbi.nlm.nih.gov/37662267/):

Kwak, M. G., Su, Y., Chen, K., Weidman, D., Wu, T., Lure, F., & Li, J. (2023). A Mutual Knowledge Distillation-Empowered AI Framework for Early Detection of Alzheimer’s Disease Using Incomplete Multi-Modal Images. <i>medRxiv</i>.

The updated version is currently under revision for publication in a peer-reviewed journal.

We propose a deep learning-based framework that employs Mutual Knowledge Distillation (MKD) to jointly model different sub-cohorts based on their respective available image modalities.
In MKD, the model with more modalities (e.g., MRI and PET) is considered a teacher while the model with fewer modalities (e.g., only MRI) is considered a student.
Our proposed MKD framework includes three key components:
1. A teacher model that is student-oriented, namely the Student-oriented Multi-modal Teacher (SMT), through multi-modal information disentanglement. 
2. Train the student model by not only minimizing its classification errors but also learning from the SMT teacher. 
3. Update the teacher model by transfer learning from the student’s feature extractor because the student model is trained with more samples.

## Requirements
### Installation
To use this package safely, ensure you have the following:
* Python 3.10+ environment
* PyTorch 2.0.0+

Additionally, we recommend using [wandb](https://wandb.ai/site) to track model training and evaluation. It can be enabled by `enable_wandb` argument.

## File Description
The modified implementations (`modality.py`, `ANTs.py`, and `preprocessor.py`) locate in `modified` directory.
```    .[.gitignore](.gitignore)
    ├── configs/                        # task-specific arguments
    ├── datasets/                       # pytorch Dataset and transformation functions
    ├── models/                         # backbone and head: subnetworks of DenseNet and ResNet
    ├── shell/                          # shell scripts for repeating experiments
    ├── tasks/                          # training teacher, distillation, and enhancing the final model                         
    ├── utils/                          # various functions for GPU setting, evaluation, optimization, and so on.
    ├── run_general_teacher.py          # step 1. train teacher
    ├── run_general_distillation.py     # step 2. knowledge disilltation
    ├── run_final_multi.py              # step 3. update teacher by student
    ├── run_single.py                   # baseline for single-modality
    └── run_multi.py                    # baseline for multi-modalities
```

## Usage
### main run codes
1. Train a teacher model by `run_general_teacher.py`
2. Conduct knowledge distillation by `run_general_distillation.py`
3. Update the multi-modal teacher by `run_final_multi.py`

```
python run_general_teacher.py
python run_general_distillation.py --teacher_dir your_teacher_dir
python run_final_multi.py --student_dir your_student_dir
```

`run_final_multi.py` automatically tracks the general teacher checkpoint, which was used for student training.

### Citation
If you use this project in your research, please cite it as follows:
```
@article{kwak2023mutual,
  title={A Mutual Knowledge Distillation-Empowered AI Framework for Early Detection of Alzheimer’s Disease Using Incomplete Multi-Modal Images},
  author={Kwak, Min Gu and Su, Yi and Chen, Kewei and Weidman, David and Wu, Teresa and Lure, Fleming and Li, Jing},
  journal={medRxiv},
  year={2023},
  publisher={Cold Spring Harbor Laboratory Preprints}
}
```
