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
    ├── configs/                  # task-specific arguments
    ├── datasets/                 # pytorch Dataset and transformation functions
    ├── layers/                   # functions for SMoCo                     
    ├── models/                   # backbone and head: subnetworks of DenseNet and ResNet
    ├── tasks/                    # SMoCo, fine-tuning (w/ and w/o demographic information), and external evaluation on AIBL dataset                         
    ├── utils/                    # various functions for GPU setting, evaluation, optimization, and so on.
    ├── run_supmoco.py            # the main run code for SMoCo pre-training
    ├── run_finetune.py           # the main run code for SMoCo fine-tuning
    ├── run_demo_finetune.py      # the main run code for SMoCo fine-tuning with demographic information
    ├── run_classification.py     # the main run code for simple classification model
    └── run_aibl.py               # The main run code for blind evaluation on AIBL dataset
```

## Usage
### main run codes
1. Pre-train SMoCo with `run_supmoco.py`.
2. Fine-tune the pretrained SMoCo with `run_finetune.py` or `run_demo_finetune.py`. Make sure to correctly input the pre-trained checkpoint directory.
3. Other run codes are implemented for additional evaluations.

```
python run_supmoco.py
python run_finetune.py --pretrained_dir your_pretrained_dir
```

### Citation
If you use this project in your research, please cite it as follows:
```
@article{kwak2023self,
  title={Self-Supervised Contrastive Learning to Predict the Progression of Alzheimer’s Disease with 3D Amyloid-PET},
  author={Kwak, Min Gu and Su, Yi and Chen, Kewei and Weidman, David and Wu, Teresa and Lure, Fleming and Li, Jing and Alzheimer’s Disease Neuroimaging Initiative},
  journal={Bioengineering},
  volume={10},
  number={10},
  pages={1141},
  year={2023},
  publisher={MDPI}
}
```
