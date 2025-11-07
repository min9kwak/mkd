import argparse
from configs.base import ConfigBase, str2bool


class MDTStudentConfig(ConfigBase):
    """Configuration for MDT-Student model.
    
    MDT-Student learns from the MDT teacher using only MRI scans,
    enabling diagnosis for patients without PET imaging.
    Trained with both complete (MRI+PET) and incomplete (MRI-only) samples.
    """

    def __init__(self, args=None, **kwargs):
        super(MDTStudentConfig, self).__init__(args, **kwargs)

    @staticmethod
    def data_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing data-related arguments."""

        parser = argparse.ArgumentParser("Data", add_help=False)
        return parser

    @staticmethod
    def model_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing model-related arguments."""
        parser = argparse.ArgumentParser("Model Configuration", add_help=False)
        return parser

    @staticmethod
    def train_parser() -> argparse.ArgumentParser:
        """Returns an `argparse.ArgumentParser` instance containing training-related arguments."""
        parser = argparse.ArgumentParser("Model Training", add_help=False)
        parser.add_argument('--epochs', type=int, default=100, help='Number of training epochs.')
        parser.add_argument('--batch_size', type=int, default=16, help='Mini-batch size.')
        parser.add_argument('--num_workers', type=int, default=4, help='Number of CPU threads.')

        # Optimizer: MRI & PET
        parser.add_argument('--optimizer', type=str, default='adamw', choices=('sgd', 'adamw'),
                            help='Optimization algorithm.')
        parser.add_argument('--learning_rate', type=float, default=0.0001, help='Base learning rate to start from.')
        parser.add_argument('--weight_decay', type=float, default=0.0001, help='Weight decay factor.')

        # Scheduler
        parser.add_argument('--cosine_warmup', type=int, default=0,
                            help='Number of warmups before cosine LR scheduling (-1 to disable.)')
        parser.add_argument('--cosine_cycles', type=int, default=1,
                            help='Number of hard cosine LR cycles with hard restarts.')
        parser.add_argument('--cosine_min_lr', type=float, default=0.0,
                            help='LR lower bound when cosine scheduling is used.')
        parser.add_argument('--mixed_precision', type=str2bool, default=True, help='Use float16 precision.')

        return parser

    @staticmethod
    def task_specific_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser('General Teacher', add_help=False)
        parser.add_argument('--balance', type=str2bool, help='apply class balance weight')
        parser.add_argument('--sampler_type', type=str, default='stratified', choices=('over', 'stratified'))
        parser.add_argument('--different_lr', type=str2bool, help='apply class balance weight')

        # Weight Alpha
        parser.add_argument('--alpha_ce', type=float, default=1.0)
        parser.add_argument('--alpha_kd_repr', type=float, default=10.0)
        parser.add_argument('--alpha_kd_clf', type=float, default=10.0)

        # Knowledge Distillation
        parser.add_argument('--temperature', type=float, default=5.0)
        parser.add_argument('--teacher_dir', type=str)
        parser.add_argument('--teacher_position', type=str, default='last')
        parser.add_argument('--use_teacher', type=str2bool, default=True, help='student uses pretrained teacher.')

        # Student network type
        parser.add_argument('--use_specific', type=str2bool, default=False)
        parser.add_argument('--inherit_classifier', type=str2bool, default=False)

        return parser
