from models.slice.backbone import ResNetBackbone, DenseNetBackbone
from models.slice.head import LinearClassifier


def build_networks(config, **kwargs):

    # 1. Encoder
    if 'resnet' in config.encoder_type:
        encoder_mri = ResNetBackbone(name=config.encoder_type, in_channels=1)
        encoder_pet = ResNetBackbone(name=config.encoder_type, in_channels=1)
    elif 'densenet' in config.encoder_type:
        encoder_mri = DenseNetBackbone(name=config.encoder_type, in_channels=1)
        encoder_pet = DenseNetBackbone(name=config.encoder_type, in_channels=1)
    else:
        raise ValueError

    # 2. Classifier
    if hasattr(config, 'add_type'):
        if config.add_type == 'concat':
            out_dim = encoder_mri.out_channels * 2
        else:
            out_dim = encoder_mri.out_channels
    else:
        out_dim = encoder_mri.out_channels
    classifier_mri = LinearClassifier(name=config.encoder_type, in_channels=encoder_mri.out_channels, num_features=2)
    classifier_pet = LinearClassifier(name=config.encoder_type, in_channels=encoder_pet.out_channels, num_features=2)
    classifier = LinearClassifier(name=config.encoder_type, in_channels=out_dim, num_features=2)

    # Return
    networks = dict(encoder_mri=encoder_mri, encoder_pet=encoder_pet,
                    classifier_mri=classifier_mri, classifier_pet=classifier_pet,
                    classifier=classifier)

    return networks
