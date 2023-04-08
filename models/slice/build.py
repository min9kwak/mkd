from models.slice.backbone import ResNetBackbone, DenseNetBackbone
from models.slice.head import LinearClassifier


def build_networks_single(config, **kwargs):

    # 1. Encoder
    if 'resnet' in config.encoder_type:
        encoder = ResNetBackbone(name=config.encoder_type, in_channels=1)
    elif 'densenet' in config.encoder_type:
        encoder = DenseNetBackbone(name=config.encoder_type, in_channels=1)
    else:
        raise ValueError

    if config.small_kernel:
        encoder._fix_first_conv()

    # 2. Classifier
    out_dim = encoder.out_channels
    classifier = LinearClassifier(name=config.encoder_type, in_channels=out_dim, num_features=2)

    # Return
    networks = dict(encoder=encoder, classifier=classifier)

    return networks


def build_networks_multi(config, **kwargs):

    # 1. Encoder
    if 'resnet' in config.encoder_type:
        encoder_mri = ResNetBackbone(name=config.encoder_type, in_channels=1)
    elif 'densenet' in config.encoder_type:
        encoder_pet = DenseNetBackbone(name=config.encoder_type, in_channels=1)
    else:
        raise ValueError

    if config.small_kernel:
        encoder_mri._fix_first_conv()
        encoder_pet._fix_first_conv()

    # 2. Classifier
    if config.add_type == 'concat':
        out_dim = encoder_mri.out_channels + encoder_pet.out_channels
    else:
        assert encoder_mri.out_channels == encoder_pet.out_channels
        out_dim = encoder_mri.out_channels
    classifier = LinearClassifier(name=config.encoder_type, in_channels=out_dim, num_features=2)

    # Return
    networks = dict(encoder_mri=encoder_mri, encoder_pet=encoder_pet, classifier=classifier)

    return networks


def build_networks_kd(config, **kwargs):

    # 1. Encoder
    if 'resnet' in config.encoder_type:
        encoder_s_mri = ResNetBackbone(name=config.encoder_type, in_channels=1)
        encoder_t_mri = ResNetBackbone(name=config.encoder_type, in_channels=1)
    elif 'densenet' in config.encoder_type:
        encoder_t_pet = DenseNetBackbone(name=config.encoder_type, in_channels=1)
    else:
        raise ValueError

    if config.small_kernel:
        encoder_s_mri._fix_first_conv()
        encoder_t_mri._fix_first_conv()
        encoder_t_pet._fix_first_conv()

    # 2. Classifier
    classifier_s = LinearClassifier(name=config.encoder_type, in_channels=encoder_s_mri.out_channels, num_features=2)

    if config.add_type == 'concat':
        out_dim = encoder_t_mri.out_channels + encoder_t_pet.out_channels
    else:
        assert encoder_t_mri.out_channels == encoder_t_pet.out_channels
        out_dim = encoder_t_mri.out_channels
    classifier_t = LinearClassifier(name=config.encoder_type, in_channels=out_dim, num_features=2)

    # Return
    networks = dict(encoder_s_mri=encoder_s_mri, encoder_t_mri=encoder_t_mri, encoder_t_pet=encoder_t_pet,
                    classifier_t=classifier_t)

    return networks
