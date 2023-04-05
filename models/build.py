import os

from models.backbone.resnet import build_resnet_backbone
from models.backbone.densenet import DenseNetBackbone
from models.backbone.swin import SwinEncoder
from models.head.projector import ProjectorLinear, ProjectorMLP
from models.head.predictor import PredictorMLP
from models.head.classifier import LinearClassifier

from models.backbone.base import calculate_out_features


def build_network_single(config):

    # 0. Default
    encoder, projector, classifier = None, None, None

    # 1. Encoder
    size = config.crop_size if config.crop_size is not None else config.image_size
    if config.network_type == 'swin':
        if config.pretrained_path is not None:
            pretrained_path = os.path.join(config.root, config.pretrained_path)
        else:
            pretrained_path = None
        encoder = SwinEncoder(network_type=config.architecture,
                              pretrained_path=pretrained_path,
                              img_size=(size, size, size),
                              in_channels=1,
                              out_channels=14,
                              feature_layer=config.feature_layer)

        if config.feature_layer == 'bottleneck':
            del encoder.encoder2, encoder.encoder3, encoder.encoder4

    elif config.network_type == 'resnet':
        encoder = build_resnet_backbone(arch=int(config.architecture),
                                        no_max_pool=False,
                                        in_channels=1)
    elif config.network_type == 'densenet':
        encoder = DenseNetBackbone(in_channels=1)
    else:
        raise ValueError(f"{config.network_type} must be one of (resnet, densenet, swin)")

    # 1-2. MRI Projector
    out_dim = calculate_out_features(encoder, 1, size)
    if config.projector_type == 'linear':
        projector = ProjectorLinear(in_channels=out_dim, num_features=config.projector_dim)
    elif config.projector_type == 'mlp':
        projector = ProjectorMLP(in_channels=out_dim, num_features=config.projector_dim)
    else:
        projector = None

    # 2. Classifier
    if projector is None:
        out_features = calculate_out_features(encoder, 1, size)
        gap = True
    else:
        out_features = config.projector_dim
        gap = False

    classifier = LinearClassifier(in_channels=out_features, num_classes=2, gap=gap)
    networks = dict(encoder=encoder, projector=projector, classifier=classifier)

    return networks


def build_network(config):

    # 0. Default
    mri_encoder, pet_encoder = None, None
    mri_projector, pet_projector, mri_common_projector, pet_common_projector = None, None, None, None
    predictor = None
    mri_classifier, pet_classifier, common_classifier = None, None, None

    # 1. MRI
    if hasattr(config, 'mri_network_type'):
        mri_size = config.mri_crop_size if config.mri_crop_size is not None else config.mri_image_size
        # 1-1. MRI Encoder
        if config.mri_network_type == 'swin':
            if config.mri_pretrained_path is not None:
                pretrained_path = os.path.join(config.root, config.mri_pretrained_path)
            else:
                pretrained_path = None
            mri_encoder = SwinEncoder(network_type=config.mri_architecture,
                                      pretrained_path=pretrained_path,
                                      img_size=(mri_size, mri_size, mri_size),
                                      in_channels=1,
                                      out_channels=14,
                                      feature_layer=config.mri_feature_layer)
            if config.mri_feature_layer == 'bottleneck':
                del mri_encoder.encoder2, mri_encoder.encoder3, mri_encoder.encoder4

        elif config.mri_network_type == 'resnet':
            mri_encoder = build_resnet_backbone(arch=int(config.mri_architecture),
                                                no_max_pool=False,
                                                in_channels=1)
        elif config.mri_network_type == 'densenet':
            mri_encoder = DenseNetBackbone(in_channels=1)
        else:
            raise ValueError(f"{config.mri_network_type} must be one of (resnet, densenet, swin)")

        # 1-2. MRI Projector
        mri_out_dim = calculate_out_features(mri_encoder, 1, mri_size)
        if config.mri_projector_type == 'linear':
            mri_projector = ProjectorLinear(in_channels=mri_out_dim, num_features=config.projector_dim)
            mri_common_projector = ProjectorLinear(in_channels=mri_out_dim, num_features=config.projector_dim)
        elif config.mri_projector_type == 'mlp':
            mri_projector = ProjectorMLP(in_channels=mri_out_dim, num_features=config.projector_dim)
            mri_common_projector = ProjectorMLP(in_channels=mri_out_dim, num_features=config.projector_dim)

    # 2. PET
    if hasattr(config, 'pet_network_type'):
        pet_size = config.pet_crop_size if config.pet_crop_size is not None else config.pet_image_size
        # 2-1. PET Encoder
        if config.pet_network_type == 'swin':
            if config.pet_pretrained_path is not None:
                pretrained_path = os.path.join(config.root, config.pet_pretrained_path)
            else:
                pretrained_path = None
            pet_encoder = SwinEncoder(network_type=config.pet_architecture,
                                      pretrained_path=pretrained_path,
                                      img_size=(pet_size, pet_size, pet_size),
                                      in_channels=1,
                                      out_channels=14,
                                      feature_layer=config.pet_feature_layer)
            if config.pet_feature_layer == 'bottleneck':
                del pet_encoder.encoder2, pet_encoder.encoder3, pet_encoder.encoder4

        elif config.pet_network_type == 'resnet':
            pet_encoder = build_resnet_backbone(arch=int(config.pet_architecture),
                                                no_max_pool=False,
                                                in_channels=1)
        elif config.pet_network_type == 'densenet':
            pet_encoder = DenseNetBackbone(in_channels=1)
        else:
            raise ValueError(f"{config.pet_network_type} must be one of (resnet, densenet, swin)")

        # 2-2. PET Projector
        pet_out_dim = calculate_out_features(pet_encoder, 1, pet_size)

        if config.pet_projector_type == 'linear':
            pet_projector = ProjectorLinear(in_channels=pet_out_dim, num_features=config.projector_dim)
            pet_common_projector = ProjectorLinear(in_channels=pet_out_dim, num_features=config.projector_dim)
        elif config.pet_projector_type == 'mlp':
            pet_projector = ProjectorMLP(in_channels=pet_out_dim, num_features=config.projector_dim)
            pet_common_projector = ProjectorMLP(in_channels=pet_out_dim, num_features=config.projector_dim)

    # 3. Predictor
    if hasattr(config, 'predictor_type'):
        if config.predictor_type == 'linear':
            raise NotImplementedError
        elif config.predictor_type == 'mlp':
            predictor = PredictorMLP(in_channels=config.projector_dim, num_features=config.projector_dim)

    # 5. Classifier
    if config.add_type == 'add':
        classifier_dim = config.projector_dim
    elif config.add_type == 'concat':
        classifier_dim = config.projector_dim * 2
    else:
        raise ValueError

    mri_classifier = LinearClassifier(in_channels=classifier_dim, num_classes=2, gap=False)
    pet_classifier = LinearClassifier(in_channels=classifier_dim, num_classes=2, gap=False)
    common_classifier = LinearClassifier(in_channels=classifier_dim, num_classes=2, gap=False)

    # Return
    networks = dict(mri_encoder=mri_encoder, pet_encoder=pet_encoder,
                    mri_projector=mri_projector, pet_projector=pet_projector,
                    mri_common_projector=mri_common_projector, pet_common_projector=pet_common_projector,
                    predictor=predictor,
                    mri_classifier=mri_classifier, pet_classifier=pet_classifier, common_classifier=common_classifier)

    for name, network in dict(networks).items():
        if network is None:
            del networks[name]

    return networks


if __name__ == '__main__':
    from easydict import EasyDict as edict

    config = edict()

    config.root = 'D:/data/ADNI'
    config.data_file = 'labels/data_info_multi.csv'
    config.pet_type = 'FBP'

    config.mri_image_size = 96
    config.mri_crop_size = None
    config.pet_image_size = 96
    config.pet_crop_size = None

    config.mri_network_type = 'swin'
    config.mri_architecture = 'small'
    config.mri_pretrained_path = None
    config.mri_feature_layer = 'bottleneck'
    config.mri_projector_type = 'mlp'

    config.pet_network_type = 'resnet'
    config.pet_architecture = '50'
    config.pet_small_kernel = True
    config.pet_projector_type = 'mlp'

    config.common_type = 'concat'
    config.common_projector_type = 'linear'
    config.projector_dim = 64

    config.predictor_type = 'mlp'

    config.add_type = 'concat'

    # config.predictor_type = 'linear'
    networks = build_network(config=config)

    # Test
    print(networks.keys())
    _ = [network.cuda() for name, network in networks.items()]

    import torch
    x = torch.rand(size=(2, 1, 96, 96, 96)).cuda()

    h_mri = networks['mri_encoder'](x) # (2, 384, 96, 96, 96)
    h_pet = networks['pet_encoder'](x) # (2, 2048, 96, 96, 96)

    z_mri = networks['mri_projector'](h_mri)
    z_mri_common = networks['mri_common_projector'](h_mri)

    z_pet = networks['pet_projector'](h_pet)
    z_pet_common = networks['pet_common_projector'](h_pet)

    z_pet_common_pred = networks['predictor'](z_mri_common)

    if config.add_type == 'concat':
        z_m = torch.concat([z_mri, z_pet_common], dim=1)
        z_c = torch.concat([z_mri_common, z_pet_common], dim=1)
        z_p = torch.concat([z_pet, z_mri_common], dim=1)
    else:
        z_m = z_mri + z_pet_common
        z_c = z_mri_common + z_pet_common
        z_p = z_pet + z_mri_common

    logit_m = networks['mri_classifier'](z_m)
    logit_c = networks['common_classifier'](z_c)
    logit_p = networks['pet_classifier'](z_p)
