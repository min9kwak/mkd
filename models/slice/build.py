from models.slice.backbone import ResNetBackbone, DenseNetBackbone
from models.slice.head import GAPLinearClassifier, GAPLinearProjector, LinearEncoder, LinearDecoder
from models.slice.head import Classifier, TransformerEncoder


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
    classifier = GAPLinearClassifier(name=config.encoder_type, in_channels=out_dim, n_classes=2)

    # Return
    networks = dict(encoder=encoder, classifier=classifier)

    return networks


def build_networks_multi(config, **kwargs):

    # 1. Encoder
    if 'resnet' in config.encoder_type:
        encoder_mri = ResNetBackbone(name=config.encoder_type, in_channels=1)
        encoder_pet = ResNetBackbone(name=config.encoder_type, in_channels=1)
    elif 'densenet' in config.encoder_type:
        encoder_mri = DenseNetBackbone(name=config.encoder_type, in_channels=1)
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
    classifier = GAPLinearClassifier(name=config.encoder_type, in_channels=out_dim, n_classes=2)

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
    classifier_s = GAPLinearClassifier(name=config.encoder_type, in_channels=encoder_s_mri.out_channels, n_classes=2)

    if config.add_type == 'concat':
        out_dim = encoder_t_mri.out_channels + encoder_t_pet.out_channels
    else:
        assert encoder_t_mri.out_channels == encoder_t_pet.out_channels
        out_dim = encoder_t_mri.out_channels
    classifier_t = GAPLinearClassifier(name=config.encoder_type, in_channels=out_dim, n_classes=2)

    # Return
    networks = dict(encoder_s_mri=encoder_s_mri, encoder_t_mri=encoder_t_mri, encoder_t_pet=encoder_t_pet,
                    classifier_t=classifier_t)

    return networks


def build_networks_general_teacher(config, **kwargs):

    # 1. Extractor
    if 'resnet' in config.extractor_type:
        extractor_mri = ResNetBackbone(name=config.extractor_type, in_channels=1)
        extractor_pet = ResNetBackbone(name=config.extractor_type, in_channels=1)
    elif 'densenet' in config.extractor_type:
        extractor_mri = DenseNetBackbone(name=config.extractor_type, in_channels=1)
        extractor_pet = DenseNetBackbone(name=config.extractor_type, in_channels=1)
    else:
        raise ValueError

    if config.small_kernel:
        extractor_mri._fix_first_conv()
        extractor_pet._fix_first_conv()

    # 2. Projector
    if config.use_projector:
        projector_mri = GAPLinearProjector(name=config.extractor_type,
                                           in_channels=extractor_mri.out_channels,
                                           out_channels=config.hidden)
        projector_pet = GAPLinearProjector(name=config.extractor_type,
                                           in_channels=extractor_pet.out_channels,
                                           out_channels=config.hidden)
    else:
        projector_mri, projector_pet = None, None

    # 3. Encoder
    if config.use_projector:
        encoder_general = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                        act=config.encoder_act)
        encoder_mri = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                    act=config.encoder_act)
        encoder_pet = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                    act=config.encoder_act)
    else:
        assert extractor_mri.out_channels == extractor_pet.out_channels
        encoder_general = LinearEncoder(in_channels=extractor_mri.out_channels, out_channels=config.hidden,
                                        act=config.encoder_act)
        encoder_mri = LinearEncoder(in_channels=extractor_mri.out_channels, out_channels=config.hidden,
                                    act=config.encoder_act)
        encoder_pet = LinearEncoder(in_channels=extractor_pet.out_channels, out_channels=config.hidden,
                                    act=config.encoder_act)

    # 4. Decoder
    if config.use_projector:
        decoder_mri = LinearDecoder(in_channels=config.hidden // 2, out_channels=config.hidden)
        decoder_pet = LinearDecoder(in_channels=config.hidden // 2, out_channels=config.hidden)
    else:
        decoder_mri = LinearDecoder(in_channels=config.hidden, out_channels=extractor_mri.out_channels)
        decoder_pet = LinearDecoder(in_channels=config.hidden, out_channels=extractor_pet.out_channels)

    # 5. Classifier
    if config.use_transformer:
        transformer_encoder = TransformerEncoder(in_channels=config.hidden // 2)
    else:
        transformer_encoder = None

    if config.add_type == 'concat':
        if config.use_specific:
            classifier = Classifier(in_channels=config.hidden * 4 // 2, n_classes=2, mlp=config.mlp, dropout=config.dropout)
        else:
            classifier = Classifier(in_channels=config.hidden * 2 // 2, n_classes=2, mlp=config.mlp, dropout=config.dropout)

    if config.add_type == 'add':
        if config.use_projector:
            classifier = Classifier(in_channels=config.hidden // 2, n_classes=2, mlp=config.mlp, dropout=config.dropout)
        else:
            classifier = Classifier(in_channels=config.hidden, n_classes=2, mlp=config.mlp, dropout=config.dropout)

    networks = dict(extractor_mri=extractor_mri, extractor_pet=extractor_pet,
                    projector_mri=projector_mri, projector_pet=projector_pet,
                    encoder_general=encoder_general, encoder_mri=encoder_mri, encoder_pet=encoder_pet,
                    decoder_mri=decoder_mri, decoder_pet=decoder_pet,
                    transformer_encoder=transformer_encoder, classifier=classifier)

    return networks


def build_networks_student(config, **kwargs):

    # 1. Extractor
    if 'resnet' in config.extractor_type:
        extractor_mri_s = ResNetBackbone(name=config.extractor_type, in_channels=1)
    elif 'densenet' in config.extractor_type:
        extractor_mri_s = DenseNetBackbone(name=config.extractor_type, in_channels=1)
    else:
        raise ValueError

    if config.small_kernel:
        extractor_mri_s._fix_first_conv()

    # 2. Projector
    if config.use_projector:
        projector_mri_s = GAPLinearProjector(name=config.extractor_type,
                                             in_channels=extractor_mri_s.out_channels,
                                             out_channels=config.hidden)
    else:
        projector_mri_s = None

    # 3. Encoder
    if config.use_projector:
        encoder_mri_s = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden,
                                      act=config.encoder_act)
    else:
        encoder_mri_s = LinearEncoder(in_channels=extractor_mri_s.out_channels, out_channels=config.hidden,
                                      act=config.encoder_act)

    # 5. Classifier
    if config.use_transformer:
        transformer_encoder_s = TransformerEncoder(in_channels=config.hidden)
    else:
        transformer_encoder_s = None

    if config.use_specific:
        classifier_s = Classifier(in_channels=config.hidden * 4, n_classes=2, mlp=config.mlp, dropout=config.dropout)
    else:
        classifier_s = Classifier(in_channels=config.hidden * 2, n_classes=2, mlp=config.mlp, dropout=config.dropout)

    if config.add_type == 'add':
        classifier_s = Classifier(in_channels=config.hidden, n_classes=2, mlp=config.mlp, dropout=config.dropout)

    networks = dict(extractor_mri_s=extractor_mri_s, projector_mri_s=projector_mri_s,
                    encoder_mri_s=encoder_mri_s, transformer_encoder_s=transformer_encoder_s,
                    classifier_s=classifier_s)

    return networks


def build_networks_swap(config, **kwargs):

    # 1. Extractor
    if 'resnet' in config.extractor_type:
        extractor_mri = ResNetBackbone(name=config.extractor_type, in_channels=1)
        extractor_pet = ResNetBackbone(name=config.extractor_type, in_channels=1)
    elif 'densenet' in config.extractor_type:
        extractor_mri = DenseNetBackbone(name=config.extractor_type, in_channels=1)
        extractor_pet = DenseNetBackbone(name=config.extractor_type, in_channels=1)
    else:
        raise ValueError

    if config.small_kernel:
        extractor_mri._fix_first_conv()
        extractor_pet._fix_first_conv()

    # 2. Projector
    projector_mri = GAPLinearProjector(name=config.extractor_type,
                                       in_channels=extractor_mri.out_channels,
                                       out_channels=config.hidden)
    projector_pet = GAPLinearProjector(name=config.extractor_type,
                                       in_channels=extractor_pet.out_channels,
                                       out_channels=config.hidden)

    # 3. Encoder
    encoder_general_mri = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                        act=config.encoder_act)
    encoder_general_pet = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                        act=config.encoder_act)
    encoder_mri = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                act=config.encoder_act)
    encoder_pet = LinearEncoder(in_channels=config.hidden, out_channels=config.hidden // 2,
                                act=config.encoder_act)

    # 4. Decoder
    decoder_mri = LinearDecoder(in_channels=config.hidden // 2, out_channels=config.hidden)
    decoder_pet = LinearDecoder(in_channels=config.hidden // 2, out_channels=config.hidden)

    # 5. Classifier
    if config.add_type == 'add':
        classifier = Classifier(in_channels=config.hidden // 2, n_classes=2, mlp=config.mlp, dropout=config.dropout)
    else:
        classifier = Classifier(in_channels=config.hidden, n_classes=2, mlp=config.mlp, dropout=config.dropout)

    networks = dict(extractor_mri=extractor_mri, extractor_pet=extractor_pet,
                    projector_mri=projector_mri, projector_pet=projector_pet,
                    encoder_general_mri=encoder_general_mri, encoder_general_pet=encoder_general_pet,
                    encoder_mri=encoder_mri, encoder_pet=encoder_pet,
                    decoder_mri=decoder_mri, decoder_pet=decoder_pet,
                    classifier=classifier)

    return networks
