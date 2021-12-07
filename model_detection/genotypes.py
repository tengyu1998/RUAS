from collections import namedtuple

Genotype = namedtuple('Genotype', 'enhance_op detection_op')

vgg_multibox_freeze_backbone = Genotype(
    enhance_op=[('skip_connect', 0), ('conv_1x1', 1), ('resconv_1x1', 2),
                ('resconv_1x1', 3), ('resconv_1x1', 4), ('resconv_1x1', 5),
                ('resconv_3x3', 6)],
    detection_op=[
        'conv_7x7', 'conv_7x7', 'conv_7x7', 'conv_3x3', 'conv_7x7',
        'dilconv_3x3', 'dilconv_3x3', 'dilconv_3x3', 'dilconv_3x3', 'conv_3x3',
        'dilconv_3x3', 'conv_7x7', 'conv_7x7', 'conv_5x5', 'dilconv_5x5',
        'conv_7x7'
    ])
