from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

IEM = Genotype(normal=[('skip_connect', 0), ('resconv_1x1', 1), ('resdilconv_3x3', 2), ('conv_3x3', 3), ('conv_3x3', 4), ('skip_connect', 5), ('conv_3x3', 6)], normal_concat=None, reduce=None, reduce_concat=None)
NRM = Genotype(normal=[('resconv_1x1', 0), ('resconv_3x3', 1), ('dilconv_3x3', 2), ('resconv_1x1', 3), ('resconv_3x3', 4), ('conv_3x3', 5), ('resconv_3x3', 6)], normal_concat=None, reduce=None, reduce_concat=None)
