from .part_enhance import *
from .part_detection import *


class Network(nn.Module):

    def __init__(self, phase, num_classes, genotype):
        super(Network, self).__init__()

        self.phase = phase
        self.genotype = genotype
        self.num_classes = num_classes
        self.iem_nums = 3
        self.enhance_channel = 3
        self.enhance_net = EnhanceNetwork(iteratioin=self.iem_nums, channel=self.enhance_channel,
                                          genotype=self.genotype)

        self.detection_net = DetectionNetwork(self.phase, self.num_classes, self.genotype)
        self.constant = torch.Tensor([123, 117, 104]).unsqueeze(0).unsqueeze(2).unsqueeze(3)

    def forward(self, inputs, targets):
        new_input = (inputs + self.constant) / 255
        u_list, t_list = self.enhance_net(new_input)
        detect_img = u_list[-1] * 255 - self.constant

        out = self.detection_net(detect_img, targets)

        return u_list, t_list, out
