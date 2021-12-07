# -*- coding:utf-8 -*-
import os
import time
import torch
import argparse
import numpy as np
from torch.autograd import Variable
import torch.backends.cudnn as cudnn
import cv2
from data.config import cfg
from PIL import Image
from utils import to_chw_bgr
from model_detection.genotypes import *
from model_detection.model_train_coor import Network

parser = argparse.ArgumentParser(description='demo')
parser.add_argument('--data_path', type=str, default='./data/detection_test_data',
                    help='location of the data corpus')
parser.add_argument('--save_dir',
                    type=str, default='result/detection',
                    help='Directory for detect result')
parser.add_argument('--model',
                    type=str,
                    default='weights/detection.pth',
                    help='trained model')
parser.add_argument('--thresh',
                    default=0.4, type=float,
                    help='Final confidence threshold')

args = parser.parse_args()

if not os.path.exists(args.save_dir):
    os.makedirs(args.save_dir)

use_cuda = torch.cuda.is_available()

if use_cuda:
    torch.set_default_tensor_type('torch.cuda.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')


def detect(net, img_path, thresh):
    img = Image.open(img_path)
    if img.mode == 'L':
        img = img.convert('RGB')

    img = np.array(img)
    height, width, _ = img.shape
    max_im_shrink = np.sqrt(
        1500 * 1000 / (img.shape[0] * img.shape[1]))
    image = cv2.resize(img, None, None, fx=max_im_shrink,
                       fy=max_im_shrink, interpolation=cv2.INTER_LINEAR)

    x = to_chw_bgr(image)
    x = x.astype('float32')
    x -= cfg.img_mean
    x = x[[2, 1, 0], :, :]

    x = Variable(torch.from_numpy(x).unsqueeze(0))
    if use_cuda:
        x = x.cuda()

    targets = 0
    u_list, t_list, y = net(x, targets)
    detections = y.data
    scale = torch.Tensor([img.shape[1], img.shape[0],
                          img.shape[1], img.shape[0]])

    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    for i in range(detections.size(1)):
        j = 0
        while detections[0, i, j, 0] >= thresh:
            score = detections[0, i, j, 0]
            pt = (detections[0, i, j, 1:] * scale).cpu().numpy().astype(int)

            left_up, right_bottom = (pt[0], pt[1]), (pt[2], pt[3])
            j += 1
            cv2.rectangle(img, left_up, right_bottom, (0, 0, 255), 2)
            conf = "{:.6}".format(score)
            text_size, baseline = cv2.getTextSize(
                conf, cv2.FONT_HERSHEY_SIMPLEX, 0.3, 1)

            p1 = (left_up[0], left_up[1] - text_size[1])
            cv2.rectangle(img, (p1[0] - 2 // 2, p1[1] - 2 - baseline),
                          (p1[0] + text_size[0], p1[1] + text_size[1]), [255, 0, 0], -1)

            cv2.putText(img, conf, (p1[0], p1[
                1] + baseline), cv2.FONT_HERSHEY_SIMPLEX, 0.3, (255, 255, 255), 1, 8)

    print('detect:{}'.format(img_path))

    save_img_path = args.save_dir
    os.makedirs(save_img_path, exist_ok=True)
    cv2.imwrite(os.path.join(save_img_path, os.path.basename(img_path)), img)


if __name__ == '__main__':
    net = Network('test', 2, genotype=vgg_multibox_freeze_backbone)
    net.load_state_dict(torch.load(args.model))
    net.eval()

    if use_cuda:
        net.cuda()
        cudnn.benckmark = True

    img_list = [os.path.join(args.data_path, x)
                for x in os.listdir(args.data_path) if x.endswith('png')]

    with torch.no_grad():
        for path in img_list:
            detect(net, path, args.thresh)
