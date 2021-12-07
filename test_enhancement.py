import sys
import os
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from model_enhance.model import Network
from data.enhance_data_read import MemoryFriendlyLoader


parser = argparse.ArgumentParser("ruas")
parser.add_argument('--data_path', type=str, default='./data/enhance_test_data/lol/test',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='./result/lol', help='location of the data corpus')
parser.add_argument('--model', type=str, default='weights/lol.pt', help='location of the data corpus')
parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()

os.makedirs(args.save_path, exist_ok=True)
test_low_data_names = args.data_path + '/*.png'


TestDataset = MemoryFriendlyLoader(img_dir=test_low_data_names, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    if not torch.cuda.is_available():
        print('no gpu device available')
        sys.exit(1)

    np.random.seed(args.seed)
    torch.cuda.set_device(args.gpu)
    cudnn.benchmark = True
    torch.manual_seed(args.seed)
    cudnn.enabled = True
    torch.cuda.manual_seed(args.seed)
    print('gpu device = %d' % args.gpu)
    print("args = %s", args)

    model = Network()
    model = model.cuda()
    model_dict = torch.load(args.model)
    model.load_state_dict(model_dict)

    model.eval()

    with torch.no_grad():
        for _, (input, image_name) in enumerate(test_queue):
            input = Variable(input).cuda()
            image_name = image_name[0].split('.')[0]
            u_list, r_list, _ = model(input)

            u_name = '%s.png' % (image_name)
            u_path = args.save_path + '/' + u_name
            print('processing {}'.format(u_name))
            save_images(u_list[-1], u_path)


if __name__ == '__main__':
    main()
