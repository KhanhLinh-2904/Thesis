import os
import sys
import warnings
import numpy as np
import torch
import argparse
import torch.utils
import torch.backends.cudnn as cudnn
from PIL import Image
from torch.autograd import Variable
from tqdm import tqdm
from model import Finetunemodel
import time

from multi_read_data import MemoryFriendlyLoader
warnings.filterwarnings("ignore")


parser = argparse.ArgumentParser("SCI")
parser.add_argument('--data_path', type=str, default='Zero-DCE++/data/SICE/SICE_Part2',
                    help='location of the data corpus')
parser.add_argument('--save_path', type=str, default='SCI/data/finetune/result_Test_Part2', help='location of the data corpus')
parser.add_argument('--model', type=str, default='SCI/results/new_finetune/Epoch91.pth', help='location of the data corpus')
# parser.add_argument('--gpu', type=int, default=0, help='gpu device id')
# parser.add_argument('--seed', type=int, default=2, help='random seed')

args = parser.parse_args()
save_path = args.save_path
os.makedirs(save_path, exist_ok=True)

TestDataset = MemoryFriendlyLoader(img_dir=args.data_path, task='test')

test_queue = torch.utils.data.DataLoader(
    TestDataset, batch_size=1,
    pin_memory=True, num_workers=0)


def save_images(tensor, path):
    # image_numpy = tensor[0].cpu().float().numpy()
    image_numpy = tensor[0].float().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)))
    im = Image.fromarray(np.clip(image_numpy * 255.0, 0, 255.0).astype('uint8'))
    im.save(path, 'png')


def main():
    # if not torch.cuda.is_available():
    #     print('no gpu device available')
    #     sys.exit(1)

    model = Finetunemodel(args.model)
    # model = model.cuda()
    sum_time = 0
    model.eval()
    with torch.no_grad():
        for _, (input, image_name) in tqdm(enumerate(test_queue)):
            # input = Variable(input, volatile=True).cuda()
            input = Variable(input, volatile=True)
            # print("input: ", input.shape)
            # print("input name before pro: ", image_name[0].split('/')[-1].split('.')[0])
            image_name = image_name[0].split('/')[-1].split('.')[0]
            # print("image_name",image_name)
            start_time = time.time()
            i, r = model(input)
            sum_time  = sum_time + (time.time() - start_time)
            u_name = '%s.png' % (image_name)
            # print('processing {}'.format(u_name))
            u_path = save_path + '/' + u_name
            save_images(r, u_path)

    print(sum_time)

if __name__ == '__main__':
    main()
