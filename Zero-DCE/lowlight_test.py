import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time

from tqdm import tqdm
import dataloader
import model
import numpy as np
from torchvision import transforms
from PIL import Image
import glob
import time


 
def lowlight(file_path, file_name, save_path):
	# os.environ['CUDA_VISIBLE_DEVICES']='0'
	data_lowlight = Image.open(file_path)

 

	data_lowlight = (np.asarray(data_lowlight)/255.0)


	data_lowlight = torch.from_numpy(data_lowlight).float()
	data_lowlight = data_lowlight.permute(2,0,1)
	# data_lowlight = data_lowlight.cuda().unsqueeze(0)
	data_lowlight = data_lowlight.unsqueeze(0)


	# DCE_net = model.enhance_net_nopool().cuda()
	DCE_net = model.enhance_net_nopool()

	DCE_net.load_state_dict(torch.load('Zero-DCE/snapshots/Epoch99.pth'))
	start = time.time()
	_,enhanced_image,_ = DCE_net(data_lowlight)

	end_time = (time.time() - start)
	# print(end_time)
	result_path = os.path.join(save_path, file_name)
	torchvision.utils.save_image(enhanced_image, result_path)
	return end_time

if __name__ == '__main__':
# test_images
	with torch.no_grad():
		filePath = 'Zero-DCE++/data/SICE/SICE_Part2'
		save_path = 'Zero-DCE/data/result_Test_Part2'
		file_list = os.listdir(filePath)
		sum_time = 0
		for file_name in tqdm(file_list):
			file_path = os.path.join(filePath, file_name)
			sum_time = sum_time + lowlight(file_path, file_name, save_path)

		print(sum_time)

		

