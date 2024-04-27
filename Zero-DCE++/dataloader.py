import os
import sys
import PIL

import torch
import torch.utils.data as data

import numpy as np
from PIL import Image
import glob
import cv2
import random
from torch.utils.data import random_split


generator = torch.Generator()
generator.manual_seed(1143)
random.seed(1143)

import warnings
warnings.filterwarnings('ignore')

def populate_train_list(lowlight_images_path):


	# image_list_lowlight = glob.glob(lowlight_images_path + "*.jpg")
	image_list_lowlight = glob.glob(lowlight_images_path + '/*.jpg')
	train_list = image_list_lowlight
	random.shuffle(train_list)

	return train_list

def remove_noise(image):
   blurred_img = cv2.fastNlMeansDenoisingColored(image, None, 3,3,7,21)
   return blurred_img	

class lowlight_loader(data.Dataset):

	def __init__(self, dataset):

		self.train_list = dataset
		self.size = 512

		self.data_list = self.train_list
		print("Total training examples:", len(self.train_list))


	def __getitem__(self, index):
		data_lowlight_path = self.data_list[index]
		
		data_lowlight = Image.open(data_lowlight_path)
		# data_lowlight = remove_noise(data_lowlight)

		data_lowlight = data_lowlight.resize((self.size,self.size), PIL.Image.Resampling.LANCZOS)
		data_lowlight = (np.asarray(data_lowlight)/255.0) 
		data_lowlight = torch.from_numpy(data_lowlight).float()

		return data_lowlight.permute(2,0,1)

	def __len__(self):
		return len(self.data_list)

