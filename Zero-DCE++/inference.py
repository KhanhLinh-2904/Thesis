import torch
import torch.optim
import os
import time
import time
import cv2
from tqdm import tqdm
from lowlight_enhance import LowLightEnhancer
model_onnx= 'Zero-DCE++/ZeroDCE++99.onnx'
dataset = 'Zero-DCE++/data/FAS_Thuan/train'	
save_path = 'Zero-DCE++/data/FAS_Thuan/result_train_100'
scale_factor = 12

def lowlight(img, image_name, save_path, model):
	start = time.time()
	img_enhance = model.enhance(img)
	img_enhance = img_enhance[:, :, ::-1]
	end_time = (time.time() - start)
	result_path = os.path.join(save_path, image_name)
	cv2.imwrite(result_path, img_enhance)
	return end_time
# inference folder
if __name__ == '__main__':

	with torch.no_grad():

		file_list = os.listdir(dataset)
		sum_time = 0
		threshold = 100
		lowlight_enhance = LowLightEnhancer(scale_factor, model_onnx)
		for file_name in tqdm(file_list):
			path_to_image = os.path.join(dataset, file_name)
			img = cv2.imread(path_to_image)
			img = img[:, :, ::-1]
			# if lowlight_enhance.is_lowlight(img, threshold):
			sum_time = sum_time + lowlight(img, file_name, save_path, lowlight_enhance)
		print(sum_time)
		
