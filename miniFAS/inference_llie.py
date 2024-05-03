import onnxruntime
import torch
import torch.nn as nn
import torch.nn.functional as F 
import os
import torch.backends.cudnn as cudnn
import torch.optim
import numpy as np
import torchvision 
from PIL import Image
import cv2
from matplotlib import pyplot as plt
from torchvision import  utils
from tqdm import tqdm

from function_model.llie import LowLightEnhancer

filePath = ''	
savePath = ''
threshold = 100
scale_factor = 1
model_onnx = 'miniFAS/model_onnx/Zero_DCE++new.onnx'
if __name__ == '__main__':

    with torch.no_grad():
        file_list = os.listdir(filePath)
        sum_time = 0
        lowlight_enhancer = LowLightEnhancer(scale_factor=scale_factor, model_onnx=model_onnx)
        print("len folder: ",len(file_list))
        for file_name in tqdm(file_list):
            path_to_image = os.path.join(filePath, file_name)
            img = cv2.imread(path_to_image)
            img = img[:, :, ::-1]
            # if lowlight_enhancer.is_lowlight(img,threshold):
                # img = lowlight_enhancer.enhance(img)
            img = lowlight_enhancer.enhance(img)
            img = img[:, :, ::-1]
            result_path = os.path.join(savePath, file_name)
            cv2.imwrite(result_path, img)
            
       
        