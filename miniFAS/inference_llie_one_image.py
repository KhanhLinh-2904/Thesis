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

from function_model.llie import LowLightEnhancer

filePath = ''	
image_name = ''
savePath = ''
threshold = 100
scale_factor = 1
model_onnx = 'miniFAS/model_onnx/ZeroDCE++new.onnx'
if __name__ == '__main__':

    with torch.no_grad():
        print("file_name:",filePath)
        lowlight_enhancer = LowLightEnhancer(scale_factor=1, model_onnx=model_onnx)
        img = cv2.imread(filePath) 
        img = img[:, :, ::-1]
        # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if lowlight_enhancer.is_lowlight(img,threshold):
            img = lowlight_enhancer.enhance(img)
        img = img[:, :, ::-1]
        result_path = os.path.join(savePath, image_name)
        cv2.imwrite(result_path, img)
       
