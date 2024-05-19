
import torch
import torch.optim
import cv2
import os
import warnings
from tqdm import tqdm
import numpy as np
# from function_model.Zero_DCE import LowLightEnhancer
from function_model.SCI import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing

warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt
threshold = 100
scale_factor = 12
model_onnx = 'miniFAS/model_onnx/SCI.onnx'


def predict_one_image():
    filePath = "miniFAS/datasets/Test/FAS_Thuan/fake/IMG_0357.jpg"
    model_1 = "miniFAS/model_onnx/new/2.7_80x80_MiniFASNetV2.onnx"
    model_2 = "miniFAS/model_onnx/new/4_0_0_80x80_MiniFASNetV1SE.onnx"
    fas_model1 = FaceAntiSpoofing(model_1)
    fas_model2 = FaceAntiSpoofing(model_2)

    prediction = np.zeros((1, 3))
    img = cv2.imread(filePath)
    res1 = fas_model1.predict(img)
    res2 = fas_model2.predict(img)
    if res1 is None or res2 is None:
        print("There is no face here!")
    else:
        prediction = fas_model1.predict(img) + fas_model2.predict(img)
        label = np.argmax(prediction)
        print(label)
        if label == 1:
            print("Real")
        else:
            print("Fake")

def enhance_one_image():
    filePath = 'miniFAS/1_1.jpg'	
    image_name = 're2.jpg'
    savePath = 'miniFAS/'

    with torch.no_grad():
        print("file_name:",filePath)
        lowlight_enhancer = LowLightEnhancer(scale_factor=scale_factor, model_onnx=model_onnx)
        img = cv2.imread(filePath) 
        if lowlight_enhancer.is_lowlight(img,threshold):
            img = lowlight_enhancer.enhance(img)

        result_path = os.path.join(savePath, image_name)
        cv2.imwrite(result_path, img)

def enhance_folder():
    filePath = 'miniFAS/datasets/Test/new_dataset/train/0'	
    savePath = 'miniFAS/datasets/Test/train_no/0'
    
    with torch.no_grad():
        file_list = os.listdir(filePath)
        sum_time = 0
        len_llie = 0
        lowlight_enhancer = LowLightEnhancer(scale_factor=scale_factor, model_onnx=model_onnx)
        print("len folder: ",len(file_list))
        for file_name in tqdm(file_list):
            path_to_image = os.path.join(filePath, file_name)
            img = cv2.imread(path_to_image)
            if lowlight_enhancer.get_threshold(img) < threshold:
                img = lowlight_enhancer.enhance(img)
                len_llie += 1
            # img = lowlight_enhancer.enhance(img)
            result_path = os.path.join(savePath, file_name)
            cv2.imwrite(result_path, img)
        print(len_llie)
            
       
if __name__ == '__main__':
    # predict_one_image()
    # enhance_one_image()
    enhance_folder()