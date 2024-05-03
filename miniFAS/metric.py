import os
import shutil
import cv2
import cv2 as cv
import numpy as np
import argparse
import warnings
import time
from tqdm import tqdm
import torch

from function_model.fas import FaceAntiSpoofing
from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name

warnings.filterwarnings('ignore')
model_test = AntiSpoofPredict(0)
dataset ="/home/user/low_light_enhancement/Zero-DCE++/data/FAS_Thuan/groundtruth"
model_1 =''
model_2 = ''
model_1_path = 'miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx'
model_2_path = 'miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx'
model1_new_path = 'miniFAS/model_onnx/new/2.7_80x80_MiniFASNetV2.onnx'
model2_new_path = 'miniFAS/model_onnx/new/4_0_0_80x80_MiniFASNetV1SE.onnx'
fas_model1 = FaceAntiSpoofing(model_1_path)
fas_model2 = FaceAntiSpoofing(model_2_path)
fas_model1_new = FaceAntiSpoofing(model1_new_path)
fas_model2_new = FaceAntiSpoofing(model2_new_path)
def cal_mean_abs_diff(prediction_1, prediction_2):
    
    diff = cv2.absdiff(prediction_1, prediction_2)
    # print("diff: ", diff)
    mean_absolute_difference = np.mean(diff)
    return mean_absolute_difference


def predict_two_model_pytorch(image):
    model_test = AntiSpoofPredict(0)
    image_cropper =  CropImage()
    image_bbox, conf = model_test.get_bbox(image)
    if conf < 0.7:
        return "none"
    prediction = np.zeros((1,3))
    for model in [ model_1, model_2]:
        model_name = model.split("/")[-1]
        h_input, w_input, model_type, scale = parse_model_name(model_name)
        param = {
            "org_img": image,
            "bbox": image_bbox,
            "scale": scale,
            "out_w": w_input,
            "out_h": h_input,
            "crop": True,
        }
        if scale is None:
            param["crop"] = False
        img = image_cropper.crop(**param)[0]

        prediction += model_test.predict(img, model)
    return prediction
    
def predict_onnx(image, model1, model2):
    prediction = np.zeros((1, 3))
    pred1 = model1.predict(image)
    pred2 = model2.predict(image)
    if pred1 is None or pred2 is None:
        return None
    prediction = pred1+ pred2
    return prediction
    
if __name__ == "__main__":
    
    mean_absolute_differences = []
    images = os.listdir(dataset)
    print("len folder: ", len(images))
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
       
        prediction_old = predict_onnx(img, fas_model1, fas_model2)
        prediction_new = predict_onnx(img, fas_model1_new, fas_model2_new)
        if prediction_old is None or prediction_new is None:
            continue
        maf_value = cal_mean_abs_diff(prediction_old, prediction_new)
        # print(type(maf_value))
        mean_absolute_differences.append(maf_value)
        # print(mean_absolute_differences)
    overall_mean_absolute_difference = np.mean(mean_absolute_differences)
    print(f"Overall Mean Absolute Difference: {overall_mean_absolute_difference:.5f}")
