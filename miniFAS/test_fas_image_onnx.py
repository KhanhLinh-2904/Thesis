import os
import cv2
import warnings
import time
from tqdm import tqdm
import numpy as np
import pandas as pd

from function_model.fas import FaceAntiSpoofing
from function_model.llie import LowLightEnhancer

warnings.filterwarnings("ignore")

dataset = "miniFAS/datasets/Test/test"
fas1_lowlight_path = "miniFAS/model_onnx/new_combine/2.7_80x80_MiniFASNetV2.onnx"
fas2_lowlight_path = "miniFAS/model_onnx/new_combine/4_0_0_80x80_MiniFASNetV1SE.onnx"
fas1_normal_path = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
fas2_normal_path = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/ZeroDCE++scale12.onnx'

under_threshold = 8
over_threshold = 100
scale_factor = 12
fas1_lowlight = FaceAntiSpoofing(fas1_lowlight_path)
fas2_lowlight = FaceAntiSpoofing(fas2_lowlight_path)
fas1_normal = FaceAntiSpoofing(fas1_normal_path)
fas2_normal = FaceAntiSpoofing(fas2_normal_path)
lowlight_enhancer = LowLightEnhancer(scale_factor=scale_factor, model_onnx=model_llie)

def apply_fft_and_remove_noise(image):
    #Return multidimensional discrete Fourier transform.
   blurred_img = cv2.fastNlMeansDenoisingColored(image, None, 3,3,7,21)
   return blurred_img

if __name__ == "__main__":
   
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    count_none_face = 0
    count_llie = 0
    count_undefined = 0
    
   

    dir_label = os.listdir(dataset)
    for label in dir_label:
        dir_img = os.path.join(dataset, label)
        images = os.listdir(dir_img)

        print("len folder " + label + ": ", len(images))
        TIME_START = time.time()
        for image in tqdm(images):
            prediction = np.zeros((1, 3))
            img_path = os.path.join(dir_img, image)
            img = cv2.imread(img_path)  # BGR
            threshold_img = lowlight_enhancer.get_threshold(img)
            if threshold_img < under_threshold:
                count_undefined += 1
            elif threshold_img < over_threshold and threshold_img >= under_threshold:
                img = apply_fft_and_remove_noise(img)
                img = lowlight_enhancer.enhance(img[:, :, ::-1])  
                img = img[:, :, ::-1]
                count_llie += 1
                pred1 = fas1_lowlight.predict(img)
                pred2 = fas2_lowlight.predict(img)
               
            else:
                pred1 = fas1_normal.predict(img)
                pred2 = fas2_normal.predict(img)
            
            
            if  pred1 is None or pred2 is None:
                count_none_face += 1    
            else:
                prediction = pred1 + pred2
                output = np.argmax(prediction)
                if output != 1 and label == "fake":
                    tp += 1
                elif output == 1 and label == "fake":
                    fn += 1

                elif output != 1 and label == "real":
                    fp += 1
                    
                elif output == 1 and label == "real":
                    tn += 1

   
    print("tp:", tp)
    print("fp:", fp)
    print("fn:", fn)
    print("tn:", tn)
    print("count low light: ", count_llie)
    print("count none face: ", count_none_face)
    print('count undefined: ', count_undefined)
    print("time: ", time.time() - TIME_START)
    