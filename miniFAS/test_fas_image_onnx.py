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

dataset = "miniFAS/datasets/Test/Test_LLIE_FAS"
model_1 = "miniFAS/model_onnx/new/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/new/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/Zero_DCE++new.onnx'
under_threshold = 8
over_threshold = 100
scale_factor = 12
if __name__ == "__main__":
   
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    count_img = 0
    count_none_face = 0
    count_llie = 0
    count_undefined = 0
    fas_model1 = FaceAntiSpoofing(model_1)
    fas_model2 = FaceAntiSpoofing(model_2)
    lowlight_enhancer = LowLightEnhancer(scale_factor=scale_factor, model_onnx=model_llie)
   

    dir_label = os.listdir(dataset)
    for label in dir_label:
        dir_img = os.path.join(dataset, label)
        images = os.listdir(dir_img)

        print("len folder " + label + ": ", len(images))
        TIME_START = time.time()
        for image in tqdm(images):
            count_img += 1
            prediction = np.zeros((1, 3))
            img_path = os.path.join(dir_img, image)
            img = cv2.imread(img_path)  # BGR
            img_real = img
            threshold_img = lowlight_enhancer.get_threshold(img)
            if threshold_img < under_threshold:
                count_undefined += 1
                continue
            elif threshold_img < over_threshold and threshold_img >= under_threshold:
                    img = lowlight_enhancer.enhance(img[:, :, ::-1])  # RGB
                    img = img[:, :, ::-1]
                    count_llie += 1
            pred1 = fas_model1.predict(img)
            pred2 = fas_model2.predict(img)
            if  pred1 is None or pred2 is None:
                count_none_face += 1
                continue
          
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
    