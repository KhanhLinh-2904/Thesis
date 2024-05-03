import os
import cv2
import warnings
from tqdm import tqdm
import numpy as np

from function_model.fas import FaceAntiSpoofing

warnings.filterwarnings("ignore")
from matplotlib import pyplot as plt

# model_test = AntiSpoofPredict(0)
filePath = ""
model_1 = "miniFAS/model_onnx/new/2.7_80x80_MiniFASNetV2.onnx"
model_2 = "miniFAS/model_onnx/new/4_0_0_80x80_MiniFASNetV1SE.onnx"

if __name__ == "__main__":

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
