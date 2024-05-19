
import glob
import os
import time
import warnings
import numpy as np
import torch
import scripts.data_loading as dl
import tensorflow as tf
from model.arch import LYT, Denoiser
import matplotlib.pyplot as plt
from tqdm import tqdm
import cv2
import tensorflow as tf
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# warnings.filterwarnings("ignore")

def load_image_test(image_path, crop_margin):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)

    original_shape = tf.shape(img)
    new_height = original_shape[0] - 2 * crop_margin
    new_width = original_shape[1] - 2 * crop_margin

    img = tf.image.crop_to_bounding_box(img, crop_margin, crop_margin, new_height, new_width)
    img = (tf.cast(img, tf.float32) / 127.5) - 1.0
    return img

def build_model(weights_path):

    denoiser_cb = Denoiser(16)
    denoiser_cr = Denoiser(16)
    denoiser_cb.build(input_shape=(None,None,None,1))
    denoiser_cr.build(input_shape=(None,None,None,1))

    model = LYT(filters=32, denoiser_cb=denoiser_cb, denoiser_cr=denoiser_cr)
    model.build(input_shape=(None,None,None,3))

    # Loading weights
    model.load_weights(f'{weights_path}')
    return model

def save_images(tensor, save_path, image_name):
    image_numpy = np.round(tensor[0].numpy() * 255).astype(np.uint8)
    cv2.imwrite(os.path.join(save_path, image_name), cv2.cvtColor(image_numpy, cv2.COLOR_BGR2RGB))

def lowlight(img_path, model):
    img = dl.load_image_test(img_path,0)
    # print("shape of image : ", img.shape)

    img = tf.expand_dims(img, axis=0)
    # print("-------------------- shape image: ", img.shape)
    start = time.time()
    generated_image = model(img)
    end_time = (time.time() - start)
    return generated_image, end_time
    
if __name__ == '__main__':
   # Build model
    

    with torch.no_grad():
        
        filePath = '/home/user/Thesis/Zero-DCE++/data/SICE/SICE_Part2'
        save_path = 'data/result_Test_Part2'
        file_list = os.listdir(filePath)
        sum_time = 0

        model = build_model('pretrained_weights/LOLv1.h5')

        for file_name in tqdm(file_list):
            path_to_image = os.path.join(filePath, file_name)
            # img = load_image_test(path_to_image, 0)
            inference_image, end_time = lowlight(path_to_image, model)
            inference_image = (inference_image + 1.0) / 2.0
            save_images(inference_image, save_path, file_name)
            sum_time = sum_time + end_time
        print(sum_time)