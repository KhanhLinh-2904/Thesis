import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt
import os
from tqdm import tqdm

from function_model.fas import FaceAntiSpoofing
from function_model.llie import LowLightEnhancer
from utils.custom_utils import detect_face, tracking

dataset = "miniFAS/datasets/Test/low-light-face-video-Hiep"

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

def camera(video_path):
    frame_fas = []
    cap = cv2.VideoCapture(video_path)
    # frame_number = 12
    # cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number - 1)
    count_frame = 0
    num_frames = 0
    while cap.isOpened():
       
        ret, frame = cap.read()
        is_lowlight = False

        num_frames += 1
        if ret is False:
            break
        threshold_img = lowlight_enhancer.get_threshold(frame)
        if threshold_img < under_threshold:
            continue
        elif threshold_img < over_threshold and threshold_img >= under_threshold:
            frame = apply_fft_and_remove_noise(frame)
            frame = lowlight_enhancer.enhance(frame[:, :, ::-1])  # RGB
            frame = frame[:, :, ::-1]
            is_lowlight = True

        image_bbox, _ = detect_face(frame)
        if image_bbox is not None:
            new_gister = tracking(image_bbox, frame)
        else:
            new_gister = False
            continue
        if new_gister and image_bbox is not None:
            frame_fas.append([frame, is_lowlight])
            count_frame += 1
        
        if count_frame == 5:
            break
            
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    cap.release()
    return frame_fas, num_frames

def anti_spoofing(frame_fas):
    while True:
        real, fake = 0, 0

        detections = frame_fas

        for [frame, is_lowlight] in detections:
            frame = np.asarray(frame, dtype=np.uint8) 
            if is_lowlight:
                prediction = fas1_lowlight.predict(frame) + fas2_lowlight.predict(frame)
            else:
                prediction = fas1_normal.predict(frame) + fas2_normal.predict(frame)
            output = np.argmax(prediction)

            if output == 1:
                real += 1
            else:
                fake += 1
        
        if real > fake:
           return 'REAL'
        else:
            return 'FAKE'

    
if __name__ == "__main__":
    tp = 0
    tn = 0
    fn = 0
    fp = 0
    label = 'REAL'
    total_frames = 0
    
    videos = os.listdir(dataset)
    print("len folder: ", len(videos))
    start_time = time.time()
    for video in videos:
        video_path = os.path.join(dataset, video)
        print("video_path: ",video_path)
       
        frame_fas, num_frames = camera(video_path)
        total_frames += num_frames
        output = anti_spoofing(frame_fas)
        print('output: ', output)
        print('--------------------------------------------------------------------------------------')

        if output == "FAKE" and label == 'FAKE':
            tp += 1
        elif output == 'REAL' and label == 'FAKE':
            fn += 1
        elif output == 'FAKE' and label == 'REAL':
            fp += 1
        elif output == 'REAL' and label == 'REAL':
            tn += 1  
    total_time = time.time() - start_time
    
    print('--------------------------------------------------------------------------------------')
    print('fusion matrix')
    print('tp: ', tp)
    print('tn: ', tn)
    print('fp: ', fp)
    print('fn: ', fn)
    print('Total time: ', total_time)
    print('Total frames: ', total_frames)
