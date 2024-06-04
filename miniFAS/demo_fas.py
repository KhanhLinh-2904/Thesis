import cv2
import time
import numpy as np
import multiprocessing
import matplotlib.pyplot as plt

from function_model.SCI import LowLightEnhancer
from function_model.fas import FaceAntiSpoofing
from utils.custom_utils import detect_face, tracking


fas1_lowlight_path = "miniFAS/model_onnx/train_SCI_miniFAS/2.7_80x80_MiniFASNetV2.onnx"
fas2_lowlight_path = "miniFAS/model_onnx/train_SCI_miniFAS/4_0_0_80x80_MiniFASNetV1SE.onnx"
fas1_normal_path = "miniFAS/model_onnx/2.7_80x80_MiniFASNetV2.onnx"
fas2_normal_path = "miniFAS/model_onnx/4_0_0_80x80_MiniFASNetV1SE.onnx"
model_llie = 'miniFAS/model_onnx/SCI_old.onnx'

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

def camera(frame_fas, result_fas, frame_verify, result_verify):
    batch_face = []
    start_fas = False
    label, color = False, (0, 0, 0)

    

    # Create a VideoCapture object called cap
    cap = cv2.VideoCapture(0)

    new_gister = False
    # This is an infinite loop that will continue to run until the user presses the `q` key
    count_frame = 0
    while cap.isOpened():
        tic = time.time()

        # Read a frame from the webcam
        ret, frame_root = cap.read()
        # frame_root = cv2.flip(frame_root, 1)
        frame = frame_root.copy()
        is_lowlight = False

        # If the frame was not successfully captured, break out of the loop
        if ret is False:
            break

        threshold_img = lowlight_enhancer.get_threshold(frame)
        if threshold_img < under_threshold:
            continue
        elif threshold_img < over_threshold and threshold_img >= under_threshold:
            # frame = apply_fft_and_remove_noise(frame)
            frame = lowlight_enhancer.enhance(frame)  # RGB
            is_lowlight = True
    
        image_bbox, _ = detect_face(frame)

        if start_fas:
            batch_face.append((image_bbox, frame, is_lowlight))
            count_frame += 1
        
        if count_frame == 5:
            frame_fas.put(batch_face)
            count_frame = 0
            batch_face = []
            start_fas = False

        if not result_fas.empty():
            label = result_fas.get()

        if image_bbox is not None:
            new_gister = tracking(image_bbox, frame_root)

        if new_gister:
            start_fas = True
        
        test_speed = time.time() - tic
        fps = 1/test_speed

        if label and image_bbox is not None:
            if label == "REAL":
                color = (0, 255, 0)
            elif label == "FAKE":
                color = (0, 0, 255)
            cv2.rectangle(frame_root, (image_bbox[0], image_bbox[1]), (image_bbox[0] + image_bbox[2], image_bbox[1] + image_bbox[3]), color, 2)

        # Display the FPS on the frame
        cv2.putText(frame_root, f"FPS: {fps}", (30, 30), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 255), 2, cv2.LINE_AA)

        # Display the frame on the screen
        cv2.imshow("frame", frame_root)

        # Check if the user has pressed the `q` key, if yes then close the program.
        key = cv2.waitKey(1)
        if key == ord("q"):
            break

    # Release the VideoCapture object
    cap.release()

    # Close all open windows
    cv2.destroyAllWindows()

def anti_spoofing(frame_queue, result_queue):
    while True:
        real, fake = 0, 0

        # Get frame from the queue
        detections = frame_queue.get()

        for (bbox, frame, is_lowlight) in detections:
            frame = np.asarray(frame, dtype=np.uint8) 
            if is_lowlight:
                pred1 = fas1_lowlight.predict(frame)
                pred2 = fas2_lowlight.predict(frame)
            else:
                pred1 = fas1_normal.predict(frame)
                pred2 = fas2_normal.predict(frame)
            if pred1 is None or pred2 is None:
                continue
            prediction = pred1 + pred2
            output = np.argmax(prediction)

            if output == 1:
                real += 1
            else:
                fake += 1
        
        if real > fake:
            result_queue.put("REAL")
        else:
            result_queue.put("FAKE")

if __name__ == "__main__":
    frame_verify = multiprocessing.Queue()
    result_verify = multiprocessing.Queue()
    frame_fas = multiprocessing.Queue()
    result_fas = multiprocessing.Queue()

    p1 = multiprocessing.Process(name='p1', target=camera, args=(frame_fas, result_fas, frame_verify, result_verify))
    p = multiprocessing.Process(name='p', target=anti_spoofing, args=(frame_fas, result_fas))
    p.start()
    p1.start()
    p.join()
    p1.join()
    
