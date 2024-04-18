import shutil
import warnings

from tqdm import tqdm
warnings.filterwarnings('ignore')
from src.anti_spoof_predict import Detection
import os
import cv2
dataset = 'Zero-DCE++/data/FAS_Thuan/result_train_100'
# train_path = '/home/user/low_light_enhancement/Zero-DCE++/data/FAS_Thuan/train'
new_train_path = 'miniFAS/datasets/Test/train_87'
def get_confidence(image):
    face_detection = Detection()
    image_bbox, conf = face_detection.get_bbox(image)
    return conf

if __name__ == "__main__":
    sum_conf = 0
    len_data = 0
    # labels = os.listdir(dataset)
    # for label in labels:
    #     images_path = os.path.join(dataset, label)
    #     images = os.listdir(images_path)
    #     len_data += len(images)
    #     for image in tqdm(images):
    #         img_path = os.path.join(images_path, image)
    #         img = cv2.imread(img_path)
    #         sum_conf += get_bbox_face(img)
    # average_conf = sum_conf/len_data
    # print(f"Average confidence: {average_conf:.5f}")
    images = os.listdir(dataset)
    
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        conf = get_confidence(img)
        # print(conf)
    #     if conf >= 0.87:
    #         sum_conf += conf
    #         len_data += 1
    # average_conf = sum_conf/len_data
    # print(f"Average confidence: {average_conf:.5f}")
        if conf >= 0.87:
            len_data += 1
            src_path = os.path.join(dataset, image)
            desc_path = os.path.join(new_train_path, image)
            shutil.copy(src_path, desc_path)
    print(len_data)

