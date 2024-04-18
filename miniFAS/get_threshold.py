import warnings
warnings.filterwarnings('ignore')
import os
import cv2
from tqdm import tqdm
dataset = 'miniFAS/datasets/Train/train/2.7_80x80'
def get_threshold(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_intensity = cv2.mean(gray_image)[0]
        return average_intensity

if __name__ == "__main__":
    sum_intensity = 0
    len_data = 0
    labels = os.listdir(dataset)
    for label in labels:
        images_path = os.path.join(dataset, label)
        images = os.listdir(images_path)
        len_data += len(images)
        for image in tqdm(images):
            img_path = os.path.join(images_path, image)
            img = cv2.imread(img_path)
            sum_intensity += get_threshold(img)
    average_threshold = sum_intensity/len_data
    print(f"Average threshold: {average_threshold:.5f}")
#     images = os.listdir(dataset)
#     len_data = len(images)
#     print('len :', len_data)
#     for image in tqdm(images):
#         img_path = os.path.join(dataset, image)
#         img = cv2.imread(img_path)
#         sum_intensity += get_threshold(img)
# average_threshold = sum_intensity/len_data
# print(f"Average threshold: {average_threshold:.5f}")
   