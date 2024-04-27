import shutil
import warnings

from tqdm import tqdm
warnings.filterwarnings('ignore')
from src.anti_spoof_predict import Detection
import os
import cv2
from src.anti_spoof_predict import AntiSpoofPredict, Detection
from src.generate_patches import CropImage
from src.utility import parse_model_name

def get_confidence(image):
    face_detection = Detection()
    image_bbox, conf = face_detection.get_bbox(image)
    return conf

def get_threshold(image):
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        average_intensity = cv2.mean(gray_image)[0]
        return average_intensity

def apply_fft_and_remove_noise(image):
    #Return multidimensional discrete Fourier transform.
   blurred_img = cv2.fastNlMeansDenoisingColored(image, None, 3,3,7,21)
   return blurred_img

def crop_image(image, model_name, model_test, image_cropper):
    image_bbox, _ = model_test.get_bbox(image)
    if image_bbox is not None:
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
        img = image_cropper.crop(**param)
    return img

def calculate_conf_face():
    dataset = 'Zero-DCE++/data/result_train_remove_noise'
    # train_path = '/home/user/low_light_enhancement/Zero-DCE++/data/FAS_Thuan/train'
    new_train_path = 'miniFAS/datasets/Test/train'
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
    #     print(conf)
    #     if conf < 0.5:
    #         sum_conf += conf
    #         len_data += 1
    # average_conf = sum_conf/len_data
    # print(f"Average confidence: {average_conf:.5f}")
        if conf >= 0.7:
            len_data += 1
            src_path = os.path.join(dataset, image)
            desc_path = os.path.join(new_train_path, image)
            shutil.copy(src_path, desc_path)
    print(len_data)

def calculate_threshold():
    dataset = 'miniFAS/datasets/Test/2'
    save_llie_path = 'miniFAS/datasets/Test/train/2'
    # save_dark = 'miniFAS/datasets/Test/Dark_Dataset/2'
    sum_intensity = 0
    len_data = 0
    #  for label #
    labels = os.listdir(dataset)
    # for label in labels:
    #     images_path = os.path.join(dataset, label)
    #     images = os.listdir(images_path)
    #     len_data += len(images)
    #     for image in tqdm(images):
    #         img_path = os.path.join(images_path, image)
    #         img = cv2.imread(img_path)
    #         sum_intensity += get_threshold(img)
    # average_threshold = sum_intensity/len_data
    # print(f"Average threshold: {average_threshold:.5f}")
    # for folder#
#     images = os.listdir(dataset)
#     len_data = len(images)
#     print('len :', len_data)
#     for image in tqdm(images):
#         img_path = os.path.join(dataset, image)
#         img = cv2.imread(img_path)
#         sum_intensity += get_threshold(img)
# average_threshold = sum_intensity/len_data
# print(f"Average threshold: {average_threshold:.5f}")
    # for a image #
    # img = cv2.imread(dataset)
    # print(get_threshold(img))

    #for taking dataset
    images = os.listdir(dataset)
    len_data = len(images)
    print('len :', len_data)
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        sv_path1 = os.path.join(save_llie_path, image)
        # sv_path2 = os.path.join(save_dark, image)
        if get_threshold(img) > 13:
             cv2.imwrite(sv_path1, img)
        # else:
        #      cv2.imwrite(sv_path2, img)


def remove_noise():

    source_folder = '/home/user/Thesis/Zero-DCE++/data/train'  # Replace with your source folder path
    target_folder = '/home/user/Thesis/Zero-DCE++/data/new_train'  # Replace with your target folder path

    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    all_image_files =  os.listdir(source_folder) 

    for img_file in tqdm(all_image_files):
        img_path = os.path.join(source_folder, img_file)
        img = cv2.imread(img_path)  # Load image in grayscale

        # Apply FFT and remove noise
        denoised_img = apply_fft_and_remove_noise(img)

        # Save denoised image
        target_file = os.path.join(target_folder, img_file)
        cv2.imwrite(target_file, denoised_img)
    print("Denoised images saved successfully!")


def calculate_crop_image():
    model_test = AntiSpoofPredict(0)
    image_cropper = CropImage()
    model_1 = '2.7_80x80_MiniFASNetV2.pth'
    model_2 = '4_0_0_80x80_MiniFASNetV2.pth'
    dataset_path = 'miniFAS/datasets/Test/new_dataset/train'
    check_path = 'miniFAS/datasets/Test/train'
    dataset_train = 'miniFAS/datasets/Train/train/remove_noise'
    model1_name = '2.7_80x80'
    model2_name = '4_80x80'
    train_model1 = os.path.join(dataset_train, model1_name)
    train_model2 = os.path.join(dataset_train, model2_name)
    labels = os.listdir(dataset_path)
    check_images = os.listdir(check_path)
    for label in tqdm(labels):
        model1_path = os.path.join(train_model1, label)
        model2_path = os.path.join(train_model2, label)
        label_path = os.path.join(dataset_path, label)
        images = os.listdir(label_path)
        for image in tqdm(images):
            if image in check_images:
                img_path = os.path.join(check_path, image)
                img = cv2.imread(img_path)
                img_1 = crop_image(img, model_1, model_test, image_cropper)
                img_2 = crop_image(img, model_2, model_test, image_cropper)
                cv2.imwrite(os.path.join(model1_path,image), img_1)
                cv2.imwrite(os.path.join(model2_path,image), img_2)

if __name__ == "__main__":
    # calculate_threshold()
    # calculate_conf_face()
    calculate_crop_image()
    # remove_noise()