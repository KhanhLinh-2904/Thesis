import random
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
    dataset = 'miniFAS/datasets/Test/inference_Real'
    # train_path = '/home/user/low_light_enhancement/Zero-DCE++/data/FAS_Thuan/train'
    new_train_path = 'miniFAS/datasets/Test/crop_Real'
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
        if conf >= 0.9:
            len_data += 1
            src_path = os.path.join(dataset, image)
            desc_path = os.path.join(new_train_path, image)
            shutil.copy(src_path, desc_path)
    print(len_data)

def calculate_threshold():
    dataset = 'miniFAS/datasets/Test/recog_face_Hiep'
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
    # len_data = len(images)
    # print('len :', len_data)
    for image in tqdm(images):
        img_path = os.path.join(dataset, image)
        img = cv2.imread(img_path)
        # sv_path1 = os.path.join(save_llie_path, image)
        # sv_path2 = os.path.join(save_dark, image)
        if get_threshold(img) >= 13:
            #  cv2.imwrite(sv_path1, img)
            len_data += 1
        # else:
        #      cv2.imwrite(sv_path2, img)
    print('len :', len_data)


def remove_noise():

    source_folder = 'miniFAS/datasets/Test/new_dataset/train/1'  # Replace with your source folder path
    target_folder = 'miniFAS/datasets/Test/re_noise'  # Replace with your target folder path

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
    dataset_path = 'miniFAS/datasets/Test/crop_Real'
    dataset_train = 'miniFAS/datasets/Train/train/remove_noise'
    model1_name = '2.7_80x80'
    model2_name = '4_80x80'
    train_model1 = os.path.join(dataset_train, model1_name)
    train_model2 = os.path.join(dataset_train, model2_name)
    labels = os.listdir(dataset_path)
    for label in tqdm(labels):
        model1_path = os.path.join(train_model1, label)
        model2_path = os.path.join(train_model2, label)
        label_path = os.path.join(dataset_path, label)
        images = os.listdir(label_path)
        for image in tqdm(images):
            img_path = os.path.join(label_path, image)
            img = cv2.imread(img_path)
            img_1 = crop_image(img, model_1, model_test, image_cropper)
            img_2 = crop_image(img, model_2, model_test, image_cropper)
            cv2.imwrite(os.path.join(model1_path,image), img_1)
            cv2.imwrite(os.path.join(model2_path,image), img_2)

def shuffle_and_move_data():
    input_folder = "miniFAS/datasets/Test/rec_Hiep"
    folder_A = "miniFAS/datasets/Test/new_dataset/test/1"
    folder_B = "miniFAS/datasets/Test/new_dataset/train/1"
    num_images_A = 732
    # Get list of images
    images = os.listdir(input_folder)
    random.shuffle(images)

    # Create folders if they don't exist
    os.makedirs(folder_A, exist_ok=True)
    os.makedirs(folder_B, exist_ok=True)

    # Move images to folder A and B
    for i, image in enumerate(images):
        if i < num_images_A:
            shutil.move(os.path.join(input_folder, image), os.path.join(folder_A, image))
        else:
            shutil.move(os.path.join(input_folder, image), os.path.join(folder_B, image))

def shuffle_and_move_folder():

    # Source parent directory containing folders
    source_parent_dir = "miniFAS/datasets/real_extract_video"

    # Destination directory to move folders into
    destination_dir = "miniFAS/datasets/real_train"

    # Number of folders to move
    num_folders_to_move = 75

    # Get a list of all folders in the source parent directory
    all_folders = [folder for folder in os.listdir(source_parent_dir) if os.path.isdir(os.path.join(source_parent_dir, folder))]

    # Randomly select folders to move
    folders_to_move = random.sample(all_folders, min(num_folders_to_move, len(all_folders)))

    # Move selected folders to the destination directory
    for folder in folders_to_move:
        source_path = os.path.join(source_parent_dir, folder)
        destination_path = os.path.join(destination_dir, folder)
        shutil.move(source_path, destination_path)

    print(f"{len(folders_to_move)} folders moved to {destination_dir}.")
def copy_files_to_new_folder():

    # Example usage:
    source_directory = "miniFAS/datasets/real_test"
    destination_directory = "miniFAS/datasets/Real/test"
    sub_folder = os.listdir(source_directory)
    for subfold in sub_folder:
        new_name = '1_'+ str(subfold)
        sub_folder_path = os.path.join(source_directory, subfold)
        files = os.listdir(sub_folder_path)
        cnt = 0
        for file in files:
            cnt += 1
            new_name_file = new_name + '_' +str(cnt) + '.jpg'
            destination_file_path = os.path.join(destination_directory, new_name_file)
            source_file_path = os.path.join(sub_folder_path, file)
            shutil.copy2(source_file_path, destination_file_path)
            print(new_name_file)

    # for root, _, files in os.walk(source_directory):
    #     for file in files:
    #         # Construct paths for source file and destination file
    #         source_file_path = os.path.join(root, file)
    #         destination_file_path = os.path.join(new_folder_path, file)
            
    #         # Copy the file to the new folder
    #         shutil.copy2(source_file_path, destination_file_path)
            
    # print(f"All files from subfolders copied to '{new_folder_name}' in '{destination_dir}'.")





if __name__ == "__main__":
    # calculate_threshold()
    # calculate_conf_face()
    calculate_crop_image()
    # remove_noise()
    # shuffle_and_move_data()
    # shuffle_and_move_folder()
    # copy_files_to_new_folder()
