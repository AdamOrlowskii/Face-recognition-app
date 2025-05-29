"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import logging
mpl_logger = logging.getLogger('matplotlib')
mpl_logger.setLevel(logging.WARNING)
import logging.config
logging.config.fileConfig("../config/logging.conf")
logger = logging.getLogger('api')
import cv2

import yaml
import numpy as np
from core.image_cropper.arcface_cropper.FaceRecImageCropper import FaceRecImageCropper
def gui(path_to_file_jpg, path_to_file_res0):

    image_path = 'test_images/adam.jpg'
    image_info_file = 'test_images/adam_landmark_res0.txt'
    line = open(path_to_file_res0).readline().strip()
    landmarks_str = line.split(' ')
    landmarks = [float(num) for num in landmarks_str]
    
    face_cropper = FaceRecImageCropper()
    image = cv2.imread(path_to_file_jpg)
    cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
    output_path = 'test_images/testowanie_crop.jpg'
    cv2.imwrite(output_path, cropped_image)
    logger.info('Crop image successful!')
    return output_path


if __name__ == '__main__':
    image_path = 'test_images/adam.jpg'
    image_info_file = 'test_images/adam_landmark_res0.txt'
    line = open(image_info_file).readline().strip()
    landmarks_str = line.split(' ')
    landmarks = [float(num) for num in landmarks_str]
    
    face_cropper = FaceRecImageCropper()
    image = cv2.imread(image_path)
    cropped_image = face_cropper.crop_image_by_mat(image, landmarks)
    cv2.imwrite('test_images/adam_cropped.jpg', cropped_image)
    logger.info('Crop image successful!')
