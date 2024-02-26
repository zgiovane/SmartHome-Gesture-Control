# Import necessary libraries
import cv2
import os
import numpy as np
import tensorflow as tf
import csv

# Assuming frameextractor.py and handshape_feature_extractor.py are in the same directory
from frameextractor import frameExtractor
from handshape_feature_extractor import HandShapeFeatureExtractor

class GestureDetail:
    def __init__(self, gesture_key, gesture_name, output_label):
        self.gesture_key = gesture_key
        self.gesture_name = gesture_name
        self.output_label = output_label

class GestureFeature:
    def __init__(self, gesture_detail: GestureDetail, extracted_feature):
        self.gesture_detail = gesture_detail
        self.extracted_feature = extracted_feature

def extract_feature(location, input_file, mid_frame_counter):
    path_to_input_file = os.path.join(location, input_file)
    frame_storage_path = os.path.join(location, "frames/")
    extracted_frame = frameExtractor(path_to_input_file, frame_storage_path, mid_frame_counter)
    middle_image = cv2.imread(extracted_frame, cv2.IMREAD_GRAYSCALE)
    response = HandShapeFeatureExtractor.get_instance().extract_feature(middle_image)
    return response

def decide_gesture_by_file_name(gesture_file_name, gesture_details):
    for x in gesture_details:
        if x.gesture_key == gesture_file_name.split('_')[0]:
            return x
    return None

def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter, gesture_details, featureVectorList):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)
    closest_gesture = None
    min_cosine_distance = float('inf')
    for featureVector in featureVectorList:
        cosine_distance = tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature, axis=-1)
        if cosine_distance < min_cosine_distance:
            min_cosine_distance = cosine_distance
            closest_gesture = featureVector.gesture_detail
    return closest_gesture

# Initialize gesture details
gesture_details = [
    GestureDetail("Num0", "0", "0"),
    GestureDetail("Num1", "1", "1"),
    GestureDetail("Num2", "2", "2"),
    GestureDetail("Num3", "3", "3"),
    GestureDetail("Num4", "4", "4"),
    GestureDetail("Num5", "5", "5"),
    GestureDetail("Num6", "6", "6"),
    GestureDetail("Num7", "7", "7"),
    GestureDetail("Num8", "8", "8"),
    GestureDetail("Num9", "9", "9"),
    GestureDetail("FanDown", "Decrease Fan Speed", "10"),
    GestureDetail("FanOn", "FanOn", "11"),
    GestureDetail("FanOff", "FanOff", "12"),
    GestureDetail("FanUp", "Increase Fan Speed", "13"),
    GestureDetail("LightOff", "LightOff", "14"),
    GestureDetail("LightOn", "LightOn", "15"),
    GestureDetail("SetThermo", "SetThermo", "16")
]


# Extract features for training data
path_to_train_data = "traindata/"
featureVectorList = []
count = 0
for file in os.listdir(path_to_train_data):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        gesture_detail = decide_gesture_by_file_name(file, gesture_details)
        if gesture_detail:
            feature = extract_feature(path_to_train_data, file, count)
            featureVectorList.append(GestureFeature(gesture_detail, feature))
        count += 1

# Recognize gestures in test data
test_data_path = "test/"
results = []
test_count = 0
for test_file in os.listdir(test_data_path):
    if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
        recognized_gesture_detail = determine_gesture(test_data_path, test_file, test_count, gesture_details, featureVectorList)
        results.append(int(recognized_gesture_detail.output_label))
        test_count += 1

# Write results to CSV
with open('Results.csv', 'w', newline='') as results_file:
    writer = csv.writer(results_file)
    writer.writerow(results)
