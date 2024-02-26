import cv2
import os
import tensorflow as tf
import frameextractor as fe
import handshape_feature_extractor as hfe
import csv

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
    extracted_frame = fe.frameExtractor(path_to_input_file, frame_storage_path, mid_frame_counter)
    middle_image = cv2.imread(extracted_frame, cv2.IMREAD_GRAYSCALE)
    response = hfe.HandShapeFeatureExtractor.extract_feature(hfe.HandShapeFeatureExtractor.get_instance(), middle_image)
    return response

def decide_gesture_by_file_name(gesture_file_name):
    for x in gesture_details:
        if x.gesture_key == gesture_file_name.split('_')[0]:
            return x
    return None

# use cosine similarity for comparing the vectors to classify the gesture
def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)
    max_mutations = 0
    gesture_detail: GestureDetail = GestureDetail("", "", "")
    cos_sin = 1
    position = 0
    cursor = 0
    for featureVector in featureVectorList:
        calc_cos_sin = tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature, axis=-1)
        if calc_cos_sin < cos_sin:
            cos_sin = calc_cos_sin
            position = cursor
            cursor += 1
    gesture_detail = featureVectorList[position].gesture_detail
    return gesture_detail

# list of GestureDetails: gesture name, a description, and output label
gesture_details = [
    GestureDetail("Num0", "0", "0"), GestureDetail("Num1", "1", "1"),
    GestureDetail("Num2", "2", "2"), GestureDetail("Num3", "3", "3"),
    GestureDetail("Num4", "4", "4"), GestureDetail("Num5", "5", "5"),
    GestureDetail("Num6", "6", "6"), GestureDetail("Num7", "7", "7"),
    GestureDetail("Num8", "8", "8"), GestureDetail("Num9", "9", "9"),
    GestureDetail("FanDown", "Decrease Fan Speed", "10"),
    GestureDetail("FanOn", "FanOn", "11"), GestureDetail("FanOff", "FanOff", "12"),
    GestureDetail("FanUp", "Increase Fan Speed", "13"),
    GestureDetail("LightOff", "LightOff", "14"), GestureDetail("LightOn", "LightOn", "15"),
    GestureDetail("SetThermo", "SetThermo", "16")
]

# Get the penultimate layer for training data
featureVectorList = []
path_to_train_data = "traindata/"
count = 0
for file in os.listdir(path_to_train_data):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        featureVectorList.append(GestureFeature(decide_gesture_by_file_name(file), extract_feature(path_to_train_data, file, count)))
        count += 1

# Get the penultimate layer for test data
results = []
video_locations = ["test/"]
test_count = 0
for video_location in video_locations:
    fieldnames = ['Gesture_Video_File_Name', 'Gesture_Name', 'Output_Label']
    for test_file in os.listdir(video_location):
        if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
            recognized_gesture_detail = determine_gesture(video_location, test_file, test_count)
            test_count += 1
            results.append(int(recognized_gesture_detail.output_label))

# Print the results to Results.csv
with open('Results.csv', 'w') as results_file:
    train_data_writer = csv.writer(results_file, delimiter='\n')
    train_data_writer.writerow(results)
