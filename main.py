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
    frame_storage_path = os.path.join(location, "frames")
    extracted_frame_path = fe.frameExtractor(path_to_input_file, frame_storage_path, mid_frame_counter)
    
    print(f"Attempting to extract frame from: {path_to_input_file}")  # Debug print
    print(f"Frame should be stored at: {frame_storage_path}")  # Debug print
    print(f"Extracted frame path: {extracted_frame_path}")  # Debug print

  
    if not extracted_frame_path or not os.path.exists(extracted_frame_path):
        print(f"Failed to read image for {input_file}. Check if the file exists and the path is correct.")
        return None
    
    middle_image = cv2.imread(extracted_frame_path)
    if middle_image is None:
        print(f"Failed to read image from {extracted_frame_path}.")
        return None

    # Ensure the image has the correct shape (example for resizing and adding a channel dimension)
    middle_image_resized = cv2.resize(middle_image, (300, 300))
    if len(middle_image_resized.shape) == 2:  # Grayscale image, add a channels dimension
        middle_image_resized = np.expand_dims(middle_image_resized, axis=-1)

    response = hfe.HandShapeFeatureExtractor.get_instance().extract_feature(middle_image_resized)
    return response


def decide_gesture_by_file_name(gesture_file_name):
    gesture_key = gesture_file_name.split('_')[0]
    for x in gesture_data:
        if x.gesture_key == gesture_key:
            return x
    return None

def determine_gesture(gesture_location, gesture_file_name, mid_frame_counter):
    video_feature = extract_feature(gesture_location, gesture_file_name, mid_frame_counter)
    if video_feature is None:
        print(f"Feature extraction failed for {gesture_file_name}. Skipping...")
        return None  # Skipping this file due to failure in extracting features

    min_cosine_similarity = float('inf')
    recognized_gesture_detail = None
    for featureVector in featureVectorList:
        cosine_similarity = tf.keras.losses.cosine_similarity(video_feature, featureVector.extracted_feature, axis=-1)
        if cosine_similarity < min_cosine_similarity:
            min_cosine_similarity = cosine_similarity
            recognized_gesture_detail = featureVector.gesture_detail
    return recognized_gesture_detail


# Initialize gesture details
gesture_data = [
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
# Process training data to build feature vector list
featureVectorList = []
path_to_train_data = "traindata/"
count = 0
for file in os.listdir(path_to_train_data):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        gesture_detail = decide_gesture_by_file_name(file)
        if gesture_detail is not None:
            feature = extract_feature(path_to_train_data, file, count)
            if feature is not None:
                featureVectorList.append(GestureFeature(gesture_detail, feature))
        count += 1

# Process test data
results = []
test_data_path = "test/"
test_count = 0
for test_file in os.listdir(test_data_path):
    if not test_file.startswith('.') and not test_file.startswith('frames') and not test_file.startswith('results'):
        recognized_gesture_detail = determine_gesture(test_data_path, test_file, test_count)
        if recognized_gesture_detail is not None:
            results.append(int(recognized_gesture_detail.output_label))
        test_count += 1

# Save results to CSV
with open('Results.csv', 'w', newline='') as results_file:
    csv_writer = csv.writer(results_file)
    for result in results:
        csv_writer.writerow([result])  # Write each result to its own row

