# -*- coding: utf-8 -*-

import cv2
import numpy as np
import os
import tensorflow as tf


## import the handfeature extractor class
import frameextractor as fe
import handshape_feature_extractor as hfe
import csv


class GestureDetail:
    def __init__(self, gestureKey, gestureName, outputLabel):
        self.gestureKey = gestureKey
        self.gestureName = gestureName
        self.outputLabel = outputLabel

class GestureFeature:
    def __init__(self, gestureDetail: GestureDetail, extractedFeature):
        self.gestureDetail = gestureDetail
        self.extractedFeature = extractedFeature

# Function to extract features from a given video file
def extractFeature(location, inputFile, midFrameCounter):
    pathToInputFile = os.path.join(location, inputFile)
    frameStoragePath = os.path.join(location, "frames/")
    extractedFrame = fe.frameExtractor(pathToInputFile, frameStoragePath, midFrameCounter)
    middleImage = cv2.imread(extractedFrame, cv2.IMREAD_GRAYSCALE)
    response = hfe.HandShapeFeatureExtractor.extract_feature(hfe.HandShapeFeatureExtractor.get_instance(), middleImage)
    return response

# List of predefined gestures with their details
gestureDetails = [
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

# Function to decide the gesture type by the file name
def decideGestureByFileName(gestureFileName):
        gesturePrefixMapping = {
        "DecreaseFanSpeed": "FanDown",
        "FanOff": "FanOff",
        "FanOn": "FanOn",
        "IncreaseFanSpeed": "FanUp",
        "LightOff": "LightOff",
        "LightOn": "LightOn",
    }
    # Extract the gesture prefix from the file name
    gesturePrefix = gestureFileName.split('_')[0]
    gestureKey = gesturePrefixMapping.get(gesturePrefix, gesturePrefix)  # Direct mapping or use as is
    
    for x in gestureDetails:
        if x.gestureKey == gestureFileName.split('_')[0]:
            return x
    return None

# =============================================================================
# Get the penultimate layer for trainig data
# =============================================================================
# your code goes here
# Extract the middle frame of each gesture video

featureVectorList = []
pathToTrainData = "traindata/"
count = 0
for file in os.listdir(pathToTrainData):
    if not file.startswith('.') and not file.startswith('frames') and not file.startswith('results'):
        featureVectorList.append(GestureFeature(decideGestureByFileName(file), extractFeature(pathToTrainData, file, count)))
        count += 1


# =============================================================================
# Get the penultimate layer for test data
# =============================================================================
# your code goes here 
# Extract the middle frame of each gesture video

results = []
videoLocations = ["test/"]
testCount = 0
for videoLocation in videoLocations:
    for testFile in os.listdir(videoLocation):
        if not testFile.startswith('.') and not testFile.startswith('frames') and not testFile.startswith('results'):
            recognizedGestureDetail = determineGesture(videoLocation, testFile, testCount)
            testCount += 1
            results.append(int(recognizedGestureDetail.outputLabel))


# =============================================================================
# Recognize the gesture (use cosine similarity for comparing the vectors)
# =============================================================================
def determineGesture(gestureLocation, gestureFileName, midFrameCounter):
    videoFeature = extractFeature(gestureLocation, gestureFileName, midFrameCounter)
    cosSin = 1
    position = 0
    cursor = 0
    for featureVector in featureVectorList:
        calcCosSin = tf.keras.losses.cosineSimilarity(videoFeature, featureVector.extractedFeature, axis=-1)
        if calcCosSin < cosSin:
            cosSin = calcCosSin
            position = cursor
        cursor += 1
    gestureDetail = featureVectorList[position].gestureDetail
    return gestureDetail


# Print the results to Results.csv
with open('Results.csv', 'w') as resultsFile:
    trainDataWriter = csv.writer(resultsFile, delimiter='\n')
    trainDataWriter.writerow(results)
