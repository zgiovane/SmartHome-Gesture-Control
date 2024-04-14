# Sourced this script from below
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""

import cv2  # Importing OpenCV for video processing
import os  # Importing os module for file operations

def frameExtractor(videoPath, framesPath, count):
    # Create frames directory if it doesn't exist
    if not os.path.exists(framesPath):
        os.mkdir(framesPath)
    
    # Open the video file
    capture = cv2.VideoCapture(videoPath)
    
    # Get the total number of frames in the video
    videoLength = int(capture.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    
    # Get the frame number to extract (middle frame)
    frameNum = int(videoLength / 2)
    
    # Set the frame position in the video
    capture.set(1, frameNum)
    
    # Read the frame
    ret, frame = capture.read()
    
    # Define the filename for the extracted frame
    filename = framesPath + "%#05d.png" % (count + 1)
    
    # Save the frame as an image file
    cv2.imwrite(filename, frame)
    
    return filename  # Return the filename of the extracted frame

