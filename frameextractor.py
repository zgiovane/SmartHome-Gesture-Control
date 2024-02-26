# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
#code to get the key frame from the video and save it as a png file.

import cv2
import os
#videopath : path of the video file
#frames_path: path of the directory to which the frames are saved
#count: to assign the video order to the frane.
def frameExtractor(videopath, frames_path, count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    if not cap.isOpened():
        print(f"Failed to open video: {videopath}")
        return None
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(video_length / 2)
    cap.set(1, frame_no)
    ret, frame = cap.read()
    if not ret:
        print("Failed to extract frame from video:", videopath)
        return None
    frame_filename = os.path.join(frames_path, f"{count:05d}.png")
    cv2.imwrite(frame_filename, frame)
    print(f"Frame extracted and saved to: {frame_filename}")  # Print the path here
    return frame_filename
