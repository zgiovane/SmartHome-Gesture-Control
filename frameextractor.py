# -*- coding: utf-8 -*-
"""
Created on Thu Jan 28 00:52:08 2021

@author: chakati
"""
import cv2
import os

def frameExtractor(videopath, frames_path, count):
    if not os.path.exists(frames_path):
        os.mkdir(frames_path)
    cap = cv2.VideoCapture(videopath)
    video_length = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) - 1
    frame_no = int(video_length / 2)
    # print("Extracting frame..\n")
    cap.set(1, frame_no)
    ret, frame = cap.read()
    # cv2.imwrite(frames_path + "/%#05d.png" % (count+1), frame)
    filename = frames_path + "%#05d.png" % (count + 1)
    cv2.imwrite(filename, frame)
    return filename
