from flask import Flask, render_template, Response, request
import cv2
import datetime, time
import os, sys
import numpy as np
from threading import Thread
import mediapipe as mp
import math


global capture,rec_frame, grey, switch, neg, face, rec, out 
capture=0
grey=0
neg=0
face=0
switch=1
rec=0
dot = [[434, 331], [152, 647], [698, 662], [763, 342], [598, 325]]

def real_dot(dot):
    fx = 910.014
    fy = 913.830
    cx = 675.058
    cy = 284.985
    h = 197
    u = (dot[0]-cx)/fx
    v = (dot[1]-cy)/fy

    CpPp = h * math.tan(math.pi/2 + 30 - math.atan(v))
    CPp = math.sqrt(h**2 + CpPp**2)
    Cpp = math.sqrt(1 + v**2)
    PPp = u*CPp/Cpp
    d = math.sqrt(CpPp**2 + PPp**2)
    theta = -math.atan2(PPp, CpPp)
    
    return d*math.cos(theta), d*math.sin(theta)

def real_dots(dots):
    real_point = []
    for i in range(0, len(dots)):
        x, y = real_dot(dots[i])
        real_point.append([x, y])
    return real_point

def is_inside(polygon, point):
    def cross(p1, p2):
        x1, y1 = p1
        x2, y2 = p2

        if y1 - y2 == 0:
            if y1 == point[1]:
                if min(x1, x2) <= point[0] <= max(x1, x2):
                    return 1, True
            return 0, False
        if x1 - x2 == 0:
            if min(y1, y2) <= point[1] <= max(y1, y2):
                if point[0] <= max(x1, x2):
                    return 1, point[0] == max(x1, x2)
            return 0, False
        a = (y1 - y2) /(x1 - x2)
        b = y1 - x1 * a
        x = (point[1] - b) / a
        if point[0] <= x:
            if min(y1, y2) <= point[1] <= max(y1, y2):
                return 1, point[0] == x or point[1] in (y1, y2)
        return 0, False

    cross_points = 0
    for x in range(0, 5):
        num, on_line = cross(polygon[x], polygon[x-1])
        if on_line:
            return True
        cross_points += num
    
    return cross_points % 2

def detect():
    global d_x, d_y, radius, dot, where
    tmp = 0
    camera = cv2.VideoCapture('rtsp://admin:good425425@192.168.0.198:554/stream_ch00_0')
    mp_pose = mp.solutions.pose
    
    where = 0
    with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        while True:
            ret, img = camera.read()
            if ret:
                
                img.flags.writeable = False
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                results = pose.process(img)
                img.flags.writeable = True
                img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

                landmarks = results.pose_landmarks
                if landmarks is not None:
                    # Check the number of landmarks and take pose landmarks.
                    assert len(landmarks.landmark) == 33, 'Unexpected number of predicted pose landmarks: {}'.format(len(landmarks.landmark))
                    landmarks = [[lmk.x, lmk.y, lmk.z] for lmk in landmarks.landmark]
                    frame_height, frame_width = img.shape[:2]
                    landmarks *= np.array([frame_width, frame_height, frame_width])
                    
                    mx = (landmarks[29][0]+ landmarks[30][0])/2 #left_heelx + right_heelx
                    my = (landmarks[29][1]+ landmarks[30][1])/2 #left_heely + right_heely
                    target_xy = [mx, my]
                    d_x, d_y = real_dot(target_xy)
                    

                    if is_inside(real_dots(dot), (d_x, d_y)) == 1: 
                        if tmp == 0:
                            where = 0
                            tmp = 1
                            print(where)
                    else: 
                        if tmp == 1:
                            where = 1
                            tmp = 0
                            print(where)

                if cv2.waitKey(20) & 0xFF == 27:
                    break
    camera.release()    
detect()