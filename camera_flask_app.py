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

#카메라정보
fx = 910.014
fy = 913.830
cx = 675.058
cy = 284.985
k1 = -0.392208
k2 = 0.132114
p1 = 0.002060
p2 = -0.004569
h = 197 #cm, 카메라 높이
sexy = 3
# 좌표 평면화?
dot = [[434, 331], [152, 647], [698, 662], [763, 342], [598, 325]]

real_point = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]

for i in range(0, 5):
    u = (dot[i][0]-cx)/fx
    v = (dot[i][1]-cy)/fy

    CpPp = h * math.tan(math.pi/2 + 30 - math.atan(v))
    CPp = math.sqrt(h**2 + CpPp**2)
    Cpp = math.sqrt(1 + v**2)
    PPp = u*CPp/Cpp
    d = math.sqrt(CpPp**2 + PPp**2)
    theta = -math.atan2(PPp, CpPp)

    real_point[i][0] = d*math.cos(theta)
    real_point[i][1] = d*math.sin(theta)

real_length = math.sqrt((real_point[0][0]-real_point[1][0])**2+(real_point[0][1]-real_point[1][1])**2)
dot_length = math.sqrt((dot[0][0]-dot[1][0])**2+(dot[0][1]-dot[1][1])**2)


def detect():
    global d_x, d_y, radius, dot, sexy
    camera = cv2.VideoCapture('rtsp://admin:good425425@192.168.0.198:554/stream_ch00_0')

    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    blue,red = (255,0,0),(0,0,255) 
    
    AB1 = ((real_point[0][0]-real_point[1][0])**2 + (real_point[0][1]-real_point[1][1])**2) **0.5
    AB2 = ((real_point[1][0]-real_point[2][0])**2 + (real_point[1][1]-real_point[2][1])**2) **0.5
    AB3 = ((real_point[2][0]-real_point[3][0])**2 + (real_point[2][1]-real_point[3][1])**2) **0.5
    AB4 = ((real_point[3][0]-real_point[4][0])**2 + (real_point[3][1]-real_point[4][1])**2) **0.5
    AB5 = ((real_point[4][0]-real_point[0][0])**2 + (real_point[4][1]-real_point[0][1])**2) **0.5
    
    count = 0
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
                    
                    nosex = landmarks[0][0]
                    nosey = landmarks[0][1]
                    left_heelx = landmarks[29][0]
                    left_heely = landmarks[29][1]
                    right_heelx = landmarks[30][0]
                    right_heely = landmarks[30][1]
                    mx = (left_heelx+ right_heelx)/2
                    my = (left_heely+ right_heely)/2

                    u = (mx-cx)/fx
                    v = (my-cy)/fy
                    CpPp = h * math.tan(math.pi/2 + 30 - math.atan(v))
                    CPp = math.sqrt(h**2 + CpPp**2)
                    Cpp = math.sqrt(1 + v**2)
                    PPp = u*CPp/Cpp
                    d = math.sqrt(CpPp**2 + PPp**2)
                    theta = -math.atan2(PPp, CpPp)
                    d_x = d*math.cos(theta)
                    d_y = d*math.sin(theta)

                    u = (nosex-cx)/fx
                    v = (nosey-cy)/fy
                    CpPp = h * math.tan(math.pi/2 + 30 - math.atan(v))
                    CPp = math.sqrt(h**2 + CpPp**2)
                    Cpp = math.sqrt(1 + v**2)
                    PPp = u*CPp/Cpp
                    d = math.sqrt(CpPp**2 + PPp**2)
                    theta = -math.atan2(PPp, CpPp)
                    d_nx = d*math.cos(theta)
                    d_ny = d*math.sin(theta)
                    
                    mp_drawing.draw_landmarks(
                        img,
                        results.pose_landmarks,
                        mp_pose.POSE_CONNECTIONS,
                        landmark_drawing_spec=mp_drawing_styles.get_default_pose_landmarks_style())

                    
                    # radius = math.sqrt((mx-nosex)**2 + (my-nosey)**2)
                    radius = 20
                    print(sexy)
                    if count == 0:
                        area1 = abs((real_point[0][0]-d_x) * (real_point[1][1]-d_y) - (real_point[0][1]-d_y) * (real_point[1][0] - d_x))
                        distance1 = area1/AB1
                        count += 1
                        if distance1 < radius:
                            sexy = 1
                        else: sexy = 0
                    elif count == 1:
                        area2 = abs((real_point[1][0]-d_x) * (real_point[2][1]-d_y) - (real_point[1][1]-d_y) * (real_point[2][0] - d_x))
                        distance2 = area2/AB2
                        count += 1
                        if distance2 < radius:
                            sexy = 1
                        else: sexy = 0
                    elif count == 2:
                        area3 = abs((real_point[2][0]-d_x) * (real_point[3][1]-d_y) - (real_point[2][1]-d_y) * (real_point[3][0] - d_x))
                        distance3 = area3/AB3
                        count += 1
                        if distance3 < radius:
                            sexy = 1
                        else: sexy = 0
                    elif count == 3:
                        area4 = abs((real_point[3][0]-d_x) * (real_point[4][1]-d_y) - (real_point[3][1]-d_y) * (real_point[4][0] - d_x))
                        distance4 = area4/AB4
                        count += 1
                        if distance4 < radius:
                            sexy = 1
                        else: sexy = 0
                    else:
                        area5 = abs((real_point[4][0]-d_x) * (real_point[4][1]-d_y) - (real_point[4][1]-d_y) * (real_point[0][0] - d_x))
                        distance5 = area5/AB5
                        count = 0
                        if distance5 < radius:
                            sexy = 1
                        else: sexy = 0
                if cv2.waitKey(20) & 0xFF == 27:
                    break
    camera.release()    



try:
    os.mkdir('./shots')
except OSError as error:
    pass

#Load pretrained face detection model    
net = cv2.dnn.readNetFromCaffe('./saved_model/deploy.prototxt.txt', './saved_model/res10_300x300_ssd_iter_140000.caffemodel')

#instatiate flask app  
app = Flask(__name__, template_folder='./templates')


camera = cv2.VideoCapture('rtsp://admin:good425425@192.168.0.198:554/stream_ch00_0')



def gen_frames():  # generate frame by frame from camera
    global out, capture,rec_frame
    while True:
        success, frame = camera.read() 
        if success:
            try:
                detect()
                ret, buffer = cv2.imencode('.jpg', cv2.flip(frame,1))
                frame = buffer.tobytes()
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
            except Exception as e:
                pass
                
        else:
            pass
@app.route('/nojjang', methods=['GET','POST']) 
def tmp():
    value = ' '
    return render_template('index.html', value = sexy)


@app.route('/')
def index():
    return render_template('index.html')
    
    
@app.route('/video_feed')
def video_feed():
    return Response(gen_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run()
    

camera.release()
cv2.destroyAllWindows()     