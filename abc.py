import math, time
import cv2
import mediapipe as mp
import numpy as np

import math
import matplotlib.pyplot as plt
from drawnow import *
fx = 910.014
fy = 913.830
cx = 675.058
cy = 284.985
k1 = -0.392208
k2 = 0.132114
p1 = 0.002060
p2 = -0.004569
h = 197 #cm
cap = cv2.VideoCapture('rtsp://admin:lolcololco@192.168.0.198:554/stream_ch00_0')
def show_plot():
    
    global d_x, d_y, radius
    plt.axes(xlim=(-100, 100), ylim=(-100, 100))
    # p = plt.Polygon(real_point, fill=None ,edgecolor='k',ls='solid',lw=3)
    # plt.gcf().gca().add_artist(p)
    plt.plot([real_point[0][0], real_point[1][0], real_point[2][0], real_point[3][0], real_point[4][0]],
         [real_point[0][1], real_point[1][1], real_point[2][1], real_point[3][1], real_point[4][1]])
   
    if d_x is not None:
        plt.plot([d_x], [d_y])
        c =plt.Circle((d_x, d_y), radius, fc='w', ec='b', fill = None)
        plt.gcf().gca().add_artist(c)

def detect():
    global d_x, d_y, radius, dot
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles
    mp_pose = mp.solutions.pose
    blue,red = (255,0,0),(0,0,255) 
    
    # AB1 = ((dot[0][0]-dot[1][0])**2 + (dot[0][1]-dot[1][1])**2) **0.5
    # AB2 = ((dot[1][0]-dot[2][0])**2 + (dot[1][1]-dot[2][1])**2) **0.5
    # AB3 = ((dot[2][0]-dot[3][0])**2 + (dot[2][1]-dot[3][1])**2) **0.5
    # AB4 = ((dot[3][0]-dot[4][0])**2 + (dot[3][1]-dot[4][1])**2) **0.5
    # AB5 = ((dot[4][0]-dot[0][0])**2 + (dot[4][1]-dot[0][1])**2) **0.5
    
    count = 0
    with mp_pose.Pose(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5) as pose:
        while True:
            ret, img = cap.read()
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
                    radius = 10
                    print(radius)
                    # if count == 0:
                    #     area1 = abs((dot[0][0]-mx) * (dot[1][1]-my) - (dot[0][1]-my) * (dot[1][0] - mx))
                    #     distance1 = area1/AB1
                    #     count += 1
                    #     if distance1 < radius:
                    #         print("Danger1")
                    # elif count == 1:
                    #     area2 = abs((dot[1][0]-mx) * (dot[2][1]-my) - (dot[1][1]-my) * (dot[2][0] - mx))
                    #     distance2 = area2/AB2
                    #     count += 1
                    #     if distance2 < radius:
                    #         print("Danger2")
                    # elif count == 2:
                    #     area3 = abs((dot[2][0]-mx) * (dot[3][1]-my) - (dot[2][1]-my) * (dot[3][0] - mx))
                    #     distance3 = area3/AB3
                    #     count += 1
                    #     if distance3 < radius:
                    #         print("Danger3")
                    # elif count == 3:
                    #     area4 = abs((dot[3][0]-mx) * (dot[4][1]-my) - (dot[3][1]-my) * (dot[4][0] - mx))
                    #     distance4 = area4/AB4
                    #     count += 1
                    #     if distance4 < radius:
                    #         print("Danger4")
                    # else:
                    #     area5 = abs((dot[4][0]-mx) * (dot[0][1]-my) - (dot[4][1]-my) * (dot[0][0] - mx))
                    #     distance5 = area5/AB5
                    #     count = 0
                    #     if distance5 < radius:
                    #         print("Danger5")
                    cv2.circle(img, (int(mx), int(my)), int(radius), blue, 4)
                cv2.line(img, dot[0], dot[1], red, 3)
                cv2.line(img, dot[1], dot[2], red, 3)
                cv2.line(img, dot[2], dot[3], red, 3)
                cv2.line(img, dot[3], dot[4], red, 3)
                cv2.line(img, dot[4], dot[0], red, 3)
                cv2.imshow("test", img)
                drawnow(show_plot)
             
                if cv2.waitKey(20) & 0xFF == 27:
                    break
    cap.release()    
dot = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
real_point = [[0, 0], [0, 0], [0, 0], [0, 0], [0, 0]]
def area(event, x, y, flags, param):
    global dot, img, i
    
    if event == cv2.EVENT_LBUTTONDOWN: 
        img_draw = img.copy()     # 왼쪽 마우스 버튼 다운, 드래그 시작
        if i == 0:
            print(x, y)
            dot[i][0] = x
            dot[i][1] = y
            i += 1
        
        elif i == 1:
            print(x, y)
            dot[i][0] = x
            dot[i][1] = y
            cv2.line(img, dot[i-1], dot[i], (0, 0, 255), 3)

            i += 1
            
        elif i == 2:
            print(x, y)
            dot[i][0] = x
            dot[i][1] = y
            cv2.line(img, dot[i-1], dot[i], (0, 0, 255), 3)

            i += 1
        elif i == 3:
            print(x, y)
            dot[i][0] = x
            dot[i][1] = y
            cv2.line(img, dot[i-1], dot[i], (0, 0, 255), 3)

            i += 1
        elif i == 4:
            print(x, y)
            dot[i][0] = x
            dot[i][1] = y
            cv2.line(img, dot[i-1], dot[i], (0, 0, 255), 3)
            cv2.line(img, dot[i], dot[0], (0, 0, 255), 3)
            i += 1
            
        cv2.imshow('roi', img_draw)
        print(i)

if cap.isOpened():
    i = 0
    while True:
        # time.sleep(2)
        ret, img2 = cap.read()
        if ret:
            cv2.namedWindow("roi", 1)
            cv2.setMouseCallback("roi", area)
            img = cv2.flip(img2, 1)
            img = cv2.flip(img, 1)
            
            cv2.imshow("roi", img)
            if cv2.waitKey(0) == ord('c'):
                cv2.destroyWindow("roi")
                break
        else:
            print("no fram")
            break
else:
    print("can't open camera")  

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
detect()


