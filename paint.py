# import necessary libraries
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import numpy as np
import os

# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# in model folder

frame_count = 0
global latest_result
latest_result = None

def paint_function(result, output_image, frame_count):
    global latest_result
    latest_result = result
    #callback function to get results from model

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'hand_landmarker.task')
#finds folder needed to run the model and creates path to it

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
running_mode=vision.RunningMode.LIVE_STREAM
options = vision.HandLandmarkerOptions(base_options=base_options, 
                                       num_hands=1, 
                                       min_hand_detection_confidence=0.85,
                                       min_hand_presence_confidence=0.85,
                                       min_tracking_confidence=0.85,
                                       running_mode=running_mode,
                                       result_callback=paint_function)
#finds path and configures model

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
#configures model to options & starts webcam

ret, frame = cap.read()
frame = cv2.flip(frame, 1)
canvas = np.zeros_like(frame)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
#checks to see if webcam works

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #opencv = BGR, mediapipe = RGB
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    result = detector.detect_async(mp_image, timestamp_ms=frame_count)
    #flips frame, gets dimensions, converts to RGB, creates mediapipe image, and runs model on it

    if latest_result is not None and latest_result.hand_landmarks:
        for hand_landmarks in latest_result.hand_landmarks:
            index_finger_tip = hand_landmarks[8]
            #check for hand landmarks and get index finger tip (landmark 8)
            x,y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.line(canvas, (x,y), (x,y), (127,0,255), 10, lineType=cv2.LINE_AA)
            # cv2.circle(canvas, (x,y), 10, (127,0,255), -1, lineType=cv2.LINE_AA)
    else:
        pass

    cv2.imshow('Finger Paint', canvas)

    frame_count += 1

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
