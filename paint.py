# import necessary libraries
import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import os

# https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
# in model folder

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

MODEL_PATH = os.path.join(SCRIPT_DIR, 'model', 'hand_landmarker.task')
#finds folder needed to run the model and creates path to it

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(base_options=base_options, 
                                       num_hands=1, 
                                       min_hand_detection_confidence=0.85,
                                       min_hand_presence_confidence=0.85,
                                       min_tracking_confidence=0.85)
#finds path and configures model

detector = vision.HandLandmarker.create_from_options(options)
cap = cv2.VideoCapture(0)
#configures model to options & starts webcam

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
    result = detector.detect(mp_image)
#flips frame, gets dimensions, converts to RGB, creates mediapipe image, and runs model on it

    if result.hand_landmarks:
        for hand_landmarks in result.hand_landmarks:
            index_finger_tip = hand_landmarks[8]
            #check for hand landmarks and get index finger tip (landmark 8)
            x,y = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cv2.circle(frame, (x,y), 10, (127,0,255), -1, lineType=cv2.LINE_AA)

    cv2.imshow('Finger Paint', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
