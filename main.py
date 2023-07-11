#                           Browser control
import cv2
import numpy as np
import mediapipe as mp
import pyautogui
from tensorflow.keras.models import load_model
import time

#counting no.of fingers
def count_fingers(lst):
    cnt =   0

    thresh = (lst.landmark[0].y*100 - lst.landmark[9].y*100)/2

    if (lst.landmark[5].y*100 - lst.landmark[8].y*100) > thresh:
        cnt += 1

    if (lst.landmark[9].y*100 - lst.landmark[12].y*100) > thresh:
        cnt += 1

    if (lst.landmark[13].y*100 - lst.landmark[16].y*100) > thresh:
        cnt += 1

    if (lst.landmark[17].y*100 - lst.landmark[20].y*100) > thresh:
        cnt += 1

    if (lst.landmark[5].x*100 - lst.landmark[4].x*100) > 6:
        cnt += 1


    return cnt
# initialize mediapipe
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Load the gesture recognizer model
model = load_model('mp_hand_gesture')

# Load class names
f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

# Initialize the webcam for Hand Gesture Recognition Python project
cap = cv2.VideoCapture(0)

start_init = False
prev = ''

while True:
          end_time = time.time()
          # Read each frame from the webcam
          _, frame = cap.read()
          x , y, c = frame.shape

          # Flip the frame vertically
          frame = cv2.flip(frame, 1)
          framergb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
          # Get hand landmark prediction
          result = hands.process(framergb)

          className = ''

          # post process the result
          if result.multi_hand_landmarks:
                  hand_keyPoints = result.multi_hand_landmarks[0]

                  landmarks = []
                  for handslms in result.multi_hand_landmarks:
                          for lm in handslms.landmark:
                               #print(lm)
                              lmx = int(lm.x * x)  #0 to 1 x=480 y=640
                              lmy = int(lm.y * y)  #0.something into integer values


                              landmarks.append([lmx, lmy])

                          # Drawing landmarks on frames
                          mpDraw.draw_landmarks(frame, handslms, mpHands.HAND_CONNECTIONS)

                          # Predict gesture in Hand Gesture Recognition project
                          prediction = model.predict([landmarks])
                          #print(prediction)
                          classID = np.argmax(prediction)
                          className = classNames[classID]
                          cnt = count_fingers(hand_keyPoints)

                  if not(prev == className) :

                          if not (start_init):
                              start_time = time.time()
                              start_init = True
                          elif (end_time - start_time) > 0.5:
                              if (className == 'rock' and cnt == 2) :
                                  print("forward")
                                  print(cnt)
                                  pyautogui.press("right")

                              elif (className == 'smile'):
                                  print("backward")
                                  print(cnt)
                                  pyautogui.press("left")

                              elif (className == 'rock' and cnt == 1):
                                  print("volume up")
                                  print(cnt)
                                  pyautogui.press("up")

                              elif (className == 'thumbs down'):
                                  print("volume dowm")
                                  print(cnt)
                                  pyautogui.press("down")

                              elif ((className == 'stop' or className == 'live long') and cnt == 5):
                                  print("stop")
                                  print(cnt)
                                  pyautogui.press("space")

                              elif (className == 'peace' and cnt == 2):
                                  print("new tab")
                                  print(cnt)
                                  pyautogui.keyDown('ctrl')
                                  pyautogui.press('t')
                                  pyautogui.keyUp('ctrl')

                              elif (className == 'call me'):
                                  print("next open tab")
                                  print(cnt)
                                  pyautogui.keyDown('ctrl')
                                  pyautogui.press('tab')
                                  pyautogui.keyUp('ctrl')


                              elif (className == 'thumbs up'):
                                  print("previous open tab")
                                  print(cnt)
                                  pyautogui.keyDown('ctrl')
                                  pyautogui.press('pageup')
                                  pyautogui.keyUp('ctrl')

                              elif (className == 'fist' and cnt== 0):
                                  print("close")
                                  print(cnt)
                                  pyautogui.keyDown('ctrl')
                                  pyautogui.press('w')
                                  pyautogui.keyUp('ctrl')

                              prev = className
                              start_init = False
          # show the prediction on the frame
          cv2.putText(frame, className, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
          # Show the final output
          cv2.imshow("Output", frame)

          if cv2.waitKey(1) == ord('q'):
              break
# release the webcam and destroy all active windows
cap.release()
cv2.destroyAllWindows()


