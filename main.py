import cv2
import time
import numpy as np
import mediapipe as mp
import math
import pulsectl
import sys

class HandTrackingModule:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

    def findHands(self, img, draw=True):
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        lmList = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                h, w, c = img.shape
                for id, lm in enumerate(myHand.landmark):
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    lmList.append((id, cx, cy))
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)
        return lmList

width, height = 640, 480
capture = cv2.VideoCapture(0)
capture.set(3, width)
capture.set(4, height)
previous_time = 0
hand_detector = HandTrackingModule(detectionCon=0.7)

try:
    pulse = pulsectl.Pulse('volume_control')
    sinks = pulse.sink_list()
    
    if not sinks:
        print("Error: No audio sinks found!")
        sys.exit(1)
    
    default_sink = None
    for sink in sinks:
        if sink.name == pulse.server_info().default_sink_name:
            default_sink = sink
            break
    
    if default_sink is None:
        default_sink = sinks[0]
        print(f"Using audio sink: {default_sink.description}")
    
except Exception as e:
    print(f"Error connecting to PulseAudio: {e}")
    print("Make sure PulseAudio is running: pulseaudio --start")
    sys.exit(1)

if not capture.isOpened():
    print("Error: Could not open camera.")
    print("Try running: sudo dnf install v4l-utils")
    print("And check available cameras with: v4l2-ctl --list-devices")
    sys.exit(1)

print("Hand volume control started. Press 'q' to quit.")
print("Pinch thumb and index finger together to control volume.")

while True:
    success, image = capture.read()
    if not success:
        print("Failed to read from camera")
        break

    image = cv2.flip(image, 1)
    
    image = hand_detector.findHands(image)
    lmList = hand_detector.findPosition(image, draw=False)
    
    if len(lmList) != 0:
        x1, y1 = lmList[4][1], lmList[4][2]  # Thumb tip
        x2, y2 = lmList[8][1], lmList[8][2]  # Index finger tip
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

        cv2.circle(image, (x1, y1), 10, (100, 0, 0), cv2.FILLED)
        cv2.circle(image, (x2, y2), 10, (100, 0, 0), cv2.FILLED)
        cv2.line(image, (x1, y1), (x2, y2), (100, 0, 0), 2)
        cv2.circle(image, (cx, cy), 10, (100, 0, 0), cv2.FILLED)

        length = math.hypot(x2 - x1, y2 - y1)
        vol_level = np.interp(length, [30, 200], [0.0, 1.0])
        vol_level = max(0.0, min(1.0, vol_level))  # Clamp between 0.0 and 1.0

        try:
            pulse.volume_set_all_chans(default_sink, vol_level)
        except Exception as e:
            print(f"Error setting volume: {e}")

        bar_height = np.interp(vol_level, [0.0, 1.0], [400, 150])
        deep_blue = (139, 0, 255)  # BGR for a deep blue
        cv2.rectangle(image, (50, 150), (85, 400), deep_blue, 2)
        cv2.rectangle(image, (50, int(bar_height)), (85, 400), deep_blue, cv2.FILLED)
        cv2.putText(image, f'{int(vol_level*100)}%', (40, 430), cv2.FONT_HERSHEY_PLAIN, 2, deep_blue, 2)

        if length < 50:
            cv2.circle(image, (cx, cy), 10, (0, 255, 0), cv2.FILLED)

    current_time = time.time()
    fps = 1 / (current_time - previous_time) if (current_time - previous_time) > 0 else 0
    previous_time = current_time

    cv2.putText(image, f"FPS: {int(fps)}", (40, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 0, 0), 2)
    cv2.putText(image, "Press 'q' to quit", (40, height - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (100, 0, 0), 2)

    cv2.imshow("Hand Volume Control", image)
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()
pulse.close()
