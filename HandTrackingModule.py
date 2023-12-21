# Tutorial for Code: https://www.youtube.com/watch?v=NZde8Xt78Iw&ab_channel=Murtaza%27sWorkshop-RoboticsandAI

import cv2
import mediapipe as mp
import enum
import time

class HandLandmark(enum.IntEnum):
  """The 21 hand landmarks"""
  WRIST = 0
  THUMB_LOW = 1
  THUMB_JOINT_LOWER = 2
  THUMB_JOINT_UPPER = 3
  THUMB_TIP = 4
  INDEX_FINGER_LOW = 5
  INDEX_FINGER_JOINT_LOWER = 6
  INDEX_FINGER_JOINT_UPPER = 7
  INDEX_FINGER_TIP = 8
  MIDDLE_FINGER_LOW = 9
  MIDDLE_FINGER_JOINT_LOWER = 10
  MIDDLE_FINGER_JOINT_UPPER = 11
  MIDDLE_FINGER_TIP = 12
  RING_FINGER_LOW = 13
  RING_FINGER_JOINT_LOWER = 14
  RING_FINGER_JOINT_UPPER = 15
  RING_FINGER_TIP = 16
  PINKY_FINGER_LOW = 17
  PINKY_FINGER_JOINT_LOWER = 18
  PINKY_FINGER_JOINT_UPPER = 19
  PINKY_FINGER_TIP = 20

class HandDetector():
    """HandDectecor class"""
    def __init__(self, mode=False, maxHands=2, modelComplexity=1, detectionConfidence=0.5, trackConfidence=0.5) -> None:
        """__init__

        Parameteters
        ------------
        @param mode : default False
            Whether to treat the input images as a batch of static
            and possibly unrelated images, or a video stream.
        @param maxHands : default 2
            Maximum number of hands to detect
        @oaram modelComplexity : default 1
            Complexity of the hand landmark model: 0 or 1.
            Landmark accuracy as well as inference latency 
            generally go up with the model complexity.
        @param detectionConfidence : default 0.5
            Minimum confidence value ([0.0, 1.0]) for hand
            detection to be considered successful.
        @param tackConfidence : default 0.5
            Minimum confidence value ([0.0, 1.0]) for the
            hand landmarks to be considered tracked successfully.
        """
        self.mode=mode
        self.maxHands=maxHands
        self.modelComplexity=modelComplexity
        self.detectionConfidence=detectionConfidence
        self.trackConfidence=trackConfidence
        
        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(self.mode, self.maxHands, self.modelComplexity, self.detectionConfidence, self.trackConfidence)
        self.mpDraw = mp.solutions.drawing_utils

    def find_hands(self, img, flip=False, draw=True):
        """find_hands method
        
        Parameters
        ----------
        @param img : required
            The image you will be finding the hands on.
        @param flip : default False
            Whether to flip the camera horizontally or not.
        @param draw : default True
            Whether to draw the landmarks or not.
        """
        if flip:
            img = cv2.flip(img, 1)

        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)
        # Find hand landmarks and draw them to image
        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        
        return img

    def find_position(self, img, handNo=0):
        """find_position method
        @param img : required
            The image to find the hand positions on.
        @param handNo : default 0
            Which hand to find the position of (0 or 1).
        """
        lmList = []
        if self.results.multi_hand_landmarks:
            myHand = self.results.multi_hand_landmarks[handNo]
            for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x*w), int(lm.y*h)
                    lmList.append([id, cx, cy])
        return lmList

def main():
    pTime = 0
    cTime = 0
    cap = cv2.VideoCapture(0)
    detector: HandDetector = HandDetector()

    while True:
    # Read camera image
        success, img = cap.read()

    # Find hands and position of landmakrs
        img = detector.find_hands(img, True)
        lmList = detector.find_position(img)
    
    # FPS tracking
        cTime = time.time()
        fps = 1/(cTime-pTime)
        pTime = cTime
        cv2.putText(img, str(f'FPS: {int(fps)}'),(10,50), cv2.FONT_HERSHEY_PLAIN, 3, (25,255,25), 3)
    # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()