import cv2
import time
import mediapipe as mp
import enum
import os
import math
import HandTrackingModule as htm
import sys

###########################
wCam, hCam = 1280, 960
###########################

def main():
    # Webcam perameters
    cap = cv2.VideoCapture(0)
    cap.set(3, wCam)
    cap.set(4, hCam)

    # Hand detector vars
    detector = htm.HandDetector(detectionConfidence=0.5)
    landmarks = htm.HandLandmark
    drawring = []
    while True:
    # Read camera image and add number overlay
        success, img = cap.read()
    # Find hands and position of landmakrs
        img = detector.find_hands(img, True)
        lmList = detector.find_position(img)

        # These conditions determine whether someone is lifting a finger or not
        thumb, index, middle, ring, pinky = 0, 0, 0, 0, 0
        if len(lmList) != 0:
            if lmList[landmarks.THUMB_TIP][1] > lmList[landmarks.THUMB_JOINT_UPPER][1]:
                thumb = 1
            else:
                thumb = 0
            if lmList[landmarks.INDEX_FINGER_TIP][2] < lmList[landmarks.INDEX_FINGER_JOINT_LOWER][2]:
                index = 1
            else:
                index = 0
            if lmList[landmarks.MIDDLE_FINGER_TIP][2] < lmList[landmarks.MIDDLE_FINGER_JOINT_LOWER][2]:
                middle = 1
            else:
                middle = 0
            if lmList[landmarks.RING_FINGER_TIP][2] < lmList[landmarks.RING_FINGER_JOINT_LOWER][2]:
                ring = 1
            else:
                ring = 0
            if lmList[landmarks.PINKY_FINGER_TIP][2] < lmList[landmarks.PINKY_FINGER_JOINT_LOWER][2]:
                pinky = 1
            else:
                pinky = 0

        # Sum the number of lifted fingers
        sum = thumb + index + middle + ring + pinky

        cv2.putText(img, "Face your left palm forwards.", (wCam//16, hCam - 25), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)
        cv2.putText(img, str(sum), (wCam//2,25), cv2.FONT_HERSHEY_PLAIN, 2, (255,255,255), 2)

    # Display image
        cv2.imshow("Image", img)
        cv2.waitKey(1)

if __name__ == "__main__":
    main()