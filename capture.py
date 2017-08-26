import cv2
# help show framerate
import time
import imutils

import numpy as np


FRAMES = 5

camera = cv2.VideoCapture(0)

if imutils.is_cv2():
  fgbg = cv2.BackgroundSubtractorMOG2(1, 80, 0)

else:
  cv2.createBackgroundSubtractorMOG2(history=1,detectShadows=False)


# people detector, built in
#hog = cv2.HOGDescriptor()
#hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())

def WriteMessage(message, frame):

    location = (frame.shape[1]//2 - frame.shape[1]//4, 10)
    color = (255, 255, 255)
    cv2.putText(frame,
    message,
    location,
    cv2.FONT_HERSHEY_SIMPLEX,
    0.35,
    color,
    1)


unSatisfied = True
while unSatisfied:

    (grabbed, background) = camera.read()

    if not grabbed:
        print ("fail to capture frame")
        break

    background = cv2.flip(background, 1)
    written = background.copy()

    WriteMessage("Please press the c key when you would like to set the background", written)
    cv2.imshow('Background', written)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("c"):

        while(True):

            written = background.copy()
            WriteMessage("Are you satisfied with the captured background? y/n", written)
            cv2.imshow('Background', written)

            key = cv2.waitKey(0) & 0xFF

            if key == ord("n"):
                break
            elif key == ord("y"):
                cv2.destroyAllWindows()
                unSatisfied = False
                break


fps = 0
frame_count = 0

while True:

    if frame_count == 0:
        start = time.time()
    elif frame_count % FRAMES == 0:
        end = time.time()
        seconds = end - start
        fps = FRAMES / seconds
        start = end

    # get next webcam frame
    (grabbed, frame) = camera.read()

    if not grabbed:
        print ("fail to capture frame")
        break

    flipframe = cv2.flip(frame,1)
    orig = flipframe

    # generate mask frame
    fgmask = fgbg.apply(background)
    fgmask = fgbg.apply(flipframe)

    # erosians and dilations remove any small remaining imperfections in the mask
    fgmask = cv2.erode(fgmask, None, iterations=2)
    fgmask = cv2.dilate(fgmask, None, iterations=2)

    cv2.imshow('Frame', orig)

    bmask = cv2.GaussianBlur(fgmask, (5,5),0)

    # show the fps on the mask
    cv2.putText(bmask, "fps: {}".format(fps),
    (bmask.shape[1] - 80, bmask.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX,
    0.35, (255, 255, 255), 1)

    if imutils.is_cv2():
      cnts, _ = cv2.findContours(bmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    else:
      _, cnts, _ = cv2.findContours(bmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)


    if (len(cnts) > 0):

        c = max(cnts, key=cv2.contourArea)
        x,y,w,h = cv2.boundingRect(c)
        cv2.rectangle(bmask,(x,y),(x+w,y+h),(255,0,0),2)

    cv2.imshow('frame',bmask)
    key = cv2.waitKey(10) & 0xFF

    frame_count+=1

    if key == ord("q"):
      break

camera.release()
cv2.destroyAllWindows()
