import os
import cv2
import time
import numpy as np
from imutils.video import WebcamVideoStream

vs = WebcamVideoStream(src = 0).start()

print("[INPUT] Enter your name : ", end="")
name = input()

DATA_DIR = 'faces/' + name + "/"
if(not os.path.exists(DATA_DIR)):
    print("[INFO] Data directory does not exist, creating .. ")
    os.mkdir(DATA_DIR)
    print("[INFO] Data directory created ... ")

print("[INFO] Reading detection model ... ")
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "dnn_model.caffemodel")

captured = False
num_captured = 0
while(True):
    FACE_DETECTED  = False
    frame = vs.read()

    (H, W) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), [104,111,123])
    net.setInput(blob)
    detections = net.forward()

    num_faces = 0

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if(confidence < 0.7):
            continue
        else:
            FACE_DETECTED = True
            num_faces += 1
            box = np.array([W,H,W,H]) * detections[0,0,i,3:7]
            (startX, startY, endX, endY) = box.astype("int")

            cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)

    if(captured):
        if(FACE_DETECTED):
            if(num_captured < 20):
                cv2.imwrite(DATA_DIR + name + "_" + str(num_captured) + ".jpg", frame)
                num_captured += 1

            cv2.putText(frame, "Image has been captured | num frames : " + str(num_captured), (20,20), \
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    cv2.imshow("Frame",frame)

    key = cv2.waitKey(1)
    if(key == ord("q")):
        break
    elif(key == ord("s")):
        cv2.imwrite(DATA_DIR + name + "_0.jpg", frame)
        # os.mkdir()
        # modify so that each person has 20 frames
        captured = True

vs.stop()
cv2.destroyAllWindows()
    
