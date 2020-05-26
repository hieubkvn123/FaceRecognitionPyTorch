import os
import cv2
import torch
import time
import imutils
import numpy as np
import face_recognition

from triplet_net import TripletLossNet
from scipy.spatial.distance import cosine
from imutils.video import WebcamVideoStream

HEIGHT, WIDTH, CHANNELS = 128, 128, 3

print("[INFO] Starting camera stream ... ")
vs = WebcamVideoStream(src=0).start()

print("[INFO] Warming up the camera ... ")
time.sleep(2.0)

# loading dnn model
print("[INFO] Loading detection model ...")
prototxt = 'deploy.prototxt'
caffe_model = 'dnn_model.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# loading recognizer model
print("[INFO] Loading recognizer model ...")
model = TripletLossNet()
model = torch.load('pytorch_embedder.pb')

known_faces = list()
known_names = list()

DATA_DIR = 'faces/'
print("[INFO] Loading known faces ... ")

def standardize(a):
    mean = a.mean()
    std = a.std()

    a_std = (a - mean)/std

    return a_std

for (dir, dirs, files) in os.walk(DATA_DIR):
    for file in files:
        abs_path = DATA_DIR + file
        img = cv2.imread(abs_path)
        (H, W) = img.shape[:2]

        # detect the face inside the image
        blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104,111,123])
        net.setInput(blob)
        detections = net.forward()
        face = None

        for i in range(detections.shape[2]):
            confidence = detections[0,0,i,2]
            if(confidence < 0.5):
                continue

            box = np.array([W,H,W,H]) * detections[0,0,i,3:7]
            (startX, startY, endX, endY) = box.astype("int")
            face = img[startY:endY,startX:endX]

            print("    [INFO] Face detected at " + str(abs_path))


        face = cv2.resize(face, (WIDTH, HEIGHT))
        face = np.array([face])
        face = torch.Tensor(face).reshape(1, CHANNELS, HEIGHT, WIDTH)

        embedding = model(face)
        embedding = embedding.detach().numpy()[0]
        #embedding = standardize(embedding)

        label = file.split(".")[0]
        known_faces.append(embedding)
        known_names.append(label)

known_faces = np.array(known_faces)
known_names = np.array(known_names)

# neutralizes the lumination of the image
def lumination_correct(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)

    # applying histogram equalization to the l-channel
    clahe = cv2.createCLAHE(clipLimit = 3.0, tileGridSize=(8,8))
    l_clahe = clahe.apply(l)

    # merge the image again
    lab_clahe = cv2.merge((l_clahe, a, b))

    # convert back to bgr
    bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

    return bgr

def recognize(img, tolerance = 2.0):
    label = "Unkown"
    global model # load the model from outside
    # first, generate the embedding of this face
    # resize image to the size of network's input shape
    face = cv2.resize(img, (WIDTH, HEIGHT))

    face = np.array([face])
    face = torch.Tensor(face).reshape(1, CHANNELS, HEIGHT, WIDTH)

    outputs = model(face)
    outputs = outputs.detach().numpy()[0] # the validating vector
    #outputs = standardize(outputs)

    # now compare to the known faces
    matches = face_recognition.compare_faces(known_faces, outputs, tolerance=tolerance)

    distances = face_recognition.face_distance(known_faces, outputs)
    best_match = np.argmin(distances)
    # print(distances)

    if(matches[best_match]):
        cosine_sim = 1 - cosine(known_faces[best_match], outputs)
        print(cosine_sim)
        if(cosine_sim >= 0.98):
            label = known_names[best_match]

    return label

while(True):
    frame = vs.read()
    # frame = lumination_correct(frame)
    (H, W) = frame.shape[:2]

    # convert image to blob for detection
    blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,111.0,123.0))
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]

        if(confidence < 0.5):
            continue

        box = np.array([W,H,W,H]) * detections[0,0,i,3:7]
        (startX, startY, endX, endY) = box.astype("int")

        face = frame[max(startY,0):min(endY,H), max(startX,0):min(endX,W)]
        label = recognize(face)

        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
        cv2.putText(frame, label, (startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)

    frame = imutils.resize(frame, width=1000)
    cv2.imshow("Frame", frame)

    key = cv2.waitKey(1)

    if(key == ord("q")):
        break

vs.stop()
cv2.destroyAllWindows()