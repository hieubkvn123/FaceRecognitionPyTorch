import os
import cv2
import torch
import pickle
import time
import imutils
import threading
import numpy as np
import insightface
import face_recognition
import matplotlib.pyplot as plt

from triplet_net import TripletLossNet
from scipy.spatial.distance import cosine
from imutils.video import WebcamVideoStream
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.decomposition import PCA

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", "--video", required = False, help = "Path to input video")

args = vars(parser.parse_args())

# Some constants
IMG_WIDTH = 112
IMG_HEIGHT = 112

print("[INFO] Starting camera stream ... ")

video = False
if(args['video'] == None):
    vs = WebcamVideoStream(src=0).start()
else:
    vs = cv2.VideoCapture(args['video'])
    video = True

print("[INFO] Warming up the camera ... ")
time.sleep(2.0)

# loading dnn model
print("[INFO] Loading detection model ...")
prototxt = 'deploy.prototxt'
caffe_model = 'dnn_model.caffemodel'
net = cv2.dnn.readNetFromCaffe(prototxt, caffe_model)

# loading recognizer model
print("[INFO] Loading recognizer model ...")
model = insightface.model_zoo.get_model('arcface_r100_v1')
model.prepare(ctx_id = -1)


known_faces = list()
known_names = list()

DATA_DIR = 'faces/'
print("[INFO] Loading known faces ... ")

def normalize(a):
    length = np.linalg.norm(a)
    a_norm = a/length
    return a_norm

def standardize(a):
    mean = a.mean()
    std = a.std()

    a_std = (a - mean)/std

    # then normalize the vector : v = v / ||v||
    # length = np.linalg.norm(a)

    return a_std

if(not os.path.exists("faces_arc.pickle") or not os.path.exists("labels_arc.pickle")):
    for (dir, dirs, files) in os.walk(DATA_DIR):
        if(dir != DATA_DIR or dir == DATA_DIR):
            for file in files:
                abs_path = dir + "/" + file

                label = abs_path.split("/")[::-1][0]
                label = label.split(".")[0]
                label = label.split("_")[0]

                img = cv2.imread(abs_path)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                (H, W) = img.shape[:2]

                # detect the face inside the image
                blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104,111,123])
                net.setInput(blob)
                detections = net.forward()
                face = None

                for i in range(detections.shape[2]):
                    confidence = detections[0,0,i,2]
                    if(confidence < 0.8):
                        continue

                    box = np.array([W,H,W,H]) * detections[0,0,i,3:7]
                    (startX, startY, endX, endY) = box.astype("int")
                    face = img[startY:endY,startX:endX]

                    print("    [INFO] Face detected at " + str(abs_path))


                # cv2.imshow("Face", face)
                #cv2.waitKey(0)

                face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
                embedding = model.get_embedding(face)[0]
                embedding = normalize(embedding)

                known_faces.append(embedding)
                known_names.append(label)



    known_faces = np.array(known_faces)
    known_names = np.array(known_names)

    pickle.dump(known_faces, open("faces_arc.pickle", "wb"))
    pickle.dump(known_names, open("labels_arc.pickle", "wb"))

else:
    known_faces = pickle.load(open("faces_arc.pickle", "rb"))
    known_names = pickle.load(open("labels_arc.pickle", "rb"))
#vs = WebcamVideoStream(src = 0).start()
PROCESS_FRAME = True

# to retain locations and names after each processing
face_locations = []
names = []



while(True):
    frame = vs.read()
    frame = cv2.flip(frame, flipCode=0)
    (H, W) = frame.shape[:2]

    if(PROCESS_FRAME):
        face_locations = []
        names = []
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300,300), (104,111,123))
        net.setInput(blob)
        detections = net.forward()

        for i in range(0, detections.shape[2]):
            name = "Unknown"
            confidence = detections[0,0,i,2]
            if(confidence < 0.5):
                continue

            box = detections[0,0,i,3:7] * np.array([W,H,W,H])
            (startX, startY, endX, endY) = box.astype("int")
            # print(startX, startY, endX, endY)

            startX, startY = max(0, startX), max(0,startY)
            endX, endY = min(endX, W), min(endY, H)

            face = frame[startY:endY, startX:endX]
            face = cv2.resize(face, (IMG_WIDTH, IMG_HEIGHT))
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            emb = model.get_embedding(face)
            emb = normalize(emb)

            matches = face_recognition.compare_faces(known_faces, emb, tolerance = 1.0)
            distances = face_recognition.face_distance(known_faces, emb)

            # print(distances)

            best = np.argmin(distances)

            if(matches[best]):
                #enc = known_embeddings[best]
                #sim = cosine(enc, emb)

                #print(sim)
                name = known_names[best]

            face_locations.append((startX, startY, endX, endY))
            names.append(name)

    PROCESS_FRAME = not PROCESS_FRAME

    for (startX, startY, endX, endY), name in zip(face_locations, names):
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 1)
        cv2.putText(frame, name, (startX, startY), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0,0,255), 2)

    cv2.imshow("Frame", frame)
    key = cv2.waitKey(1)

    if(key == ord("q")):
        break


vs.stop()
cv2.destroyAllWindows()
print("[INFO] Program ended ... ")
