import os
import cv2
import torch
import time
import imutils
import numpy as np
import face_recognition
import matplotlib.pyplot as plt

from triplet_net import TripletLossNet
from arcface_net import ArcFaceNet
from scipy.spatial.distance import cosine
from imutils.video import WebcamVideoStream
from sklearn.ensemble import RandomForestClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn.decomposition import PCA

from argparse import ArgumentParser

parser = ArgumentParser()
parser.add_argument("-v", "--video", required = False, help = "Path to input video")

args = vars(parser.parse_args())

HEIGHT, WIDTH, CHANNELS = 128, 128, 3

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
model = ArcFaceNet()
model.load_state_dict(torch.load('arcface_pytorch.pt', map_location=torch.device('cpu')))
model.eval()

# include a classifier
#clf = SVC(kernel='rbf', C = 1.0, class_weight='balanced', probability=True, gamma='auto')
#clf = LinearDiscriminantAnalysis(n_components=2)
clf = LinearSVC()

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

for (dir, dirs, files) in os.walk(DATA_DIR):
	if(dir != DATA_DIR or dir == DATA_DIR):
		for file in files:
			abs_path = dir + "/" + file
			img = cv2.imread(abs_path)
			#img = lumination_correct(img)
			img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
			img2 = cv2.resize(img, (0,0), fx=0.5,fy=0.5)
			(H, W) = img.shape[:2]

			# detect the face inside the image
			blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), [104,111,123])
			net.setInput(blob)
			detections = net.forward()
			face = None
			face3 = None
			num_faces = 0

			for i in range(detections.shape[2]):
				confidence = detections[0,0,i,2]
				if(confidence < 0.5):
					continue

				num_faces += 1
				box = np.array([W,H,W,H]) * detections[0,0,i,3:7]
				(startX, startY, endX, endY) = box.astype("int")
				face = img[startY:endY,startX:endX]

				box2 = np.array([W/2,H/2,W/2,H/2])*detections[0,0,i,3:7]
				(startX, startY, endX, endY) = box2.astype("int")
				face3 = img2[startY:endY,startX:endX]

				print("    [INFO] Face detected at " + str(abs_path))


			# cv2.imshow("Face", face)
			#cv2.waitKey(0)
			if(num_faces == 0): continue
			face1 = cv2.resize(face, (WIDTH, HEIGHT))
			face1 = np.array([face1])
			face1 = torch.Tensor(face1).reshape(1, CHANNELS, HEIGHT, WIDTH)

			embedding = model(face1)
			embedding = embedding.detach().numpy()[0]
			embedding = normalize(embedding)
			#embedding = standardize(embedding)
			embedding = embedding / (np.linalg.norm(embedding)**0.5)
			
			label = file.split(".")[0]
			label = label.split("_")[0]
			known_faces.append(embedding)
			known_names.append(label)
			
			face2 = cv2.resize(cv2.flip(face, flipCode=1), (WIDTH, HEIGHT))
			face2 = np.array([face2])
			face2 = torch.Tensor(face2).reshape(1, CHANNELS, HEIGHT, WIDTH)

			embedding = model(face2)
			embedding = embedding.detach().numpy()[0]
			embedding = normalize(embedding)
			#embedding = standardize(embedding)
			embedding = embedding / (np.linalg.norm(embedding)**0.5)

			label = file.split(".")[0]
			label = label.split("_")[0]
			known_faces.append(embedding)
			known_names.append(label)

			face3 = cv2.resize(face3, (WIDTH, HEIGHT))
			face3 = np.array([face3])
			face3 = torch.Tensor(face3).reshape(1, CHANNELS, HEIGHT, WIDTH)

			embedding = model(face3)
			embedding = embedding.detach().numpy()[0]
			embedding = normalize(embedding)
			#embedding = standardize(embedding)
			embedding = embedding / (np.linalg.norm(embedding)**0.5)

			label = file.split(".")[0]
			label = label.split("_")[0]
			known_faces.append(embedding)
			known_names.append(label)


known_faces = np.array(known_faces)
known_names = np.array(known_names)

print(known_names)

pca = PCA(n_components = 3)
out = pca.fit_transform(known_faces)
ax = plt.axes(projection='3d')

for label in np.unique(known_names):
	cluster = out[np.where(known_names == label)]
	x = cluster[:,0]
	y = cluster[:,1]
	z = cluster[:,2]


	ax.scatter3D(x, y, z, alpha=0.3, label=label)

clf.fit(known_faces, known_names)

def cross_validate(enc, known_enc):
	dist = known_enc - enc 
	dist = np.delete(dist, 0)

	dist = np.sort(dist)
	min_dist = dist[0]

	return min_dist

def recognize(img, tolerance = 0.1):
	label = "Unkown"
	global model # load the model from outside
	global pca
	# first, generate the embedding of this face
	# resize image to the size of network's input shape
	#face = lumination_correct(img)
	face = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
	face = cv2.resize(face, (WIDTH, HEIGHT))

	# cv2.imshow("Face_", face)
	# cv2.waitKey(0)

	face = np.array([face])
	face = torch.Tensor(face).reshape(1, CHANNELS, HEIGHT, WIDTH)

	outputs = model(face)
	outputs = outputs.detach().numpy()[0] # the validating vector
	outputs = normalize(outputs)
	point = pca.transform([outputs])

	#outputs = standardize(outputs)
	outputs = outputs / (np.linalg.norm(outputs)**0.5)

	# now compare to the known faces
	
	matches = face_recognition.compare_faces(known_faces, outputs, tolerance=0.3)


	distances = face_recognition.face_distance(known_faces, outputs)
	print(distances)
	# distances = distances / sum(distances)
	best_match = np.argmin(distances)
	
	label = 'Unknown'
	if(matches[best_match]):
		cosine_sim = 1 - cosine(known_faces[best_match], outputs)
		#print(cosine_sim)
		#mean_dist = np.mean(distances)
		#min_dist = cross_validate(distances[best_match], distances)
		#if(distances[best_match] > 1.5 * mean_dist):
		#    label = known_names[best_match]
		#else:
		if(cosine_sim >= 0.99):
			label = known_names[best_match]
		
	#label_ = str(clf.predict(np.array([outputs]))[0])

	#_label_ = 'Unkown'
	#if(label == label_):
	#	_label_ = label
	#proba = clf.predict_proba(np.array([outputs]))

	#label = label[0] + " - {0:.1f}%".format(proba[0][np.argmax(proba[0])]*100)
	
	return label, point

print("-------------------------------------------------")
print("[INFO] Running recognition app ... ")
while(True):
	if(not video):
		frame = vs.read()
		frame = cv2.flip(frame, flipCode=0)
	else:
		ret, frame = vs.read()

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
		label, point = recognize(face)

		ax.scatter3D(point[:,0], point[:,1], point[:,2], color='brown', alpha=1.0)
		cv2.rectangle(frame, (startX, startY), (endX, endY), (0,255,0), 2)
		cv2.putText(frame, label, (startX,startY), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,255), 2)

	frame = imutils.resize(frame, width=1000)
	cv2.imshow("Frame", frame)

	key = cv2.waitKey(1)

	if(key == ord("q")):
		break

if(not video):
	vs.stop()
else: 
	vs.release()

cv2.destroyAllWindows()
plt.legend()
plt.show()

print("-------------------------------------------------")
print("[INFO] App stopped ...")
