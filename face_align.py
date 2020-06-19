import os
import cv2
import imutils
import dlib

import numpy as np
from imutils import face_utils

prototxt = 'deploy.prototxt'
caffemodel = 'dnn_model.caffemodel'
dat_68_landmark = 'shape_predictor_68_face_landmarks.dat'

face_detector = cv2.dnn.readNetFromCaffe(prototxt, caffemodel)
shape_predictor = dlib.shape_predictor(dat_68_landmark)

'''
	This function will both align the face, 
	perform preprocessing steps like illumination correction
	and resizing for the face image
'''

### Returns a list of align faces ### 
def align(img, width=128, height=128, operation='clahe', desiredLeftEye=(0.2,0.2)):
	def lumination_correct(img):
		lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
		l, a, b = cv2.split(lab)

		clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
		l_clahe = clahe.apply(l)

		lab_clahe = cv2.merge((l_clahe, a, b))
		bgr = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)

		return bgr 

	if(operation == 'clahe'):
		img = lumination_correct(img)
	elif(operation == 'gray'):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	elif(operation == 'rgb'):
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

	### Face Alignment Process ###
	'''
		1. Calculate left eyes and right eye coordinates
		2. Calculate the rotation angle using arc tan of dx and dy
		3. Get a rotation matrix
		4. warpAffine
	'''

	# Extact the face
	(H, W) = img.shape[:2]
	blob = cv2.dnn.blobFromImage(img, 1.0, (300,300), (104,111,123))
	face_detector.setInput(blob)
	detections = face_detector.forward() 

	faces = list()
	faces_locations = list()

	for i in range(0, detections.shape[2]):
		confidence = detections[0,0,i,2]
		if(confidence < 0.50):
			continue

		box = detections[0,0,i,3:7] * np.array([W,H,W,H])
		(startX, startY, endX, endY) = box.astype("int")
		faces_locations.append((startX, startY, endX, endY))
		face_width = endX - startX
		face_height = endY - startY

		# must convert to dlib.rectangle
		# because shape_predictor accepts dlib.rectangle only
		rect = dlib.rectangle(startX, startY, endX, endY)

		# for each face, we align them
		face = img[startY:endY, startX:endX]

		# convert the landmarks of right eye and left eye to x,y coordinates
		shape = shape_predictor(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), rect)
		shape = face_utils.shape_to_np(shape)

		(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS['left_eye']
		(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

		leftEyePts = shape[lStart:lEnd]
		rightEyePts = shape[rStart:rEnd]

		# compute the center of mass for each eye
		leftEyeCenter = leftEyePts.mean(axis=0).astype("int")
		rightEyeCenter = rightEyePts.mean(axis=0).astype("int")

		# compute the angle between eye centroids
		dY = rightEyeCenter[1] - leftEyeCenter[1]
		dX = rightEyeCenter[0] - leftEyeCenter[0]
		angle = np.degrees(np.arctan2(dY, dX)) - 180

		# compute the desired right eye x-coordinate based on the
		# desired x-coordinate of the left eye
		desiredRightEyeX = 1.0 - desiredLeftEye[0]
		# determine the scale of the new resulting image by taking
		# the ratio of the distance between eyes in the *current*
		# image to the ratio of distance between eyes in the
		# *desired* image
		dist = np.sqrt((dX ** 2) + (dY ** 2))
		desiredDist = (desiredRightEyeX - desiredLeftEye[0])
		desiredDist *= face_width
		scale = desiredDist / dist


		# compute center (x, y)-coordinates (i.e., the median point)
		# between the two eyes in the input image
		eyesCenter = ((leftEyeCenter[0] + rightEyeCenter[0]) // 2, (leftEyeCenter[1] + rightEyeCenter[1]) // 2)
		# grab the rotation matrix for rotating and scaling the face
		M = cv2.getRotationMatrix2D(eyesCenter, angle, scale)
		# update the translation component of the matrix
		tX = face_width * 0.5
		tY = face_height * desiredLeftEye[1]
		M[0, 2] += (tX - eyesCenter[0])
		M[1, 2] += (tY - eyesCenter[1])

		# apply the affine transformation
		(w, h) = (face_width, face_height)
		output = cv2.warpAffine(img, M, (w, h), flags=cv2.INTER_CUBIC)

		output = cv2.resize(output, (width, height))

		faces.append(output)


	
	return faces, faces_locations

'''
img = cv2.imread("faces/hieu/hieu_3.jpg")
faces = align(img, operation=None)

for face in faces:
	cv2.imshow("Frame", face)
	cv2.waitKey(0)

cv2.destroyAllWindows()
''' 

