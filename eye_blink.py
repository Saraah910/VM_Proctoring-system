from datetime import datetime,timedelta
from tracemalloc import start
from typing import Counter
from scipy.spatial import distance as dist
from imutils.video import FileVideoStream
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import imutils
import wget
import time
import dlib
import cv2
import os
import threading
import schedule
import math
import csv
import json
def eye_aspect_ratio(eye):
	
	A = dist.euclidean(eye[1], eye[5])
	B = dist.euclidean(eye[2], eye[4])

	
	C = dist.euclidean(eye[0], eye[3])

	ear = (A + B) / (2.0 * C)

	return ear



EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 3

COUNTER = 0
TOTAL = 0
rate = 0

print("[INFO] loading facial landmark predictor...")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_68.dat")


(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]


cap = cv2.VideoCapture(0)
start_time = datetime.now()
target = start_time + timedelta(seconds=10)
print(start_time.strftime("%M:%S"),target.strftime("%M:%S"))


def blinkrate():
	dict = {}
	now = datetime.now()
	dict[TOTAL] = now.strftime("%M:%S")
	print(dict)
	if dict[TOTAL] == target:
		return TOTAL,k



fps = 0
while True:
	ret,frame = cap.read()
	gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

	fps+=1
	rects = detector(gray, 0)
	frame_count = int(cap. get(cv2. CAP_PROP_FRAME_COUNT))

	for rect in rects:
		
		shape = predictor(gray, rect)
		shape = face_utils.shape_to_np(shape)

		leftEye = shape[lStart:lEnd]
		rightEye = shape[rStart:rEnd]
		leftEAR = eye_aspect_ratio(leftEye)
		rightEAR = eye_aspect_ratio(rightEye)


		ear = (leftEAR + rightEAR) / 2.0

		leftEyeHull = cv2.convexHull(leftEye)
		rightEyeHull = cv2.convexHull(rightEye)
		cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
		cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

		if ear < EYE_AR_THRESH:
			COUNTER += 1
		
		else:
			
			if COUNTER >= EYE_AR_CONSEC_FRAMES:
				TOTAL += 1
			
				rate = math.sqrt((TOTAL/COUNTER)*60)

			COUNTER = 0

		

		cv2.putText(frame, "Blinks: {}".format(TOTAL), (10, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
		cv2.putText(frame, "EAR: {:.2f}".format(ear), (300, 30),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 200, 0), 2)
		# k = 0		
		# blinkrate()
		# k = TOTAL
		# print(k)
		
		
	
		# r = Blink_rate(TOTAL)
		cv2.putText(frame, "Blink rate: {:.2f}".format(rate), (10, 70),
			cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
	
	data = {
		'Blink_Count': TOTAL,
		'Blink_Rate': rate,
		'frame_count': fps
	}
	
	with open('sample.json','r+') as file:
		file_data = json.load(file)
		file_data["Eye_data"].append(data)
		file.seek(0)
		json.dump(file_data, file, indent = 4)

	cv2.imshow("Frame", frame)
	key = cv2.waitKey(1) & 0xFF


	if key == ord("q"):
		break


cv2.destroyAllWindows()
vs.stop()

