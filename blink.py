import cv2 # for video rendering
import dlib # for face and landmark detection
import imutils
# for calculating dist b/w the eye landmarks
from scipy.spatial import distance as dist
# to get the landmark ids of the left and right eyes
# you can do this manually too
from imutils import face_utils
import time
import datetime
import math
cam = cv2.VideoCapture(0)


# defining a function to calculate the EAR
def calculate_EAR(eye):

	# calculate the vertical distances
	y1 = dist.euclidean(eye[1], eye[5])
	y2 = dist.euclidean(eye[2], eye[4])

	# calculate the horizontal distance
	x1 = dist.euclidean(eye[0], eye[3])

	# calculate the EAR
	EAR = (y1+y2) / x1
	return EAR

firstblink = 0
secondblink = 0
# Variables
blink_thresh = 0.75
succ_frame = 2
count_frame = 0
eyeblink = 0
# Eye landmarks
(L_start, L_end) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(R_start, R_end) = face_utils.FACIAL_LANDMARKS_IDXS['right_eye']

# Initializing the Models for Landmark and
# face Detection
detector = dlib.get_frontal_face_detector()
landmark_predict = dlib.shape_predictor('shape_68.dat')
rate = 0

while 1:

	# If the video is finished then reset it
	# to the start
	if cam.get(cv2.CAP_PROP_POS_FRAMES) == cam.get(
			cv2.CAP_PROP_FRAME_COUNT):
		cam.set(cv2.CAP_PROP_POS_FRAMES, 0)

	else:
		_, frame = cam.read()
		frame = imutils.resize(frame, width=640)
		firstblink = datetime.datetime.strptime(str(datetime.datetime.now().strftime("%H-%M-%S")),"%H-%M-%S")

		# converting frame to gray scale to
		# pass to detector
		img_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

		# detecting the faces
		faces = detector(img_gray)
		for face in faces:

			# landmark detection
			shape = landmark_predict(img_gray, face)

			# converting the shape class directly
			# to a list of (x,y) coordinates
			shape = face_utils.shape_to_np(shape)

			# parsing the landmarks list to extract
			# lefteye and righteye landmarks--#
			lefteye = shape[L_start: L_end]
			righteye = shape[R_start:R_end]

			# Calculate the EAR
			left_EAR = calculate_EAR(lefteye)
			right_EAR = calculate_EAR(righteye)

			# Avg of left and right eye EAR
			avg = (left_EAR+right_EAR)/2
			print('Average',avg)
			if avg < blink_thresh:

				print("Count",count_frame)
				count_frame += 1 # incrementing the frame count
			else:

				if count_frame >= succ_frame:
					eyeblink+=1
					# blinkrate = datetime.datetime.strptime(str(datetime.datetime.now().strftime("%H-%M-%S")),"%H-%M-%S") - firstblink
					rate = math.sqrt((eyeblink/count_frame)*60)
				count_frame = 0

		cv2.putText(frame, f'Blink Detected:{eyeblink}\nBlink rate:{rate}', (30, 30),
					cv2.FONT_HERSHEY_DUPLEX, 1, (0, 200, 0), 1)

		cv2.imshow("Video", frame)
		if cv2.waitKey(5) & 0xFF == ord('q'):
			break

cam.release()
cv2.destroyAllWindows()