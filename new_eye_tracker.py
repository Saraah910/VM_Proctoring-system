import dlib
import cv2
import numpy as np
from scipy.spatial import distance


detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_68.dat")

def calculate_EAR(eye):
    A = distance.euclidean(eye[1], eye[5])
    B = distance.euclidean(eye[2], eye[4])
    C = distance.euclidean(eye[0], eye[3])
    ear_aspect_ratio = (A+B)/(2.0*C)
    return ear_aspect_ratio

def is_close(y0,y1): 
    if abs(y0 - y1) < 10:
        return True
    return False

def get_center(gray_img):
    moments = cv2.moments(gray_img, False)
    try:
        
        return int(moments['m10']/moments['m00']), int(moments['m01'] / moments['m00'])
    except:
        return None

def p(img, parts, eye):
    if eye[0]:
        cv2.circle(img, eye[0], 3, (255,255,0), -1)
    if eye[1]:
        cv2.circle(img, eye[1], 3, (255,255,0), -1)  
    for i in parts:
        cv2.circle(img, (i.x, i.y), 3, (255, 0, 0), -1)

    cv2.imshow("me", img)  

def get_eye_parts(parts, left = True):
    if left:
        eye_parts = [
                parts[36],
                min(parts[37],parts[38], key=lambda x: x.y),
                max(parts[40],parts[41], key=lambda x: x.y),
                parts[39],
               ]
    else:
        eye_parts = [
                parts[42],
                min(parts[43],parts[44], key = lambda x: x.y),
                max(parts[46],parts[47], key=lambda x: x.y),
                parts[45],
               ]
    return eye_parts



def get_eye_image(img, parts, left = True): 
    if left:
        eyes = get_eye_parts(parts, True)
    else:
        eyes = get_eye_parts(parts, False)
    org_x = eyes[0].x
    org_y = eyes[1].y

    if is_close(org_y, eyes[2].y):
        return None
    eye = img[org_y:eyes[2].y, org_x:eyes[-1].x] #
    height = eye.shape[0]
    width = eye.shape[1]
    resize_eye = cv2.resize(eye , (int(width*5.0), int(height*5.0)))
    
    return eye

def get_eye_center(img, parts, left = True): 
        if left:
            eyes = get_eye_parts(parts, True)
        else:
            eyes = get_eye_parts(parts, False) 

        x_center = int(eyes[0].x + (eyes[-1].x - eyes[0].x)/2)
        y_center = int(eyes[1].y + (eyes[2].y - eyes[1].y)/2)

        cv2.circle(img, (x_center, y_center), 3, (255,255,0), -1)
        return x_center, y_center

def get_pupil_location(img, parts, left = True):
     if left:
            eyes = get_eye_parts(parts, True)
     else:
            eyes = get_eye_parts(parts, False)
     org_x = eyes[0].x
     org_y = eyes[1].y
     if is_close(org_y, eyes[2].y):
        return None
     eye = img[org_y:eyes[2].y, org_x:eyes[-1].x]
     _, threshold_eye = cv2.threshold(cv2.cvtColor(eye, cv2.COLOR_RGB2GRAY),45, 255, cv2.THRESH_BINARY_INV)#第一引数を無視して二値化
     
     height = threshold_eye.shape[0]
     width = threshold_eye.shape[1]
     resize_eye = cv2.resize(threshold_eye , (int(width*5.0), int(height*5.0)))

     
     center = get_center(threshold_eye)

     if center:
         cv2.circle(img, (center[0] + org_x, center[1] + org_y), 3, (255, 0, 0), -1)
         return center[0] + org_x, center[1] + org_y
     return center

def calculate_relative_pupil_position(img,eye_center, pupil_locate, left = True): 
    if not eye_center:
        return
    if not pupil_locate:
        return
    
    relative_pupil_x = pupil_locate[0] - eye_center[0]
    relative_pupil_y = pupil_locate[1] - eye_center[1]
    
    return relative_pupil_x, relative_pupil_y

def calculate_direction(img, parts, pupil_locate):
    if not pupil_locate:
        return

    eyes = get_eye_parts(parts, True)
    
    left_border = eyes[0].x + (eyes[3].x - eyes[0].x)/3 
    right_border = eyes[0].x  + (eyes[3].x - eyes[0].x) * 2/3 
    up_border = eyes[1].y + (eyes[2].y - eyes[1].y)/3 
    down_border = eyes[1].y + (eyes[2].y - eyes[1].y) * 2/3
    
    if eyes[0].x <= pupil_locate[0] < left_border:
        
        show_text(img,"LOOKING LEFT",50,50)
    elif left_border <= pupil_locate[0] <= right_border:
        
        show_text(img,"LOOKING STRAIGHT",50,50) 
    elif right_border <= pupil_locate[0] <= eyes[3].x :
        
        show_text(img,"LOOKING RIGHT",50,50) 
    else:
        
        show_text(img,"NONE",50,50) 
    
    if pupil_locate[1] <= up_border:
        
        show_text(img, "UP", 50, 100)
    elif up_border <= pupil_locate[1] <= down_border:
        
        show_text(img, "MIDDLE", 50, 100)
    # elif pupil_locate[1] >= down_border:
        
    #     show_text(img, "DOWN", 50, 100)
    return


def show_text(img, text, x, y):
    cv2.putText(img,
            text,
            org=(x, y),
            fontFace=cv2.FONT_HERSHEY_SIMPLEX,
            fontScale=1.0,
            color=(250, 0, 0),
            thickness=2,
            lineType=cv2.LINE_4)
    return


cap = cv2.VideoCapture(0)
while True:
   ret, frame = cap.read() 
   grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   faces = detector(grey)

   for face in faces:
        face_landmarks = predictor(grey,face)
        lefteye = []
        righteye = []
        for n in range(36,42):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            lefteye.append((x,y))
            next_point = n+1
            if n == 41:
                next_point = 36
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            

        for n in range(42,48):
            x = face_landmarks.part(n).x
            y = face_landmarks.part(n).y
            righteye.append((x,y))
            next_point = n+1
            if n == 47:
                next_point = 42
            x2 = face_landmarks.part(next_point).x
            y2 = face_landmarks.part(next_point).y
            
        left_ear = calculate_EAR(lefteye)
        right_ear = calculate_EAR(righteye)

        EAR = (left_ear + right_ear)/1.5
        EAR = round(EAR,2)

        if EAR < 0.26:
            show_text(frame,"Looking Down",50,50)
            show_text(frame,"Please focus at the centre",50,100)
            
        
   dets = detector(frame[:, :, ::-1])
   if len(dets) > 0:
       parts = predictor(frame, dets[0]).parts()
       
       left_eye_image =get_eye_image(frame,parts, True)
       right_eye_image = get_eye_image(frame,parts,False)
       left_eye_center = get_eye_center(frame,parts, True)
       right_eye_center = get_eye_center(frame,parts, False)
       left_pupil_location = get_pupil_location(frame, parts, True)
       right_pupil_location = get_pupil_location(frame, parts, False)
       left_relative_pupil_position = calculate_relative_pupil_position(frame, left_eye_center,left_pupil_location, True)
       right_relative_pupil_position = calculate_relative_pupil_position(frame, right_eye_center,right_pupil_location, False)
       calculate_direction(frame,parts,left_pupil_location)
    
       cv2.imshow("me", frame)
       
   key = cv2.waitKey(1) 

   if key == 27: 
       break
 
 
cap.release()
cv2.destroyAllWindows()