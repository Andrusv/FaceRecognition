import numpy as np
import cv2
import pickle

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()

#load the name of the person
labels = {"person_name":1}
with open("labels.pickle", "rb") as f:
    og_labels = pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}

#Recognizing trainned data
recognizer.read("trainner.yml")

cap = cv2.VideoCapture(0)

#Functions
def person_name_in_frame(label_name,conf):
    conf = round(conf,2)
    font = cv2.FONT_ITALIC
    frame_text = label_name+' '+str(conf)
    color = (255,255,255)
    stroke = 2
    cv2.putText(frame, str(frame_text), (x,y-10), font, 0.6, color, stroke, cv2.LINE_AA)
    print(label_name+' '+str(conf))


#Cam frames and operations
while(True):
    ret, frame = cap.read()

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    eyes = eye_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
        #scaleFactor: if higher, the scan would be more accurate
        #minNeighbors: if less, detection become more sensitive
    
    for (x,y,w,h) in faces:        
        #Face detection in grayscale
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        #Recognizer algorithm
        id_, conf = recognizer.predict(roi_gray)

        #Name of the person in frame
        if conf >= 60:
            person_name_in_frame(labels[id_],conf)
        else:
            person_name_in_frame('Unknown',conf)

        #Take a picture of your face: Change roi_color to gray if you want the photo with colors 
        img_item = "my-photo.png"
        cv2.imwrite(img_item, roi_color)

        #Drawn rectangle
        color = (255,0,0)   #Rectangle color BGR -> 0-255
        stroke = 2          #How thick our lines are going to be
        end_cord_x = x + w
        end_cord_y = y + h
        cv2.rectangle(frame, (x,y), (end_cord_x,end_cord_y), color, stroke)

        #Eye detection
        for (ex, ey, ew, eh) in eyes:
            eroi_gray = gray[ey:ey+eh, ex:ex+ew]
            eroi_color = frame[ey:ey+eh, ex:ex+ew]

            id_, econf = recognizer.predict(eroi_gray)
            end_cord_ex = ex + ew
            end_cord_ey = ey + eh
            if econf >= 95:
                cv2.rectangle(frame, (ex,ey), (end_cord_ex,end_cord_ey), (0,255,0), stroke)


    cv2.imshow('WebCam1',frame)
    if cv2.waitKey(20) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
