import cv2
import os
import pickle
import numpy as np
from PIL import Image


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir = os.path.join(BASE_DIR, "images")

face_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_frontalface_alt2.xml')
eye_cascade = cv2.CascadeClassifier('cascades\data\haarcascade_eye.xml')

recognizer = cv2.face.LBPHFaceRecognizer_create()


current_id = 0
label_ids = {}
y_labels = []
x_train = []
ey_labels = []
ex_train = []

#Show images inside the images folder
for root, dirs, files in os.walk(image_dir):
    for file in files:
        if file.endswith('png') or file.endswith('jpg'):
            path = os.path.join(root,file)
            label = os.path.basename(root).replace(" ","-").lower()

            #Validate: Same id in image save folder
            if not label in label_ids:
                label_ids[label] = current_id
                current_id += 1
            id_ = label_ids[label]

            #Convert Images into numbers in Array
            pil_image = Image.open(path).convert("L") #GrayScale
            size = (550,550)
            final_image = pil_image.resize(size, Image.ANTIALIAS) #Change size of image
            image_array = np.array(final_image, "uint8")

            #face and eyes detection
            faces = face_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            eyes = eye_cascade.detectMultiScale(image_array, scaleFactor=1.5, minNeighbors=5)
            for (x,y,w,h) in faces:
                roi = image_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
                for (ex,ey,ew,eh) in eyes:
                    e_roi = image_array[ey:ey+eh, ex:ex+ew]
                    x_train.append(roi)
                    y_labels.append(id_)                    

                


with open("labels.pickle", "wb") as f:
    pickle.dump(label_ids, f)

#Training the face recognizer
recognizer.train(x_train, np.array(y_labels))
recognizer.save("trainner.yml")

print('Train done!')