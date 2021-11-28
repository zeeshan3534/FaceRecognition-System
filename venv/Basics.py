import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

path='ImagesAttendance'
images = []
classNames = []
#this will taking out all the images from the given path already define
myList = os.listdir(path)
for cls in myList:
    #reading the file from teh given path
    curImg = cv2.imread(f'{path}/{cls}')
    images.append(curImg)
    #its taking the file name but without extension
    classNames.append(os.path.splitext(cls)[0])


#its will compute all the encoding for us
def findEncodings(images):
    encodeList =[]
    for img in images:
        #first step Conversion BGR to RGB
        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        #second Step finding the encoding
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)

    return encodeList
encodeListKnown=findEncodings(images)
print('Encoding Complete')

#initializing webcame

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    #reducing the size of image for speeding up the process
    imgS= cv2.resize(img,(0,0),None,0.25,0.25)
    #now converting the webcame image to RGB
    imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

    #finding the location
    facesCurFrame = face_recognition.face_locations(imgS)
    # encoding for webcame image
    encodeCurFrame = face_recognition.face_encodings(imgS,facesCurFrame)
    #Finding the matchs for this we iterate all the faces

    #one by one it will grab faces and than grab encoding
    for encodeFace,faceLoc in zip(encodeCurFrame,facesCurFrame):
        matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
        #it will return the list as well and return 3 values because we have 3 values in the dir
        faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)
        matchIndex = np.argmin(faceDis)

        if matches[matchIndex]:
            name = classNames[matchIndex].upper()
            y1,x2,y2,x1= faceLoc
            y1, x2, y2, x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img,(x1,y1),(x2,y2),(0,255,0),2)
            cv2.rectangle(img,(x1,y2-35),(x2,y2),(0,255,0),cv2.FILLED)
            cv2.putText(img,name,(x1+6,y2-6),cv2.FONT_HERSHEY_COMPLEX,1,(255,255,255),2)



    cv2.imshow('webcam',img)
    cv2.waitKey(1)








