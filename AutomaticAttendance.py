from datetime import datetime

import face_recognition
import cv2
import numpy as np
import os

path = 'Dataset'
listImages = os.listdir(path)
#print(listImages)
images = []
names = []
for ind in listImages:
    currimage = cv2.imread(f'{path}/{ind}')
    images.append(currimage)
    names.append(os.path.splitext(ind)[0])
#print(names)

def findEncodings(images):
    encodeList = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);
        #faceLoc = face_recognition.face_locations(img)[0]
        encode = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        #cv2.rectangle(img, (faceLoc[3], faceLoc[0]), (faceLoc[1], faceLoc[2]), (255, 0, 255), 2)
    return encodeList

def markAttendance(name):
    with open('AttendanceSheet.csv','r+') as f:
        myDataList = f.readlines()
        #print(myDataList)
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            timeString = now.strftime('%H:%M:%S')
            dateString = now.strftime('%d/%m/%Y')
            f.writelines(f'\n{name},{timeString},{dateString}')

#markAttendance('a')

allEncodings = findEncodings(images)
#print(len(allEncodings))
print('Encodings completed.')

cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    imgSmall = cv2.resize(img, (0,0), None, 0.25,0.25)
    imgSmall = cv2.cvtColor(img, cv2.COLOR_BGR2RGB);

    currFaceFrameLoc = face_recognition.face_locations(imgSmall)
    currFaceFrameEncode = face_recognition.face_encodings(imgSmall, currFaceFrameLoc)

    for faceLoc, faceEncode in zip(currFaceFrameLoc, currFaceFrameEncode):
        matches = face_recognition.compare_faces(allEncodings,faceEncode)
        faceDistance = face_recognition.face_distance(allEncodings,faceEncode)
        print(faceDistance)
        matchIndex = np.argmin(faceDistance)

        if matches[matchIndex]:
            name = names[matchIndex].upper()
            print(name)
            y1,x2,y2,x1 = faceLoc
            #y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            markAttendance(name)

    cv2.imshow('Webcam',img)
    cv2.waitKey(1)