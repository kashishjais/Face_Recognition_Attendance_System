import cv2
import numpy as np
import face_recognition
import os
path='images'
imagelist=[]
classNames=[]
myList=os.listdir(path)
print(myList)
for cl in myList:
    curImg=cv2.imread(f'{path}/{cl}')
    imagelist.append(os.path.splitext(cl)[0])
print(classNames)    