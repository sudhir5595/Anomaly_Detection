import numpy as np
import cv2 as cv
import sys
import cv2
import os
import glob
from PIL import Image
import imutils

img_dir="/home/yogesh/Desktop/computer_vision_sub/project/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test006"
#img_dir="/home/yogesh/Desktop/computer_vision_sub/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"
data_path = os.path.join(img_dir,'*f')
files = sorted(glob.glob(data_path))
data = []
crop=[]

for f1 in files:
    img = Image.open(f1)
    img = np.array(img) 
    data.append(img)
old_frame=None
for frame in data:
    gray = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    #print(gray.shape)
    #gray = cv2.GaussianBlur(gray, (21, 21), 0)
    #fgmask = fgbg.apply(frame)
    if old_frame is None:
        old_frame = gray
        
        continue
   
    frameDelta = cv2.absdiff(old_frame, gray)
    thresh = cv2.threshold(frameDelta, 25, 255, cv2.THRESH_BINARY)[1]
    thresh = cv2.dilate(thresh, None, iterations=2)
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    all_rect=[]
    for c in cnts:
		# if the contour is too small, ignore it
        if cv2.contourArea(c) < 2500 and cv2.contourArea(c)>200:
            
            (x, y, w, h) = cv2.boundingRect(c)
            if y < 25 or y > 140: # do not take into considerations objects with y<25 or y>140
                continue
            crop=frame[y:y + h,x:x + w]
            x1=int((x+w)/2)
            y1=int((y+h)/2)
            if True:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                all_rect.append([x1,y1])
            text = "Occupied"
            
            #cv2.imshow('frame',crop) 
    #print(frame.shape) 
    old_frame=gray
    cv2.imshow('frame',frameDelta)
    cv2.imshow('frame',frame)

    print(all_rect)
    k = cv2.waitKey(300) & 0xff
    if k == 27:
        break
    


cv2.destroyAllWindows()

