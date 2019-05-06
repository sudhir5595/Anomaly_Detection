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
net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt","models/MobileNetSSD_deploy.caffemodel")
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

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
    frame = imutils.resize(frame, width=500)

    (H, W) = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (W, H), 127.5)
    net.setInput(blob)
    detections = net.forward()
    all_rect=[]
    for i in np.arange(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                if CLASSES[idx] != "person":
                    continue
                box = detections[0, 0, i, 3:7] * np.array([W, H, W, H])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                #rect = dlib.rectangle(startX, startY, endX, endY)
                rectangle_center = ((int((startX+ endX)/2)), (int((startY+endY)/2)))
                all_rect.append(rectangle_center)
                
               

    
    print(all_rect)
    m=1
cv2.imshow("Security Feed", frame)
#cv2.imshow("Foreground Model", fgmask)


key = cv2.waitKey(1) & 0xFF


cv2.destroyAllWindows()



