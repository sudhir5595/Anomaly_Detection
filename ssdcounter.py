
from get_id import  unique_ids
import argparse
import time
import cv2
import get_id
import imutils
import numpy as np

line_point1 = (200,0)
line_point2 = (200,500)


CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
	"bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
	"dog", "horse", "motorbike", "person", "pottedplant", "sheep",
	"sofa", "train", "tvmonitor"]

#in this case above the line and inbetween the two points is considered in

ENTERED_STRING = "ENTERED_THE_AREA" 
LEFT_AREA_STRING = "LEFT_THE_AREA"
NO_CHANGE_STRING = "NOTHIN_HOMEBOY"

LOWEST_CLOSEST_DISTANCE_THRESHOLD = 100
#"/home/yogesh/Desktop/Project_DNA/people-counting-opencv/hghh.3gp"
video_path="videos/video1.mp4"
ct= unique_ids()

def get_footage():
   
    #camera = cv2.VideoCapture("/home/yogesh/Desktop/Project_DNA/people-counting-opencv/hghh.3gp")
    camera = cv2.VideoCapture(video_path)

    time.sleep(0.25)
    return camera

   

net = cv2.dnn.readNetFromCaffe("models/MobileNetSSD_deploy.prototxt","models/MobileNetSSD_deploy.caffemodel")

camera = get_footage()
fgbg = cv2.createBackgroundSubtractorMOG2()
frame_count = 0
people_list = []
inside_count = 0
out_count=0
m=0
id=0
while True:

    (grabbed, frame) = camera.read()
    
    if not grabbed:
        break
  
    centroids=[]
    frame = imutils.resize(frame, width=500)
    (H, W) = frame.shape[:2]
    frame_count += 1
    
    #print(frame_count)
    if True:
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
                    
                    #print(frame_count)
                    centroids.append(box)

        ids =ct.unique(centroids,frame_count)
        print(all_rect)
        m=1
    
   
    
    
    cv2.imshow("Security Feed", frame)
    #cv2.imshow("Foreground Model", fgmask)


    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        
    
        break

camera.release()
cv2.destroyAllWindows()

'''
#print(ids)
        cap = cv2.VideoCapture(video_path)
        for id in ids:
            #print(id[1][0])
            
            cap.set(1,id[0]-1)
            ret, frame = cap.read()
            frame = imutils.resize(frame, width=500)

            path="gallary/"
            #print(int(id[1][1]),int(id[1][3]),int(id[1][0]),int(id[1][2]))
            #cv2.rectangle(frame, (int(id[1][0]), int(id[1][1])), (int(id[1][2]), int(id[1][3])), (0, 255, 0), 2)
            #frame = cv2.circle(frame, (int(id[1][0]), int(id[1][1])), 20, (255,0,0), 3)
            frame=frame[int(id[1][1]):int(id[1][3]),int(id[1][0]):int(id[1][2])]
            cv2.imwrite(path+str(id)+'.jpg',frame)
'''

