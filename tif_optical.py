import numpy as np
import cv2 as cv
import sys
import cv2
import os
import glob
from PIL import Image
import imutils
img_folder='/home/yogesh/Desktop/computer_vision_sub/data'
folders = list(filter(lambda x: os.path.isdir(os.path.join(img_folder, x)), os.listdir(img_folder)))
i=0
for img_dir in folders:
    i=i+1
    #img_dir="/home/yogesh/Desktop/computer_vision_sub/project/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Test/Test006"
    #img_dir="/home/yogesh/Desktop/computer_vision_sub/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001"
    img_dir='/home/yogesh/Desktop/computer_vision_sub/data/'+img_dir
    print(img_dir)
    data_path = os.path.join(img_dir,'*f')
    files = sorted(glob.glob(data_path))
    data = []
    crop=[]

    # params for ShiTomasi corner detection
    feature_params = dict( maxCorners = 100,
                        qualityLevel = 0.3,
                        minDistance = 7,
                        blockSize = 7 )
    # Parameters for lucas kanade optical flow
    lk_params = dict( winSize  = (15,15),
                    maxLevel = 2,
                    criteria = (cv.TERM_CRITERIA_EPS | cv.TERM_CRITERIA_COUNT, 10, 0.03))


    # Create some random colors
    color = np.random.randint(0,255,(100,3))
    old_gray=cv.imread("/home/yogesh/Desktop/computer_vision_sub/UCSD_Anomaly_Dataset.v1p2/UCSDped1/Train/Train001/001.tif")

    mask = np.zeros_like(old_gray)
    for f1 in files:
        img = Image.open(f1)
        img = np.array(img) 
        data.append(img)
    old_frame=None
    empty1=np.zeros_like(data[5])
    empty2=np.zeros_like(data[5])
    for frame in data:
        
        gray = frame#cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        #print(gray.shape)
        #gray = cv2.GaussianBlur(gray, (21, 21), 0)
        #fgmask = fgbg.apply(frame)
        if old_frame is None:
            old_frame = data[5]
            mask = np.zeros_like(old_frame)
            print("sydes")
            p0 = cv.goodFeaturesToTrack(old_frame, mask = None, **feature_params)
            old_gray=old_frame
            continue
        
        #old_gray=old_frame
        #ret,frame = cap.read()
        frame_gray = frame#cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
        # calculate optical flow
        p1, st, err = cv.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
        print(st)
        
        # Select good points
        good_new = p1[st==1]
        good_old = p0[st==1]
        # draw the tracks
        
        for i,(new,old) in enumerate(zip(good_new,good_old)):
            a,b = new.ravel()
            c,d = old.ravel()

            mask = cv.line(mask, (a,b),(c,d), color[i].tolist(), 1)
            frame = cv.circle(frame,(a,b),5,color[i].tolist(),-1)
            empty1 = cv.line(empty1, (a,b),(c,d), color[i].tolist(), 1)
            empty2 = cv.circle(empty2,(a,b),5,color[i].tolist(),-1)
        img = cv.add(frame,mask)
        empty=cv.add(empty2,empty1)
        cv.imshow('frame',img)
        k = cv.waitKey(30) & 0xff
        if k == 27:
            break
        # Now update the previous frame and previous points
        old_gray = frame_gray.copy()
        p0 = good_new.reshape(-1,1,2)
    cv.imwrite("metadata/meta_%i.jpg"%i,empty)
import gc
gc.collect()
cv.destroyAllWindows()
