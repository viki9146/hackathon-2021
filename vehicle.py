from typing import Counter
import cv2
import numpy as np
from numpy.core.numeric import count_nonzero


#web camera
cap = cv2.VideoCapture("video_Slomo.mp4")
min_width_rect=80 #min width rectangle
min_height_rect=80 #min height rectangle

count_line_position = 600
count_line_position1 = 300

#initialize substractor
algo = cv2.createBackgroundSubtractorMOG2()


def center_handle(x,y,w,h):
    x1 = int(w/2)
    y1 = int(h/2)
    cx= x+x1
    cy= y+y1
    return cx,cy


detect = []
offset =6#allowable error between pixel
Counter=0


while True:
    ret, frame1 = cap.read()
    grey=cv2.cvtColor(frame1,cv2.COLOR_BGR2GRAY)
    blur =cv2.GaussianBlur(grey,(3,3),5)
    #applying on each frame
    img_sub = algo.apply(blur)
    delat=cv2.dilate(img_sub,np.ones((5,5)))
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
    dilatata = cv2.morphologyEx(delat,cv2.MORPH_CLOSE,kernel)
    dilatata = cv2.morphologyEx(dilatata,cv2.MORPH_CLOSE,kernel)
    Countersahpe,h =  cv2.findContours(dilatata, cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)

    cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(255,127,0),3)
    cv2.line(frame1,(15,count_line_position1),(500,count_line_position1),(255,127,0),3)

    for (i,c) in enumerate(Countersahpe):
        (x,y,w,h) = cv2.boundingRect(c)
        validate_counter = (w>= min_width_rect) and (w>= min_height_rect)
        if not validate_counter:
            continue

        cv2.rectangle(frame1,(x,y),(x+w,y+h),(0,255,0),2)
    
    
        center = center_handle(x,y,w,h)
        detect.append(center)
        cv2.circle(frame1,center,4,(0,0,255),-1)
    
        for (x,y) in detect:
            if y<(count_line_position1+offset) and y>(count_line_position1-offset):
                Counter-=1
            cv2.line(frame1,(25,count_line_position1),(1200,count_line_position1),(0,127,255),3)
            detect.remove((x,y))
            print("vehicle couunter:"+str(Counter))


        for (x,y) in detect:
            if y<(count_line_position+offset) and y>(count_line_position-offset):
                Counter+=1
            cv2.line(frame1,(25,count_line_position),(1200,count_line_position),(0,127,255),3)
            detect.remove((x,y))
            print("vehicle couunter:"+str(Counter))

    cv2.putText(frame1,"vehicle counter:"+str(Counter),(450,70),cv2.FONT_HERSHEY_SIMPLEX,2,(0,0,255),5)        

    
    
    cv2.imshow('Video Origional',frame1)
    if cv2.waitKey(1) == 13:
        break

cv2.destroyAllWindows()
cap.release()
