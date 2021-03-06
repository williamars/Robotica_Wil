import cv2
import numpy as np
from matplotlib import pyplot as plt
import time
import auxiliar as aux

cap = cv2.VideoCapture('hall_box_battery_1024.mp4')

pink = '#df15a8'
blue = '#2216b6'

while(True):
    # Capture frame-by-frame
    ret, frame = cap.read()
    
    if ret == False:
        print("Codigo de retorno FALSO - problema para capturar o frame")

    # Our operations on the frame come here
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) 
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    #1. Transformar em HSV 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
   
    #2. aux
    cap_pink1, cap_pink2 = aux.ranges(pink)
    mask_pink = cv2.inRange(hsv, cap_pink1, cap_pink2)
   
    # Blue
    cap_blue1, cap_blue2 = aux.ranges(blue)
    mask_blue = cv2.inRange(hsv, cap_blue1, cap_blue2)
    
    mask = mask_blue + mask_pink
    
    cv2.imshow("mask", mask)

    # Display the resulting frame
    # cv2.imshow('frame',frame)
    cv2.imshow('gray', gray)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()

