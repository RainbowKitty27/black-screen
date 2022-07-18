import cv2
import time
import numpy as np
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_file = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))
cap = cv2.VideoCapture(0)
time.sleep(2)
frame = 0
for i in range(60):
    ret, frame = cap.read()
frame = np.flip(frame, axis=1)

frame=cv2.resize(frame, (640,480))
while (cap.isOpened()):
    ret, img = cap.read()
    if not ret:
        break
    img = np.flip(img, axis=1)
    img=cv2.resize(img,(640,480))
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    l_black = np.array([104, 153, 70])
    u_black = np.array([30, 30,0])
    mask = cv2.inRange(frame, l_black, u_black)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    mask = cv2.morphologyEx(mask, cv2.MORPH_DILATE, np.ones((3, 3), np.uint8))
    mask = cv2.bitwise_not(mask)
    res_2 = cv2.bitwise_and(img, img, mask=mask)
    res_1 = cv2.bitwise_and(frame, frame, mask=mask)
    final_output = cv2.addWeighted(res_1,1,res_2,1, 0)
    output_file.write(final_output)
    cv2.imshow("magic", final_output)
    cv2.waitKey(1)
cap.release()
out.release()
cv2.destroyAllWindows()