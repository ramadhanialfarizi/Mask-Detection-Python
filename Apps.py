# import open cv
import cv2

#import cascade file
cascade_file = 'dataset/classifier/cascade.xml'
file_mask = cv2.CascadeClassifier(cascade_file)

#camera operation
cam = cv2.VideoCapture(0)
while True:
    _, frame = cam.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    face = file_mask.detectMultiScale(gray, 1.1,4)

    for x, y, w, h in face:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_color=frame[y:y+h, x:x+w]
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(frame, 'Using Mask',(55,280), font,0.5,(255,0,0))
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, 'Without Mask',(20,200), font,0.5,(255,255,255))

    cv2.imshow('MASK DETECTION', frame)

    if cv2.waitKey(1)& 0xff == ord('x'):
        break

cam.release()
cv2.destroyAllWindows()  