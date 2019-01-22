import numpy as np
import cv2
import time

haar_face_cascade = cv2.CascadeClassifier('haarcascades/haarcascade_frontalface_default.xml')
if haar_face_cascade.empty():
    print("EMPTY")

def Angle(v1,v2):
 dot = np.dot(v1,v2)
 x_modulus = np.sqrt((v1*v1).sum())
 y_modulus = np.sqrt((v2*v2).sum())
 cos_angle = dot / x_modulus / y_modulus
 angle = np.degrees(np.arccos(cos_angle))
 return angle

def FindDistance(A,B):
 return np.sqrt(np.power((A[0][0]-B[0][0]),2) + np.power((A[0][1]-B[0][1]),2))

capa = cv2.VideoCapture(0)

p=0
lower_skin = np.ndarray(shape=(1,3))
upper_skin = np.ndarray(shape=(1,3))
px = np.ndarray(shape=(4,3))
pixel_min = np.ndarray(shape=(21,3))
pixel_max = np.ndarray(shape=(21,3))

times=0

print("Ready")

k=0
start_time = time.time()
while(True):
    ret2, frame2 = capa.read()
    ret, frame = capa.read()
    b = frame.shape[1]
    frame[0:160, 0:225] = [0,0,0]
    frame[0:160, b-225:b] = [0,0,0]
    frame = cv2.flip(frame,1)
    frame2 = cv2.flip(frame2,1)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    blur = cv2.GaussianBlur(frame,(11,11),0)
    blur = cv2.medianBlur(blur,5)
    ret,thresh_frame1 = cv2.threshold(gray,205,255,cv2.THRESH_BINARY)
    cv2.imshow('thresh_frame1', thresh_frame1)
    thresh_frame1_inv = cv2.bitwise_not(thresh_frame1)
    ret,thresh_frame2 = cv2.threshold(gray,220,255,cv2.THRESH_BINARY_INV)
    cv2.imshow('thresh_frame2', thresh_frame2)
    th = thresh_frame1 - thresh_frame2
    th = cv2.bitwise_not(th)
    cv2.imshow('th', th)

    faces = haar_face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=25)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame,(x,y),(x+w,y+h),(255,0,0),2)
        roi_color = frame[y+2:y+h-2, x+2:x+w-2]
        hsv_frame = cv2.cvtColor(roi_color,cv2.COLOR_BGR2HSV)
        w_f = hsv_frame.shape[0]
        h_f = hsv_frame.shape[1]
        cv2.circle(roi_color, ((w/2).astype(int),(h/2).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/5).astype(int),(h/2).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/4).astype(int),(h/3).astype(int)), 3, (0,255,0), -1)
        cv2.circle(roi_color, ((w/2).astype(int),(h/8).astype(int)), 3, (0,255,0), -1)
        if p <= 20:
            px[0] = hsv_frame[int(w/2), int(h/2)]
            px[1] = hsv_frame[int(w/5), int(h/2)]
            px[2] = hsv_frame[int(w/4), int(h/3)]
            px[3] = hsv_frame[int(w/2), int(h/8)]
            pixel_max[p] = np.amax(px, axis=0)
            pixel_min[p] = np.amin(px, axis=0)
            print("MAX",pixel_max[p])
            print("MIN",pixel_min[p])
        if p == 21:
            lower_skin = np.mean(pixel_min, axis=0)
            lower_skin[1] = lower_skin[1]-4
            lower_skin[2] = lower_skin[2]-4
            print(lower_skin)
            upper_skin = np.amax(pixel_max, axis=0)
            print(upper_skin)

        if p > 21:
            mask = cv2.inRange(hsv, lower_skin, upper_skin)
            cv2.imshow('premask', mask)
            mask = cv2.erode(mask,(5,5),iterations = 8)
            mask = cv2.dilate(mask,(5,5),iterations = 7)
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE,(5,5))
            mask = cv2.dilate(mask, kernel)
            mask[y-15:y+h+150, x-5:x+w+10] = 0
            cv2.imshow('mask', mask)
            mask2, contours, hierarchy = cv2.findContours(mask,cv2.RETR_TREE,cv2.CHAIN_APPROX_NONE)
            if len(contours)>0:
                j=0
                max = 0
                pos = 0
                for j in range(len(contours)):
                    cnt1 = contours[j]
                    area = cv2.contourArea(cnt1)
                    if area > max:
                        max = area
                        pos = j

                cnt = contours[pos]
                perimeter = cv2.arcLength(cnt,True)
                cv2.drawContours(frame, cnt, -1, (0,0,255), 3)
                hull = cv2.convexHull(cnt)
                hull2 = cv2.convexHull(cnt,returnPoints = False)
                defects = cv2.convexityDefects(cnt,hull2)
                FarDefect = []
                if defects is None:
                    break

                else:
                    for i in range(defects.shape[0]):
                        s,e,f,d = defects[i,0]
                        start = tuple(cnt[s][0])
                        end = tuple(cnt[e][0])
                        far = tuple(cnt[f][0])
                        FarDefect.append(far)
                        cv2.line(frame,start,end,[0,255,0],1)
                        cv2.circle(frame,far,7,[100,255,255],1)
                    moments = cv2.moments(cnt)
                    if moments['m00']!=0:
                        cx = int(moments['m10']/moments['m00']) # cx = M10/M00
                        cy = int(moments['m01']/moments['m00']) # cy = M01/M00
                    centerMass=(cx,cy)
                    cv2.circle(frame,centerMass,7,[100,0,255],2)
                    final = cv2.drawContours(frame, hull, -1, (255,0,0), 3)
                    x,y,w,h = cv2.boundingRect(cnt)
                    cv2.rectangle(frame,(x,y),(x+w+20,y+h+20),(0,255,255),2)
    cv2.imshow('frame', frame)


    p+=1
    #print(p)


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

capa.release()
cv2.destroyAllWindows()
