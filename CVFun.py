#id people?
#make more accurate
#games?

import cv2
import numpy as np

face_cascade = cv2.CascadeClassifier(r"C:\Users\joshu\Desktop\projects\CV Render\haarcascade_frontalface_default.xml")

vid_type = input("Type? ")

if vid_type.lower() == "video":
    CAP = cv2.VideoCapture(r"C:\Users\joshu\Videos\security.mp4")
else:
    CAP = cv2.VideoCapture(0, cv2.CAP_DSHOW)

fourcc = cv2.VideoWriter_fourcc(*'XVID')  
out = cv2.VideoWriter(r"C:\Users\joshu\Videos\output.avi",fourcc, 10.0, (480*2,600))  

while True:
    f_ret, frame = CAP.read()

    if f_ret == True:
        #frame = cv2.flip(frame, -1z)
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5)
        people = [""]*len(faces)
        for pos, (x, y, w, h) in enumerate(faces):
            t_target = np.float32([[x, y], [x+w, y], [x, y+h], [x+w, y+h]])
            t_result = np.float32([[0, 0], [400, 0], [0, 600], [480, 600]])
            t_matrix = cv2.getPerspectiveTransform(t_target,t_result)
            t_result = cv2.warpPerspective(frame, t_matrix, (480, 600))
            
            people[pos] = t_result

            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            
        __ = []
        #config
        for col in range(0, 5):
            _ = []
            for row in range(0+(col*5), 5+(col*5)):
                try:
                    img = people[row]
                    img = cv2.resize(img, (100, 100), interpolation=cv2.INTER_AREA)
                except Exception as e:
                    placeholder = cv2.imread("C:/Users/joshu/Desktop/projects/detection/2/placeholder.png")
                    img = cv2.resize(placeholder, (100, 100), interpolation=cv2.INTER_AREA)
                finally:
                    _.append(img)
            
            __.append(_)

        h = [np.zeros((100, 100, 3), np.uint8)]*len(__)
        for x in range(0, len(__)):
            h[x] = np.hstack(__[x])
        v = np.vstack(h)
        v = cv2.resize(v, (480,600), interpolation=cv2.INTER_AREA)
        
        frame2 = cv2.resize(frame, (480,600))
        
        cam = np.concatenate((v, frame2), axis=1)
        
        out.write(cam)

        cv2.imshow("People", cam)

        if cv2.waitKey(1) & 0xFF == 27:
            break
    else:
        break

CAP.release()
out.release()
cv2.destroyAllWindows()