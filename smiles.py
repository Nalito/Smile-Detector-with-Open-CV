import cv2

from random import randrange

face_detector = cv2.CascadeClassifier('haarcascade.xml')
smile_detector = cv2.CascadeClassifier("smiles.xml")

webcam = cv2.VideoCapture(0)

while True:
    successful_frame_read, frame = webcam.read()

    if not successful_frame_read:
        break

    gs_frm = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    faces = face_detector.detectMultiScale(gs_frm)
    smiles = smile_detector.detectMultiScale(gs_frm, scaleFactor=1.7, minNeighbors=20)

    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (randrange(125, 256), randrange(125, 256), randrange(125, 256)), 2)

        face = frame[y:y+h, x:x+w]

        gs_face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)

        smiles = smile_detector.detectMultiScale(face, scaleFactor=1.7, minNeighbors=20)

        #for (a, b, c, d) in smiles:
            #cv2.rectangle(face, (a, b), (a+c, b+d), (randrange(125, 256), randrange(125, 256), randrange(125, 256)), 2)
        if len(smiles)>0:
            cv2.putText(frame, 'smiling', (x, y+h+40), fontScale=3, fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 255))

    cv2.imshow("Smile Detector", frame)

    cv2.waitKey(1)



webcam.release()