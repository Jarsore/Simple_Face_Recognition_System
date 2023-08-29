import cv2
import light

recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer.load('//home/pi/CLBDEMO/a/face_trainer/trainer.yml')
cascadePath = "//home/pi/CLBDEMO/a/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(cascadePath)
font = cv2.FONT_HERSHEY_SIMPLEX


names = ['xh','yy','mm']

cam = cv2.VideoCapture(0)
minW = 0.1*cam.get(3)
minH = 0.1*cam.get(4)

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = faceCascade.detectMultiScale(
        gray,
        scaleFactor=1.2,
        minNeighbors=5,
        minSize=(int(minW), int(minH))
    )

    for (x, y, w, h) in faces:

        cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
        idnum, confidence = recognizer.predict(gray[y:y+h, x:x+w])
        print(idnum)
        if confidence < 65:
            idnum = 0
            idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
            light.makerobo_set_Color(col=0x00FF)
        elif  65 < confidence < 117:
            idnum = 1
            idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))
            light.makerobo_set_Color(col=0xFF00)
        elif 117 < confidence < 133:
            idnum = 2
            idnum = names[idnum]
            confidence = "{0}%".format(round(100 - confidence))


        else:
            idnum = "unknown"
            confidence = "{0}%".format(round(100 - confidence))

        cv2.putText(img, str(idnum), (x+5, y-5), font, 1, (0, 0, 255), 1)
        cv2.putText(img, str(confidence), (x+5, y+h-5), font, 1, (0, 0, 0), 1)

    cv2.imshow('camera', img)
    k = cv2.waitKey(10)
    if k == 27:
        break

cam.release()
cv2.destroyAllWindows()
