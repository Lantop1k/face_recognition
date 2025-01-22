import cv2
import os

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

imagename = 'test3.jfif'

img=cv2.imread(imagename)
img=cv2.resize(img, (520,520))
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 4)

folder='saved image'
facedatabase = os.listdir(folder)
faceimagearray =[cv2.imread(folder+'/'+file) for file in facedatabase]

if len(faces) >0:
    for face in faces:
        x, y, w, h = face

        startpoint=(x,y)
        endpoint=(x+w,y+h)
        color=(255,0,0)

        cropface = img[y-5:y+h+5,x-5:x+w+5]

        matchings=[]

        for arr in faceimagearray:
            result = cv2.matchTemplate(cv2.resize(cropface, (64,64)),
                                       cv2.resize(arr, (64,64)),
                                       cv2.TM_CCOEFF_NORMED)[0][0]
            matchings.append(result)

        if max(matchings)<0.5:
            cv2.putText(img, 'unknown', (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,
                        (0,255,0),2, cv2.LINE_AA)
        else:
            t=facedatabase[matchings.index(max(matchings))]
            cv2.putText(img, t.split('.')[0], (x,y),cv2.FONT_HERSHEY_SIMPLEX,1,
                        (0,255,0),2, cv2.LINE_AA)
            
        cv2.rectangle(img, startpoint, endpoint, color, 2)
    cv2.imshow('image',img)
    cv2.waitKey(0)

else:    

    pass
