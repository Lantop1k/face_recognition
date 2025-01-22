import cv2

haar_file = 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

imagename = input('Enter the name of the image : ')
img = cv2.imread(imagename)
personname =  input('Enter the person name:' )

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.3, 4)
if len(faces) ==1:
    for face in faces:
        x, y, w, h = face

        startpoint=(x,y)
        endpoint=(x+w,y+h)
        color=(255,0,0)

        cropface = img[y-5:y+h+5,x-5:x+w+5]

        cv2.imwrite('saved image/'+personname+'.png',cropface)
        
        cv2.rectangle(img, startpoint, endpoint, color, 2)
    cv2.imshow('image',img)
    cv2.waitKey(0)
else:
    pass
