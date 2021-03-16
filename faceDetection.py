import cv2
import numpy as np

#variaveis de calibração
resolucao_w = 480 #referente ao comprimento horizontal do frame
resolucao_h = 252 #referente ao comprimento vertical do frame
min_vizinhos_frontal = 5 #quantidade minima de pontos proximos para fazer o reconhecimento da face frontal (recomendado entre 3 e 6)
min_vizinhos_perfil = 3 #quantidade minima de pontos proximos para fazer o reconhecimento da face lateral (recomendado entre 3 e 6)
tolerancia_frontal = 1.1 #aumenta ou diminui a sensibilidade de detecção da face frontal (de 1.01 a 1.99)
tolerancia_perfil = 1.15  #aumenta ou diminui a sensibilidade de detecção da face lateral  (de 1.01 a 1.99)

counter = 0
img = cv2.VideoCapture("/home/pi/Pictures/4.mp4")
#img = cv2.VideoCapture('/dev/video0')
face_cascade = cv2.CascadeClassifier("/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_frontalface_default.xml")
glasses_cascade = cv2.CascadeClassifier("/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")
profile_face_cascade = cv2.CascadeClassifier("/home/pi/opencv-3.1.0/data/haarcascades/haarcascade_profileface.xml")

while 1:
    #frame = cv2.imread("/home/pi/Pictures/download (2).jpeg")
    ret, frame = img.read()

    frame = cv2.rotate(frame, cv2.cv2.ROTATE_90_COUNTERCLOCKWISE)
    frame = cv2.resize(frame,(resolucao_h,resolucao_w))
    gray = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray,tolerancia_frontal,min_vizinhos_frontal)
    profile_face = profile_face_cascade.detectMultiScale(gray,tolerancia_perfil,min_vizinhos_perfil)
    print("tamanho do vetor: "+str(len(faces)))
    if(not(len(faces))):
        counter +=1
        print (counter)
    for (x,y,w,h) in faces:
        counter = 0
        face = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,0,255),2)
        glasses_search_gray = gray[y:y+h, x:x+w]
        glasses_search_color = face[y:y+h, x:x+w]
        glasses = glasses_cascade.detectMultiScale(glasses_search_gray)
        for(ex,ey,ew,eh) in glasses:
            cv2.rectangle(glasses_search_color, (ex,ey),(ex+ew, ey+eh), (0,255,0), 2)
    if counter > 72:
        cv2.putText(frame,"Alerta",(0,60),cv2.FONT_HERSHEY_SIMPLEX,1,(0,0,255),1)
    #for (x,y,w,h) in profile_face:
    #   pFace = cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,255),2)

    #cv2.imshow("dilation", dilation)
    #cv2.imshow("blurVideo", gray)
    cv2.imshow("frame", frame)


    if cv2.waitKey(90) & 0xFF == ord('q'):
        break

img.release()
cv2.destroyAllWindows()
