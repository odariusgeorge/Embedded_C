# START PROGRAM



# importarea de pachete necesare
from imutils.video import VideoStream
from imutils import face_utils
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2

def distanta_euclidiana(punctulA, punctulB):
    #calcularea si returnarea distantei dintre doua
    #puncte
    return np.linalg.norm(punctulA - punctulB)

def caracteristici_ochi(ochi):
    #calcularea distantelor dintre doua seturi
    #de ochi (x,y)
    A = distanta_euclidiana(ochi[1], ochi[5])
    B = distanta_euclidiana(ochi[2], ochi[4])
    
    
    # calcularea distantei orizontale 
    C = distanta_euclidiana(ochi[0], ochi[3])

    # calcularea perimetrului ochilor
    dist = (A + B) / (2.0 * C)

    # returnare caracteristici distanta ochi
    return dist
 
# construirea argumentelor si parsarea lor
# folosind opencv
# si functiile implementate
# de aceasta librarie
ap = argparse.ArgumentParser()
ap.add_argument("-c", "--cascade", required=True,
    help = "path to where the face cascade resides")
ap.add_argument("-p", "--shape-predictor", required=True,
    help="path to facial landmark predictor")
ap.add_argument("-a", "--alarm", type=int, default=0,
    help="boolean used to indicate if TraffHat should be used")
args = vars(ap.parse_args())

# verificare daca avem sistem de sunet
# integrat 
if args["alarm"] > 0:
    from gpiozero import TrafficHat
    th = TrafficHat()
    print("[INFO] utilizare sunet alarma...")

# definirea a doua constante
# pentru a indica starea ochilor
# de exemplu calcularea timpului mediu a unei clipiri
# pentru a fi ignorata de camera
# a doua constanta fiind folosita pentru a
# verifica timpul necesar pentru a declansa alarma

EYE_AR_THRESH = 0.35
EYE_AR_CONSEC_FRAMES = 6

# initializarea contorului de frameuri
# un boolean pentru a indica ca alarma este oprita

COUNTER = 0
ALARM_ON = False

# incarcarea librariei OpenCV responsabila pentru detectarea fetei
# este mai rapida decat libraria dLib responsabila cu aceasta
# dar nu este la fel de precisa
# apoi generam punctele importante ale fetei

print("[INFO] incarcare trasaturi ochi...")
detector = cv2.CascadeClassifier(args["cascade"])
predictor = dlib.shape_predictor(args["shape_predictor"])

# preluam indecsii pentru ochiul drept respectiv ochiul stang

(lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
(rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

# pornirea firului de executie
# responsabil cu video streamingul
print("[INFO] lansarea firului de executie pentru video streaming...")
#vs = VideoStream(src=0).start()
vs = VideoStream(usePiCamera=True).start()
time.sleep(1.0)

# ciclu pentru frameurile din video streaming

while True:
    # aducerea cadrelor din firul de executie
    # al video streamului
    # redimensionarea acestuia si convertirea lui in grayscale channels

    cadru = vs.read()
    cadru = imutils.resize(cadru, width=450)
    gray = cv2.cvtColor(cadru, cv2.COLOR_BGR2GRAY)
    
    # detectarea fetelor in format alb-negru
    
    rects = detector.detectMultiScale(gray, scaleFactor=1.1, 
        minNeighbors=5, minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE)
    
    # ciclu al fetelor detectate
   
    for (x, y, w, h) in rects:
        # construirea obiectelor
        
        rect = dlib.rectangle(int(x), int(y), int(x + w),
            int(y + h))
        
        # determinarea reperelor faciale
        # convertirea acestor repere faciale in coordonate (x,y)
        # intr-un array NumPy
        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)
        
        # extragea coordonatelor ochilor
        # folosirea acestor coordonate
        # pentru a calcula caracteristicle fiecarui ochi

        leftEye = shape[lStart:lEnd]
        rightEye = shape[rStart:rEnd]
        leftEAR = caracteristici_ochi(leftEye)
        rightEAR = caracteristici_ochi(rightEye)

        # media 
        ear = (leftEAR + rightEAR) / 2.0

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(cadru, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(cadru, [rightEyeHull], -1, (0, 255, 0), 1)
        
        # verificarea daca aspectului ochilor este mai mica decat
        # limita dintre clipire
        
        if ear < EYE_AR_THRESH:
            COUNTER += 1
            
            # daca ochii au fost inchisi pentru un suficient nr de frameuri
            # declansare alarma

            if COUNTER >= EYE_AR_CONSEC_FRAMES:
                # daca alarma nu este pornita, se porneste
                if not ALARM_ON:
                    ALARM_ON = True
                    # se verifica daca buzzerul TrafficHat ar trebui
                    # sa actioneze
                    
                    if args["alarm"] > 0:
                        th.buzzer.blink(0.1, 0.1, 10,
                            background=True)

                # 'deseneaza' o alarma pe frame-urile in care
                # se detecteaza adormirea
                cv2.putText(cadru, "ALERTA ADORMIRE!!!", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        # altfel, daca aspect ratio-ul nu este sub threshold
        # se reseteaza contorul si alarma

        else:
            COUNTER = 0
            ALARM_ON = False
        
        # trasarea liniilor corespunzatoare trasaturilor ochilor
        # in timp real
        cv2.putText(cadru, "EAR: {:.3f}".format(ear), (300, 30),
            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
 
    # arata imagine
    cv2.imshow("Frame", cadru)
    cheie = cv2.waitKey(1) & 0xFF
 
    # daca este apasata tasta "q" se va iesi din video stream
    if cheie == ord("q"):
        break

# eliberare memorie
cv2.destroyAllWindows()
vs.stop()