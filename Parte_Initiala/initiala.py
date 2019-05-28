# importarea de pachete initiale
from imutils import face_utils
import dlib
import cv2
 
# initializarea librariei dlib si a functiilor necesare pentru detectarea
# trasaturilor fetei

p = "forma_fata.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(p)

# preluarea imaginii de input si convertirea acesteia
# in alb-negru
imagine_initiala = cv2.imread("examplu.jpg")
imagine = cv2.imread("examplu.jpg")
alb_negru = cv2.cvtColor(imagine, cv2.COLOR_BGR2GRAY)
 
# detectarea imaginii in tonuri alb-negru
transformare = detector(alb_negru, 0)
 
# parsarea imaginii 
for (i, j) in enumerate(transformare):
        # determinarea reperelor faciale
        # convertirea acestor repere faciale in coordonate (x,y)
        # intr-un array NumPy

	forma = predictor(alb_negru, j)
	forma = face_utils.shape_to_np(forma)
         
        # parcurgerea coordonatelor (x,y) pentru reperele faciale
        # desenarea acestora pe imagine

	for (x, y) in forma:
		cv2.circle(imagine, (x, y), 2, (0, 255, 0), -1)

# afisarea imaginii cu fata detectata si trasaturile importante ale fetei
# cv2.circle este o functie de desenare aceasta ia ca parametri
# o imagine si centrul cercului, raza cercului, grosimea, tipul de linie

cv2.imshow("Forma Initiala", imagine_initiala)
cv2.imshow("Forma Finala", imagine)
cv2.waitKey(0)
