# ------------------------------------RECOGNIZER FOR ALL TPES OF ALGORITHM----------------------------------------------
# ----------------------------------------------------------------------------------------------------------------------

import cv2                                                                                        # Importing the opencv
import NameFind as nf

fc = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')  # import the Haar cascades for face and eye ditection
ec = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')


LBPH = cv2.face.LBPHFaceRecognizer_create(2, 2, 7, 7, 20)                                      # FACE RECOGNISER OBJECT
EIGEN = cv2.face.EigenFaceRecognizer_create(10, 5000)
FISHER = cv2.face.FisherFaceRecognizer_create(5, 500)


LBPH.read("Recogniser/trainingDataLBPH.xml")            # Load the training data from the trainer to recognise the faces
EIGEN.read("Recogniser/trainingDataEigan.xml")
FISHER.read("Recogniser/trainingDataFisher.xml")

#                                                -------  PHOTO INPUT  -------

img = cv2.imread('Me4.jpg')                                                                  # THE ADDRESS TO THE PHOTO

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                                # Convert the Camera to gray
faces = fc.detectMultiScale(gray, 1.3, 4)                                     # Detect the faces and store the positions
print(faces)

for (x, y, w, h) in faces:                                                        # Frames  LOCATION X, Y  WIDTH, HEIGHT
    
    Face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))                            # The Face is isolated and cropped

    ID, conf = LBPH.predict(Face)                                                                     # LBPH RECOGNITION
    print(ID)
    NAME = nf.ID2Name(ID, conf)
    nf.DispID(x, y, w, h, NAME, gray)

    ID, conf = EIGEN.predict(Face)                                                              # EIGEN FACE RECOGNITION
    print(ID)
    NAME = nf.ID2Name(ID, conf)
    nf.DispID2(x, y, w, h, NAME, gray)

    ID, conf = FISHER.predict(Face)                                                            # FISHER FACE RECOGNITION
    print(ID)
    NAME = nf.ID2Name(ID, conf)
    nf.DispID2(x, y, w, h, NAME, gray)

cv2.imshow('Face Recognition System', gray)                                                         # IMAGE DISPLAY
cv2.waitKey(0)
cv2.destroyAllWindows()