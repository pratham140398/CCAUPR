#   --------------------------------- RECOGNISER FOR THE EIGEN FACE RECOGNISER -----------------------------------------
# -------------------------------------- -------------------------------------------------------------------------------


import cv2                                                                                       # Importing the opencv
import NameFind

fc = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')  # import the Haar cascades for face and eye ditection
ec = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')

recognise = cv2.face.EigenFaceRecognizer_create(15, 4000)                               # creating EIGEN FACE RECOGNISER
recognise.read("Recogniser/trainingDataEigan.xml")                                              # Load the training data

# ------------------------------------------ START THE VIDEO FEED ------------------------------------------
cap = cv2.VideoCapture(0)                                                                                # Camera object
ID = 0
while True:
    ret, img = cap.read()                                                                       # Read the camera object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # Convert the Camera to gray
    faces = fc.detectMultiScale(gray, 1.3, 5)                                 # Detect the faces and store the positions
    for (x, y, w, h) in faces:                                                    # Frames  LOCATION X, Y  WIDTH, HEIGHT
        # ------------ BY CONFIRMING THE EYES ARE INSIDE THE FACE BETTER FACE RECOGNITION IS GAINED ------------------
        gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))                   # The Face is isolated and cropped
        eyes = ec.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            ID, conf = recognise.predict(gray_face)                                      # Determine the ID of the photo
            NAME = NameFind.ID2Name(ID, conf)
            NameFind.DispID(x, y, w, h, NAME, gray)
    cv2.imshow('EigenFace Face Recognition System', gray)                                               # Show the video
    if cv2.waitKey(1) & 0xFF == ord('q'):                                                         # Quit if the key is Q
        break
cap.release()
cv2.destroyAllWindows()
