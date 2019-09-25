#   --------------------------------- RECOGNISER FOR THE FISHER FACE RECOGNISER ----------------------------------------
# ----------------------------------------------------------------------------------------------------------------------


import cv2                                                                                       # Importing the opencv
import NameFind                                                                                      # Import Functions

fc = cv2.CascadeClassifier('Haar/haarcascade_frontalcatface.xml')  # import the Haar cascades for face and eye ditection
ec = cv2.CascadeClassifier('Haar/haarcascade_eye.xml')
recognise = cv2.face.FisherFaceRecognizer_create(2, 40)                              # creating FISHER FACE RECOGNISER
recognise.read("Recogniser/trainingDataFisher.xml")     # Load the training data from the trainer to recognise the faces

#                    ------------------------- START THE VIDEO FEED ----------------------------
cap = cv2.VideoCapture(0)                                                                                # Camera object
ID = 0
while True:
    ret, img = cap.read()                                                                       # Read the camera object
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)                                            # Convert the Camera to gray
    faces = fc.detectMultiScale(gray, 1.3, 5)                                 # Detect the faces and store the positions
    for (x, y, w, h) in faces:                                                    # Frames  LOCATION X, Y  WIDTH, HEIGHT
        gray_face = cv2.resize((gray[y: y+h, x: x+w]), (110, 110))                    # The Face is isolated and cropped
    # -------- BY CONFIRMING THE EYES ARE INSIDE THE FACE BETTER FACE RECOGNITION IS GAINED ----------------
        eyes = ec.detectMultiScale(gray_face)
        for (ex, ey, ew, eh) in eyes:
            ID, conf = recognise.predict(gray_face)                                      # Determine the ID of the photo
            NAME = NameFind.ID2Name(ID ,conf)      
            NameFind.DispID(x, y, w, h, NAME, gray)
    cv2.imshow('FisherFace Face Recognition System', gray)                                              # Show the video
    if cv2.waitKey(1) & 0xFF == ord('q'):                                                         # Quit if the key is Q
        break
cap.release()
cv2.destroyAllWindows()