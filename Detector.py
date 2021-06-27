import cv2
import pkg_resources
import numpy
import matplotlib

# Using the cascade that I have downloaded
trained_face_data = cv2.CascadeClassifier(
    'E:\Code\Smile\haarcascade_frontalface_default.xml')
smile = cv2.CascadeClassifier('E:\Code\Smile\haarcascade_smile.xml')
eyed = cv2.CascadeClassifier('E:\Code\Smile\haarcascade_eye.xml')
# Using a Webcam or linking the video file

webcam = cv2.VideoCapture(0)

# iteration
while True:

    # Reading a frame
    successful_frame_read, frame = webcam.read()

    # Grayscale is the only filter my algorithm knows
    grayscaled_img = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # COORDINATES
    coordinates = trained_face_data.detectMultiScale(grayscaled_img)
    smile_coordinates = smile.detectMultiScale(
        grayscaled_img, scaleFactor=1.7, minNeighbors=20)
    eye = eyed.detectMultiScale(
        grayscaled_img, scaleFactor=1.1, minNeighbors=10)

    # rectangles
    for (x, y, h, w) in eye:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 250), 2)

    for (x, y, h, w) in coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (256, 0, 0), 2)

        # To reduce unwanted boxes we gonna search for smiles only inside the face coordinates
        the_face = frame[y:y+h, x:x+w]

        face_gray = cv2.cvtColor(the_face, cv2.COLOR_BGR2GRAY)

        smile_coordinates = smile.detectMultiScale(
            face_gray, scaleFactor=1.7, minNeighbors=20)
        for (a, b, c, d) in smile_coordinates:

            cv2.rectangle(the_face, (a, b), (a+c, b+d), (50, 75, 0), 2)
        if len(smile_coordinates) > 0:
            cv2.putText(frame, 'Smiling', (x, y+h+40), fontScale=2,
                        fontFace=cv2.FONT_HERSHEY_PLAIN, color=(255, 255, 0))

    cv2.imshow('Spycam', frame)
    key = cv2.waitKey(1)
    if key == 81 or key == 113:
        break

webcam.release()
cv2.destroyAllWindows()
