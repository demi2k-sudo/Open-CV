import cv2

# img='car.jpg'#input

# cascade classifier
# Exact path of the xml files must be given
classifier = cv2.CascadeClassifier('E:\Code\Tesla\cars.xml')

people = cv2.CascadeClassifier('E:\Code\Tesla\ppl.xml')

#target video or webcam can also be turned on
video = cv2.VideoCapture('E:\Code\Tesla\Test2.mp4')


# cv2.imshow('Test',img)


while True:
    # Reading a frame
    successful_frame_read, frame = video.read()

    # Training from the xml
    vframe = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    coordinates = classifier.detectMultiScale(vframe)
    structure = people.detectMultiScale(vframe)

    # Rectangles
    for (x, y, w, h) in coordinates:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 256), 2)

    for (x, y, w, h) in structure:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (256, 256, 0), 2)

    # Output
    cv2.imshow('Autopilot', frame)

    key = cv2.waitKey(1)

    if key == 81 or key == 113:
        break

video.release()

print("Code completed")
