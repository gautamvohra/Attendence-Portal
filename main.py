import face_recognition
import cv2
import numpy as np
import csv
from datetime import datetime

video_capture = cv2.VideoCapture(0)

messis_image = face_recognition.load_image_file("faces/Messi.jpeg")
messis_encoding = face_recognition.face_encodings(messis_image)[0]

ronaldos_image = face_recognition.load_image_file("faces/Ronaldo.jpeg")
ronaldos_encoding = face_recognition.face_encodings(ronaldos_image)[0]

creator_image = face_recognition.load_image_file("faces/creator.jpg")
creator_encoding = face_recognition.face_encodings(creator_image)[0]

Arpan_image = face_recognition.load_image_file("faces/Arpan.jpg")
Arpan_encoding = face_recognition.face_encodings(Arpan_image)[0]

bhavik_image = face_recognition.load_image_file("faces/bhavik.jpg")
bhavik_encoding = face_recognition.face_encodings(bhavik_image)[0]

Yash_image = face_recognition.load_image_file("faces/Yash.jpg")
Yash_encoding = face_recognition.face_encodings(Yash_image)[0]

mayank_image = face_recognition.load_image_file("faces/mayank.jpg")
mayank_encoding = face_recognition.face_encodings(mayank_image)[0]

known_face_encodings = [messis_encoding, ronaldos_encoding,creator_encoding,Arpan_encoding,bhavik_encoding,Yash_encoding,mayank_encoding]
known_face_names = ["Messi", "Ronaldo","Creator","Arpan","Bhavik","Yash","Mayank"]

students = known_face_names.copy()

face_locations = []
face_encodings = []

now = datetime.now()
current_date = now.strftime("%Y-%m-%d")

f = open(f"{current_date}.csv", "w+", newline="")
lnwriter = csv.writer(f)

name = ""

while True:

    _, frame = video_capture.read()
    small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
    rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

    face_locations = face_recognition.face_locations(rgb_small_frame)
    face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

    for face_encoding in face_encodings:
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
        face_distance = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distance)

        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        if name in known_face_names:
            font = cv2.FONT_HERSHEY_SIMPLEX
            bottomLeftCornerOfText = (10, 100)
            fontScale = 1.5
            fontColor = (255, 0, 0)
            thickness = 3
            lineType = 2
            cv2.putText(frame, name + " Present", bottomLeftCornerOfText, font, fontScale, fontColor, thickness,
                        lineType)

        if name in students:
            students.remove(name)
            current_time = now.strftime("%H-%M%S")
            lnwriter.writerow([name, current_time])

    cv2.imshow("Attendance", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

video_capture.release()
cv2.destroyAllWindows()
