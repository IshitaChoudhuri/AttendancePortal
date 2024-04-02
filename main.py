import cv2
import numpy as np
import face_recognition
import os
from datetime import datetime

# Path for storing training images
training_path = 'Training_images'

# List to keep track of marked attendance
marked_names = []

def markAttendance(name):
    try:
        file_path = 'Attendance.csv'
        with open(file_path, 'a+') as f:  # Open file in append mode
            if name not in marked_names:
                now = datetime.now()
                dtString = now.strftime('%H:%M:%S')
                f.write(f'\n{name},{dtString}')
                marked_names.append(name)  # Add the name to the marked list
    except Exception as e:
        print(f"Error: {e}")

def load_training_data():
    images = []
    classNames = []
    for root, dirs, files in os.walk(training_path):
        for name in dirs:
            person_folder = os.path.join(root, name)
            for file in os.listdir(person_folder):
                if file.endswith('.jpg') or file.endswith('.jpeg') or file.endswith('.png'):
                    img_path = os.path.join(person_folder, file)
                    cur_img = cv2.imread(img_path)
                    images.append(cur_img)
                    classNames.append(name)
    return images, classNames

def main():
    images, classNames = load_training_data()

    encodeListKnown = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face_encodings = face_recognition.face_encodings(img)
        if len(face_encodings) > 0:
            encodeListKnown.append(face_encodings[0])
    
    print('Encoding Complete')

    cap = cv2.VideoCapture(0)

    while True:
        success, img = cap.read()
        imgS = cv2.resize(img, (0, 0), None, 0.25, 0.25)
        imgS = cv2.cvtColor(imgS, cv2.COLOR_BGR2RGB)

        facesCurFrame = face_recognition.face_locations(imgS)
        encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)

        for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown, encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown, encodeFace)
            matchIndex = np.argmin(faceDis)

            if matches[matchIndex]:
                name = classNames[matchIndex].upper()
                y1, x2, y2, x1 = faceLoc
                y1, x2, y2, x1 = y1 * 4, x2 * 4, y2 * 4, x1 * 4
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                cv2.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv2.FILLED)
                cv2.putText(img, name, (x1 + 6, y2 - 6), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                markAttendance(name)

        cv2.imshow('Webcam', img)
        if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to close webcam
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
