import os
import time

import cv2 as cv
import numpy as np


def detect_faces(face_cascade, frame, scaleFactor, minNeighbors, minSize):
    return face_cascade.detectMultiScale(frame, scaleFactor=scaleFactor, minNeighbors=minNeighbors, minSize=minSize,
                                         maxSize=(500, 500))


def remove_duplicates(locations, threshold_distance):
    non_duplicates = []
    for i in range(len(locations)):
        # Check if the distance between the centers of the rectangles is greater than the threshold distance
        if not any(((locations[i][0] + locations[i][2] / 2) - (locations[j][0] + locations[j][2] / 2)) ** 2 + (
                (locations[i][1] + locations[i][3] / 2) - (
                locations[j][1] + locations[j][3] / 2)) ** 2 < threshold_distance ** 2 for j in
                   range(i + 1, len(locations))):
            non_duplicates.append(locations[i])
    return non_duplicates


def draw_rectangle(frame, locations, color1, color2):
    for (x, y, w, h) in locations:
        cv.rectangle(frame, (x, y), (x + w, y + h), color1, 8)
        cv.rectangle(frame, (x, y), (x + w, y + h), color2, 2)


def eye_open(image):
    gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    cv.GaussianBlur(gray, (5, 5), 2)
    return cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 1, param1=50, param2=20, minRadius=10, maxRadius=40) is not None


def face_detect():
    video_cap = cv.VideoCapture("../input/task5/fusek_face_car_01.avi")
    face_cascade = cv.CascadeClassifier("../input/task5/haarcascades/haarcascade_frontalface_default.xml")
    face_cascade_profile = cv.CascadeClassifier("../input/task5/haarcascades/haarcascade_profileface.xml")
    eye_cascade = cv.CascadeClassifier("../input/task5/eye_cascade_fusek.xml")
    mouth_cascade = cv.CascadeClassifier("../input/task5/haarcascades/haarcascade_smile.xml")

    if not os.path.exists("../output/task5/open"):
        os.makedirs("../output/task5/open")
    if not os.path.exists("../output/task5/closed"):
        os.makedirs("../output/task5/closed")

    while True:
        ret, frame = video_cap.read()
        if frame is None:
            break
        paint_frame = frame.copy()

        locations_face_front = detect_faces(face_cascade, paint_frame, 1.2, 7, (100, 100))
        locations_face_profile = detect_faces(face_cascade_profile, frame, 1.2, 7, (100, 100))
        locations_face = locations_face_front if len(locations_face_front) else locations_face_profile

        # If both front and profile faces are detected, concatenate the two lists
        if len(locations_face_front) and len(locations_face_profile):
            locations_face = np.concatenate((locations_face_front, locations_face_profile), axis=0)
        locations_face = remove_duplicates(locations_face, 100.0)

        for one_face in locations_face:
            draw_rectangle(paint_frame, [one_face], (0, 0, 255), (203, 192, 255))

            for i, (x, y, w, h) in enumerate(locations_face):
                face_roi = frame[y:y + h, x:x + w]
                eyes = detect_faces(eye_cascade, face_roi, 1.3, 13, (30, 30))
                for eye in eyes:
                    color = (0, 255, 0) if eye_open(face_roi[eye[1]:eye[1] + eye[3], eye[0]:eye[0] + eye[2]]) else (
                        0, 0, 255)
                    draw_rectangle(paint_frame[y:y + h, x:x + w], [eye], color, (203, 192, 255))

                    # Save the frame to the corresponding folder
                    timestamp = int(time.time())
                    if is_eye_open:
                        cv.imwrite(f"../output/task5/open/{timestamp}.jpg", paint_frame)
                    else:
                        cv.imwrite(f"../output/task5/closed/{timestamp}.jpg", paint_frame)

                mouth = detect_faces(mouth_cascade, face_roi, 1.2, 50, (40, 40))
                for m in mouth:
                    draw_rectangle(paint_frame[y:y + h, x:x + w], [m], (255, 0, 0), (203, 192, 255))

            cv.imshow("face_detect", paint_frame)
            if cv.waitKey(2) == ord('q'):
                break


if __name__ == "__main__":
    face_detect()
    cv.destroyAllWindows()
