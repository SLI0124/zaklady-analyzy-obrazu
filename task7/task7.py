import glob
import time
import cv2 as cv
import dlib
import numpy as np


def get_crop(image, coords):
    start_x, start_y, width, height = coords
    end_x = start_x + width
    end_y = start_y + height

    one_place_image_res = image[start_y:end_y, start_x:end_x]

    return one_place_image_res, start_x, start_y


def calculate_EAR(points):
    points = map(lambda p: (p.x, p.y), points)

    def euclidean_distance(point1, point2):
        return np.linalg.norm(np.array(point1) - np.array(point2))

    p1, p2, p3, p4, p5, p6 = points

    numerator = euclidean_distance(p2, p6) + euclidean_distance(p3, p5)
    denominator = 2 * euclidean_distance(p1, p4)
    EAR = numerator / denominator
    return EAR


def custom_percents(value, maxValue):
    return value / maxValue


def mix_average_color(color1, color2, factor):
    factor = max(min(factor, 1.0), 0.0)

    b = round(color1[0] + (color2[0] - color1[0]) * factor)
    g = round(color1[1] + (color2[1] - color1[1]) * factor)
    r = round(color1[2] + (color2[2] - color1[2]) * factor)

    return b, g, r


def draw_eye(imageView, eye, eyeEAR):
    for part in eye:
        point = part.x, part.y
        cv.circle(imageView, point, 2, mix_average_color((0, 0, 255), (0, 255, 0), eyeEAR), cv.FILLED)


def main():
    cv.namedWindow("win", 0)

    images_names = [img for img in glob.glob("../input/task7/anomal_hd_30fps_02/*.jpg")]
    images_names.sort()

    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("../input/task7/dlib_shape_predictor_68_face_landmarks.dat")

    face_cascade = cv.CascadeClassifier("../input/task7/lbpcascade_frontalface_improved.xml")

    buffer_EAR = []
    max_buffer = 3
    threshold_for_open = 0.4

    last_faces = []
    number_of_not_found = 0

    for name in images_names:
        start_time = time.time()
        print(f"\n\033[1;38m{name}\033[0m")

        image = cv.imread(name)
        image_view = image.copy()

        faces = face_cascade.detectMultiScale(image, scaleFactor=1.2, minNeighbors=7, minSize=(50, 50),
                                              maxSize=(500, 500))

        if len(faces) == 0:
            number_of_not_found += 0

            if number_of_not_found > 3:
                faces = detector(image, 0)
                last_faces = map(lambda f: [f.left(), f.top(), f.right() - f.left(), f.bottom() - f.top()], faces)
                number_of_not_found = 0
        else:
            last_faces = faces

        for face in last_faces:
            face_x, face_y, face_w, face_h = face
            face = dlib.rectangle(face_x, face_y, face_x + face_w, face_y + face_h)
            cv.rectangle(image_view, (face_x, face_y), (face_x + face_w, face_y + face_h), (255, 255, 255), 4)

            shape = predictor(image, face)

            left_eye = shape.parts()[36:42]
            left_eye_EAR = custom_percents(calculate_EAR(left_eye), 0.5)
            right_eye = shape.parts()[42:48]
            right_eye_EAR = custom_percents(calculate_EAR(right_eye), 0.5)

            eyes_EAR = (left_eye_EAR + right_eye_EAR) / 2
            buffer_EAR.append(1 if eyes_EAR > threshold_for_open else 0)
            if len(buffer_EAR) > max_buffer:
                buffer_EAR = buffer_EAR[1:]

            eyes_open = sum(buffer_EAR) > len(buffer_EAR) / 2

            draw_eye(image_view, left_eye, eyes_open)
            draw_eye(image_view, right_eye, eyes_open)

            for i, part in enumerate(shape.parts()[:35]):
                point = part.x, part.y
                cv.circle(image_view, point, 2, (255, 200, 150), cv.FILLED)

            for i, part in enumerate(shape.parts()[43:]):
                point = part.x, part.y
                cv.circle(image_view, point, 2, (255, 200, 150), cv.FILLED)

            print(f"\033[1;36mEYES:\033[0m {buffer_EAR} ->"
                  f" {'\033[1;32mOpen' if eyes_open else '\033[1;31mClosed'}\033[0m")

            print(f"Current: {eyes_EAR}")

        delta_time = time.time() - start_time
        fps = 1 / delta_time
        print(f"\033[1;35mFPS:\033[0m {fps}")

        if cv.getWindowProperty("win", cv.WND_PROP_VISIBLE) < 1:
            break
        cv.imshow("win", image_view)
        if cv.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
