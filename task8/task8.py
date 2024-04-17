import time

import cv2 as cv
import numpy as np
import dlib


def use_cascade(cascade, frame, min_neighbors=7):
    faces = cascade.detectMultiScale(frame, scaleFactor=1.2, minNeighbors=min_neighbors, minSize=(50, 50),
                                     maxSize=(500, 500))
    return len(faces) != 0, faces


def remove_duplicates(data, threshold):
    unique_data = []
    for row in data:
        is_duplicate = False
        for existing_row in unique_data:
            if all(abs(a - b) <= threshold for a, b in zip(row, existing_row)):
                is_duplicate = True
                break
        if not is_duplicate:
            unique_data.append(row)
    return unique_data


def get_crop(image, coords):
    start_x, start_y, width, height = coords
    end_x = start_x + width
    end_y = start_y + height

    one_place_image_res = image[start_y:end_y, start_x:end_x]

    return one_place_image_res, start_x, start_y


def get_faces(image, face_cascade, profile_cascade):
    faces = []
    face_detected, location = use_cascade(face_cascade, image)
    for item in location:
        faces.append(item)
    profile_detected, location = use_cascade(profile_cascade, image)
    for item in location:
        faces.append(item)

    return remove_duplicates(faces, 100)


def copy_to_image(image, copy_image, offset_x, offset_y, width, height):
    if offset_y + height <= image.shape[0] and offset_x + width <= image.shape[1]:
        image[offset_y:offset_y + height, offset_x:offset_x + width] = copy_image


def swap_method_1(paint_frame, frame, swap_cropped, face):
    cropped, offset_x, offset_y = get_crop(frame, face)

    height, width, _ = cropped.shape
    swap_cropped_resized = cv.resize(swap_cropped, (width, height))
    cv.imshow("swap", swap_cropped_resized)

    copy_to_image(paint_frame, swap_cropped_resized, offset_x, offset_y, width, height)


def swap_method_2(paint_frame, frame, swap_cropped, face):
    cropped, offset_x, offset_y = get_crop(frame, face)

    height, width, _ = cropped.shape
    swap_cropped_resized = cv.resize(swap_cropped, (width, height))
    cv.imshow("swap", swap_cropped_resized)

    white_image = np.ones_like(cropped) * 255

    seamless_clone = cv.seamlessClone(swap_cropped_resized, cropped, white_image, (width // 2, height // 2),
                                      cv.MONOCHROME_TRANSFER)
    cv.imshow("clone", seamless_clone)

    copy_to_image(paint_frame, seamless_clone, offset_x, offset_y, width, height)


def create_convex_mask(image, shape):
    points = np.array([(part.x, part.y) for part in shape.parts()], np.int32)
    hull = cv.convexHull(points)
    mask = np.zeros(image.shape[:2], dtype=np.uint8)
    cv.fillConvexPoly(mask, hull, 255)
    return mask


def swap_method_3(paint_frame, frame, swap_cropped, swap_mask_cropped, face, predictor):
    cropped, offset_x, offset_y = get_crop(frame, face)

    height, width, _ = cropped.shape
    swap_cropped_resized = cv.resize(swap_cropped, (width, height))
    swap_mash_cropped_resized = cv.resize(swap_mask_cropped, (width, height))
    cv.imshow("mask", swap_mash_cropped_resized)

    seamless_clone = cv.seamlessClone(swap_cropped_resized, cropped, swap_mash_cropped_resized,
                                      (width // 2, height // 2),
                                      cv.NORMAL_CLONE)
    cv.imshow("clone", seamless_clone)

    copy_to_image(paint_frame, seamless_clone, offset_x, offset_y, width, height)


def main():
    cv.namedWindow("win", 0)
    video = cv.VideoCapture("../input/task5/fusek_face_car_01.avi")
    face_cascade = cv.CascadeClassifier("../input/task5/haarcascades/haarcascade_frontalface_default.xml")
    profile_cascade = cv.CascadeClassifier("../input/task5/haarcascades/haarcascade_profileface.xml")

    swap_image = cv.imread("../input/task8/bruce.jpg")
    swap_face = get_faces(swap_image, face_cascade, profile_cascade)[0]
    # swap_face = [swap_face[0] + 50, swap_face[1] + 50, swap_face[2] - 100, swap_face[3] - 100]
    swap_face = [swap_face[0] + 20, swap_face[1] + 20, swap_face[2] - 70, swap_face[3] + 100]
    swap_cropped, swap_offset_x, swap_offset_y = get_crop(swap_image, swap_face)
    cv.imshow("swap", swap_cropped)

    predictor = dlib.shape_predictor("../input/task7/dlib_shape_predictor_68_face_landmarks.dat")
    face_x, face_y, face_w, face_h = swap_face
    swap_face_dlib = dlib.rectangle(face_x, face_y, face_x + face_w, face_y + face_h)
    swap_shape = predictor(swap_image, swap_face_dlib)
    swap_mask = create_convex_mask(swap_image, swap_shape)
    swap_mask_cropped, _, _ = get_crop(swap_mask, swap_face)

    while True:
        start_time = time.time()
        ret, frame = video.read()
        if frame is None:
            break
        paint_frame = frame.copy()
        if ret is True:
            faces = get_faces(frame, face_cascade, profile_cascade)
            for face in faces:
                face = [face[0] + 50, face[1] + 50, face[2] - 100, face[3]]

                # swapMethod1(paint_frame, frame, swap_cropped, face)
                # swapMethod2(paint_frame, frame, swap_cropped, face)
                swap_method_3(paint_frame, frame, swap_cropped, swap_mask_cropped, face, predictor)

        delta_time = time.time() - start_time

        fps = 1 / delta_time
        print(f"\033[1;35mFPS:\033[0m {fps}, \033[1;35mDelta time:\033[0m {delta_time}")

        if cv.getWindowProperty("win", cv.WND_PROP_VISIBLE) < 1:
            break
        cv.imshow("win", paint_frame)
        if cv.waitKey(1) == ord("q"):
            break


if __name__ == "__main__":
    main()
