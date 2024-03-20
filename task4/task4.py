#!/usr/bin/python

import sys
import cv2 as cv
import numpy as np
import glob


def order_points(pts):
    # initialize a list of coordinates that will be ordered
    # such that the first entry in the list is the top-left,
    # the second entry is the top-right, the third is the
    # bottom-right, and the fourth is the bottom-left
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # now, compute the difference between the points, the
    # top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    # return the ordered coordinates
    return rect


def four_point_transform(image, one_c):
    # https://www.pyimagesearch.com/2014/08/25/4-point-opencv-getperspective-transform-example/

    pts = [((float(one_c[0])), float(one_c[1])),
           ((float(one_c[2])), float(one_c[3])),
           ((float(one_c[4])), float(one_c[5])),
           ((float(one_c[6])), float(one_c[7]))]

    # obtain a consistent order of the points and unpack them
    # individually
    rect = order_points(np.array(pts))
    (tl, tr, br, bl) = rect
    # compute the width of the new image, which will be the
    # maximum distance between bottom-right and bottom-left
    # x-coordiates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    max_width = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    height_a = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    height_b = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    max_height = max(int(height_a), int(height_b))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [max_width - 1, 0],
        [max_width - 1, max_height - 1],
        [0, max_height - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    matrix = cv.getPerspectiveTransform(rect, dst)
    warped = cv.warpPerspective(image, matrix, (max_width, max_height))
    # return the warped image
    return warped


def is_parking_empty(image, template, threshold: float) -> tuple[bool, float]:
    image = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
    template = cv.cvtColor(template, cv.COLOR_BGR2GRAY)

    tmp_out = cv.matchTemplate(image, template, cv.TM_SQDIFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(tmp_out)

    match_percents = (1 - min_val) * 100 - 0.1
    return match_percents > threshold, match_percents


def get_results_for_image(filename: str) -> list:
    try:
        with open(filename, "r") as f:
            data = f.read().strip().split("\n")
            return list(map(lambda x: int(x), data))
    except FileNotFoundError:
        return []


def draw_view(image_view, coords: list, index_of_parking_place, white_count, color):
    cv.line(image_view, (int(coords[0]), int(coords[1])), (int(coords[2]), int(coords[3])), color, 2)
    cv.line(image_view, (int(coords[2]), int(coords[3])), (int(coords[4]), int(coords[5])), color, 2)
    cv.line(image_view, (int(coords[4]), int(coords[5])), (int(coords[6]), int(coords[7])), color, 2)
    cv.line(image_view, (int(coords[0]), int(coords[1])), (int(coords[6]), int(coords[7])), color, 2)
    center_x = int((int(coords[0]) + int(coords[4])) / 2)
    center_y = int((int(coords[1]) + int(coords[5])) / 2)
    cv.putText(image_view, str(index_of_parking_place), (center_x, center_y), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2,
               cv.LINE_AA)
    cv.putText(image_view, str(white_count), (center_x, center_y + 25), cv.FONT_HERSHEY_SIMPLEX, .5, (75, 0, 0), 2,
               cv.LINE_AA)


def f1_score(true, predicted):
    true_positive = false_positive = false_negative = 0
    for p, t in zip(predicted, true):
        if p == t:
            true_positive += 1
        elif p == 1 and t == 0:
            false_positive += 1
        else:
            false_negative += 1

    precision = true_positive / (true_positive + false_positive)
    recall = true_positive / (true_positive + false_negative)

    if precision + recall == 0:
        return 0
    return 2 * (precision * recall) / (precision + recall)


def get_crop(image, coords):
    one_place_image = four_point_transform(image, coords).copy()
    one_place_image_res = cv.resize(one_place_image, (80, 80))

    return one_place_image_res


def match_template(image, coords, template, template_sun):
    one_place_image_res = get_crop(image, coords)

    is_empty_basic, percents_basic = is_parking_empty(one_place_image_res, template, 91.2)
    is_empty_sun, percents_sun = is_parking_empty(one_place_image_res, template_sun, 95.5)

    return is_empty_basic or is_empty_sun or max(percents_basic, percents_sun) == -0.1, max(percents_basic,
                                                                                            percents_sun)


def canny(image, coords):
    one_place_image_res = get_crop(image, coords)

    white = cv.countNonZero(one_place_image_res)
    return white <= 1200, white


def main(argv):
    pkm_file = open('../input/task4/map/parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coordinates = []

    for line in pkm_lines:
        st_line = line.strip()
        sp_line = list(st_line.split(" "))
        pkm_coordinates.append(sp_line)

    template = cv.imread("../input/task4/templates/template.jpg", 1)
    template_sun = cv.imread("../input/task4/templates/template_sun.jpg", 1)

    test_images = [img for img in glob.glob("../input/task4/*.jpg")]
    test_images.sort()
    test_images_results = [img for img in glob.glob("../input/task4/*.txt")]
    test_images_results.sort()

    cv.namedWindow("image", cv.WINDOW_NORMAL)
    cv.namedWindow("imageCanny", cv.WINDOW_NORMAL)

    all_f1_scores = []
    i = 0
    while i < len(test_images):
        print(f"Image: {test_images[i]}")
        image = cv.imread(test_images[i], 1)
        image_view = image.copy()

        image_canny = cv.Canny(image, 100, 400)
        image_canny = cv.adaptiveThreshold(image_canny, 255, cv.ADAPTIVE_THRESH_MEAN_C, cv.THRESH_BINARY_INV, 5, 2)

        detected = []
        count_of_empty = 0
        index_of_parking_place = 0
        for coords in pkm_coordinates:
            index_of_parking_place += 1

            is_empty_match, percent = match_template(image, coords, template, template_sun)
            is_empty_canny, white_count = canny(image_canny, coords)

            is_really_full_canny = white_count > 3500
            is_really_full_match = percent <= 70 and white_count > 400

            is_empty = (is_empty_match or is_empty_canny) and not is_really_full_canny and not is_really_full_match
            detected.append(int(not is_empty))
            if is_empty:
                count_of_empty += 1

            draw_view(image_view, list(map(lambda x: int(x), coords)), index_of_parking_place, white_count,
                      (0, 255, 0) if is_empty else (0, 0, 255))

            print(f"[{index_of_parking_place}]: {'Empty' if is_empty else 'Occupied'}", end='\t\t')
            print(f"Match:{percent:.2f}\t Canny:{white_count}")

        print(f"Number of Empty:{count_of_empty}/{len(pkm_coordinates)}")
        result = get_results_for_image(test_images_results[i])
        print(f"Should be Empty:{len(result) - sum(result)}/{len(pkm_coordinates)}")
        f1 = f1_score(result, detected)
        all_f1_scores.append(f1)
        print(f"F1 Score:{f1}")
        cv.putText(image_view, f"F1 Score:{f1:.2f}", (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv.LINE_AA)

        cv.imshow("image", image_view)
        cv.imshow("imageCanny", image_canny)

        key = cv.waitKey()
        if key == ord('q'):
            exit(0)
        elif key == ord('b'):
            i = max(i - 1, 0)
            all_f1_scores.pop()
            all_f1_scores.pop()
            continue
        i += 1

    print(f"Final F1 Score:{sum(all_f1_scores) / len(all_f1_scores)}")


if __name__ == "__main__":
    main(sys.argv[1:])
