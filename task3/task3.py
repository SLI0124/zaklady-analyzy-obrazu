import glob
import os

import cv2
import numpy as np
import sys


def preprocess_image(img):
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gamma = 1.5
    img_corrected = np.uint8(np.clip((img_gray / 255.0) ** gamma * 255.0, 0, 255))
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    img_clahe = clahe.apply(img_corrected)
    return img_clahe


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
    # x-coordinates or the top-right and top-left x-coordinates
    width_a = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    width_b = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(width_a), int(width_b))
    # compute the height of the new image, which will be the
    # maximum distance between the top-right and bottom-right
    # y-coordinates or the top-left and bottom-left y-coordinates
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    # now that we have the dimensions of the new image, construct
    # the set of destination points to obtain a "birds eye view",
    # (i.e. top-down view) of the image, again specifying points
    # in the top-left, top-right, bottom-right, and bottom-left
    # order
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    # compute the perspective transform matrix and then apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    # return the warped image
    return warped


def main(argv):
    if not os.path.exists("../output/task3/results"):
        os.makedirs("../output/task3/results")

    pkm_file = open('../input/task3/map/parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coords = [line.strip().split() for line in pkm_lines]

    test_images = glob.glob("../input/task3/*.jpg")
    test_results = glob.glob("../input/task3/*.txt")

    template_images = glob.glob("../input/task3/templates/*.jpg")
    test_images.sort()
    test_results.sort()
    size = (100, 100)
    total_parking_lots = 0
    total_correct_parking_lots = 0
    font = cv2.FONT_HERSHEY_SIMPLEX

    threshold_range = np.arange(0.7, 1.0, 0.005)
    threshold_results = {}

    for threshold in threshold_range:

        total_parking_lots = 0
        total_correct_parking_lots = 0

        for img_name, result in zip(test_images, test_results):
            img = cv2.imread(img_name)
            res_lines = [int(line.strip()) for line in open(result, 'r').readlines()]
            img_result = img.copy()
            idx_parking_lot = 0

            for coord, template in zip(pkm_coords, template_images):
                idx_parking_lot += 1
                total_parking_lots += 1
                one_place_img = four_point_transform(img, coord)
                one_place_img = cv2.resize(one_place_img, size)
                temp = cv2.imread(template)
                temp_gray = preprocess_image(temp)
                one_place_img_gray = preprocess_image(one_place_img)
                res = cv2.matchTemplate(one_place_img_gray, temp_gray, cv2.TM_CCORR_NORMED)
                min_val, max_val, _, _ = cv2.minMaxLoc(res)
                left_top = (int(coord[0]), int(coord[1]))
                right_bottom = (int(coord[4]), int(coord[5]))
                center_x = (left_top[0] + right_bottom[0]) // 2
                center_y = (left_top[1] + right_bottom[1]) // 2
                cv2.putText(img_result, str(idx_parking_lot), (center_x + 3, center_y + 3), font, 1, (0, 0, 0), 2)

                if max_val > threshold:
                    cv2.circle(img_result, (center_x, center_y), 10, (0, 255, 0), -1)
                    if res_lines[idx_parking_lot - 1] == 0:
                        total_correct_parking_lots += 1
                else:
                    cv2.circle(img_result, (center_x, center_y), 10, (0, 0, 255), -1)
                    if res_lines[idx_parking_lot - 1] == 1:
                        total_correct_parking_lots += 1

        accuracy = (total_correct_parking_lots / total_parking_lots) * 100 if total_parking_lots != 0 else 0
        threshold_results[threshold] = (total_parking_lots, total_correct_parking_lots, accuracy)
        print(f"Threshold: {threshold:.3f} done")

    best_thresholds = sorted(threshold_results.items(), key=lambda x: x[1][2], reverse=True)[:5]
    print("Best 5 thresholds:")
    for threshold, (total_parking_lots, total_correct_parking_lots, accuracy) in best_thresholds:
        print(f"Threshold: {threshold:.3f}, Total parking lots: {total_parking_lots}, "
              f"Total correct parking lots: {total_correct_parking_lots}, Accuracy: {accuracy:.3f}%")

    with open("../output/task3/threshold_results.txt", "w") as f:
        for threshold, (total_parking_lots, total_correct_parking_lots, accuracy) in threshold_results.items():
            f.write(f"Threshold: {threshold:.3f}, Total parking lots: {total_parking_lots}, "
                    f"Total correct parking lots: {total_correct_parking_lots}, Accuracy: {accuracy:.3f}%\n")

        # save the best 5 thresholds to a file at the end of the file
        f.write("\nBest 5 thresholds:\n")
        for threshold, (total_parking_lots, total_correct_parking_lots, accuracy) in best_thresholds:
            f.write(f"Threshold: {threshold:.3f}, Total parking lots: {total_parking_lots}, "
                    f"Total correct parking lots: {total_correct_parking_lots}, Accuracy: {accuracy:.3f}%\n")

    # run the best threshold and save the image with the best threshold
    best_threshold = best_thresholds[0][0]
    for img_name, result in zip(test_images, test_results):
        img = cv2.imread(img_name)
        res_lines = [int(line.strip()) for line in open(result, 'r').readlines()]
        img_result = img.copy()
        idx_parking_lot = 0

        for coord, template in zip(pkm_coords, template_images):
            idx_parking_lot += 1
            one_place_img = four_point_transform(img, coord)
            one_place_img = cv2.resize(one_place_img, size)
            temp = cv2.imread(template)
            temp_gray = preprocess_image(temp)
            one_place_img_gray = preprocess_image(one_place_img)
            res = cv2.matchTemplate(one_place_img_gray, temp_gray, cv2.TM_CCORR_NORMED)
            min_val, max_val, _, _ = cv2.minMaxLoc(res)
            left_top = (int(coord[0]), int(coord[1]))
            right_bottom = (int(coord[4]), int(coord[5]))
            center_x = (left_top[0] + right_bottom[0]) // 2
            center_y = (left_top[1] + right_bottom[1]) // 2
            cv2.putText(img_result, str(idx_parking_lot), (center_x + 3, center_y + 3), font, 1, (0, 0, 0), 2)

            if max_val > best_threshold:
                cv2.circle(img_result, (center_x, center_y), 10, (0, 255, 0), -1)
                if res_lines[idx_parking_lot - 1] == 0:
                    total_correct_parking_lots += 1
            else:
                cv2.circle(img_result, (center_x, center_y), 10, (0, 0, 255), -1)
                if res_lines[idx_parking_lot - 1] == 1:
                    total_correct_parking_lots += 1

        # save the results to a file with the same name as the image and the threshold
        cv2.putText(img_result, f"Threshold: {best_threshold:.3f}", (10, 30), font, 1, (0, 0, 0), 2)
        cv2.imwrite(f"../output/task3/results/{img_name.split('\\')[-1].split('.')[0]}_{best_threshold:.3f}.jpg",
                    img_result)
        cv2.imshow("Result", img_result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main(sys.argv)
