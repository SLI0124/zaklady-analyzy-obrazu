import glob

import cv2
import numpy as np
import os
import time
import json

from matplotlib import pyplot as plt
from skimage import feature
from sklearn import svm
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def get_lbp_features(image, p, r):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    lbp = feature.local_binary_pattern(gray, p, r, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=np.arange(0, p + 3), range=(0, p + 2))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-7)
    return hist


def calculate_accuracy(input_model, test_feat, test_labels):
    predictions = input_model.predict(test_feat)
    acc = accuracy_score(test_labels, predictions)
    return acc * 100


def visualize_best_result(best_result, model, empty_images, occupied_images, output_path):
    fig, axs = plt.subplots(5, 5, figsize=(15, 15))
    for i in range(5):
        for j in range(5):
            idx = np.random.randint(0, len(empty_images) + len(occupied_images))
            if idx < len(empty_images):
                img = empty_images[idx]
                label = 0
            else:
                img = occupied_images[idx - len(empty_images)]
                label = 1

            axs[i, j].imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            axs[i, j].set_title(
                f"Label: {label}, Prediction: {model.predict([get_lbp_features(img, best_result['P'], best_result['R'])])[0]}")
            axs[i, j].axis('off')

    plt.savefig(output_path)
    plt.show()


def train_lbp_svm(input_path, classes, output_path):
    images = []
    labels = []
    for i, class_name in enumerate(classes):
        class_images = [cv2.imread(input_path + class_name + '/' + file) for file in
                        os.listdir(input_path + class_name + '/')]
        images.extend(class_images)
        labels.extend([i] * len(class_images))

    lbp_configs = [(P, R) for P in range(1, 50) for R in range(1, 50)]

    results = []
    idx = 0

    for P, R in lbp_configs:
        process_start = time.time()

        features = []
        for img in images:
            features.append(get_lbp_features(img, P, R))

        features = np.array(features)
        labels = np.array(labels)

        (trainFeat, testFeat, trainLabels, testLabels) = train_test_split(features, labels, test_size=0.25,
                                                                          random_state=42)

        model = svm.SVC(kernel='linear', C=1.0, probability=True)
        model.fit(trainFeat, trainLabels)

        accuracy = calculate_accuracy(model, testFeat, testLabels)
        processing_time = time.time() - process_start

        results.append({
            'P': P,
            'R': R,
            'accuracy': accuracy,
            'processing_time': processing_time
        })

        idx += 1
        print(f"\rProgress: {idx}/{len(lbp_configs)}", end='')

    results.sort(key=lambda x: (x['accuracy'], x['processing_time'], x['P'], x['R']), reverse=True)

    with open(output_path, 'w') as json_file:
        json.dump(results, json_file)


def train_or_load_lbp_svm(input_path, classes, output_path):
    try:
        with open(output_path, 'r') as f:
            data = json.load(f)
            print("Loaded existing results")
    except FileNotFoundError:
        print("No existing results found, training SVM model...")
        train_lbp_svm(input_path, classes, output_path)
        print("Training completed")
        with open(output_path, 'r') as f:
            data = json.load(f)

    return data


def print_best_results(data, threshold=90):
    best_results = [x for x in data if x['accuracy'] >= threshold]
    best_results.sort(key=lambda x: (x['processing_time'], x['P'], x['R']))

    print("P\tR\tAccuracy\tProcessing Time")
    for i in range(len(best_results)):
        print(
            f"{best_results[i]['P']}\t{best_results[i]['R']}\t{best_results[i]['accuracy']}\t"
            f"{best_results[i]['processing_time']}")

    best_result = best_results[0]
    print(f"Best result parameters: P={best_result['P']}, R={best_result['R']}")


def calculate_total_training_time(data):
    total_time = 0
    for result in data:
        total_time += result['processing_time']

    return total_time


def final_visualization_parking_spaces(model, p, r):
    from task3.task3 import four_point_transform

    pkm_file = open('../input/task3/map/parking_map_python.txt', 'r')
    pkm_lines = pkm_file.readlines()
    pkm_coords = [line.strip().split() for line in pkm_lines]

    test_images = glob.glob("../input/task4/*.jpg")
    test_results = glob.glob("../input/task4/*.txt")

    template_images = glob.glob("../input/task3/templates/*.jpg")
    test_images.sort()
    test_results.sort()
    size = (80, 80)
    tp, tn, fp, fn = 0, 0, 0, 0
    total = 0
    idx = 0

    for image_name in test_images:
        image = cv2.imread(image_name)
        image_result = image.copy()
        idx_parking_lot = 0

        for coord, template in zip(pkm_coords, template_images):
            one_place_img = four_point_transform(image, coord)
            one_place_img = cv2.resize(one_place_img, size)

            # using the best LBP parameters, detect if the parking space is occupied or not
            features = get_lbp_features(one_place_img, p, r)
            prediction = model.predict([features])[0]

            left_top = (int(coord[0]), int(coord[1]))
            right_bottom = (int(coord[4]), int(coord[5]))
            center_x = (left_top[0] + right_bottom[0]) // 2
            center_y = (left_top[1] + right_bottom[1]) // 2

            if prediction == 0:
                cv2.circle(image_result, (center_x, center_y), 10, (0, 255, 0), -1)
            else:
                cv2.circle(image_result, (center_x, center_y), 10, (0, 0, 255), -1)

            with open(test_results[idx], 'r') as f:
                lines = f.readlines()
                lines = [line.strip().split() for line in lines]

                if prediction == 0 and int(lines[idx_parking_lot][0]) == 0:
                    tp += 1
                elif prediction == 0 and int(lines[idx_parking_lot][0]) == 1:
                    fp += 1
                elif prediction == 1 and int(lines[idx_parking_lot][0]) == 0:
                    fn += 1
                elif prediction == 1 and int(lines[idx_parking_lot][0]) == 1:
                    tn += 1

            total += 1
            idx_parking_lot += 1

            cv2.putText(image_result, str(idx_parking_lot), (center_x + 3, center_y + 3), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (0, 0, 0), 2)

        idx += 1

        scale_percent = 50
        width = int(image_result.shape[1] * scale_percent / 100)
        height = int(image_result.shape[0] * scale_percent / 100)
        dim = (width, height)
        image_result = cv2.resize(image_result, dim, interpolation=cv2.INTER_AREA)

        # cv2.imshow('Parking Spaces', image_result)
        # if cv2.waitKey(0) & 0xFF == ord('q'):
        #     break
        # cv2.waitKey(0)
        # cv2.destroyAllWindows()

    print(f"True Positive: {tp}")
    print(f"True Negative: {tn}")
    print(f"False Positive: {fp}")
    print(f"False Negative: {fn}")
    print(f"Total: {total}")
    print(f"Accuracy: {(tp + tn) / total}")
    precision = tp / (tp + fp) if tp + fp != 0 else 0
    print(f"Precision: {precision}")


def final_visualization_open_close_eyes(model, p, r):
    from task5.task5 import detect_faces
    from task5.task5 import draw_rectangle
    from task5.task5 import remove_duplicates

    def eye_open(eye_img):
        eye_img = cv2.resize(eye_img, (80, 80))
        features = get_lbp_features(eye_img, p, r)
        prediction = model.predict([features])[0]
        return prediction == 0

    video_cap = cv2.VideoCapture("../input/task5/fusek_face_car_01.avi")
    # video_cap = cv.VideoCapture(0)
    face_cascade = cv2.CascadeClassifier("../input/task5/haarcascades/haarcascade_frontalface_default.xml")
    face_cascade_profile = cv2.CascadeClassifier("../input/task5/haarcascades/haarcascade_profileface.xml")
    eye_cascade = cv2.CascadeClassifier("../input/task5/eye_cascade_fusek.xml")
    mouth_cascade = cv2.CascadeClassifier("../input/task5/haarcascades/haarcascade_smile.xml")

    while True:
        ret, frame = video_cap.read()
        if frame is None:
            break
        paint_frame = frame.copy()

        locations_face_front = detect_faces(face_cascade, paint_frame, 1.2, 7, (100, 100))
        locations_face_profile = detect_faces(face_cascade_profile, frame, 1.2, 7, (100, 100))

        # If only front faces are detected, use them
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

                mouth = detect_faces(mouth_cascade, face_roi, 1.2, 50, (40, 40))
                for m in mouth:
                    draw_rectangle(paint_frame[y:y + h, x:x + w], [m], (255, 0, 0), (203, 192, 255))

            cv2.imshow("face_detect", paint_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break


def parking_lot_lbp():
    data = train_or_load_lbp_svm('../input/task6/free_full/', ['free', 'full'],
                                 '../output/task6/parking_lot_results.json')
    best_result = data[0]

    best_results = [x for x in data if x['accuracy'] == 100]
    best_results.sort(key=lambda x: (x['processing_time'], x['P'], x['R']))

    print_best_results(data, 100)
    total_training_time = calculate_total_training_time(data)
    print(f"Total training time took {total_training_time / 60:.2f} minutes and {total_training_time % 60:.2f} seconds")

    path = '../input/task6/free_full/'
    empty_images = [cv2.imread(path + 'free/' + file) for file in os.listdir(path + 'free/')]
    occupied_images = [cv2.imread(path + 'full/' + file) for file in os.listdir(path + 'full/')]

    # Train SVM model with the best parameters for displaying predictions on random images
    model = svm.SVC(kernel='linear', C=1.0, probability=True)
    features = []
    labels = []
    for img in empty_images:
        features.append(get_lbp_features(img, best_result['P'], best_result['R']))
        labels.append(0)

    for img in occupied_images:
        features.append(get_lbp_features(img, best_result['P'], best_result['R']))
        labels.append(1)

    model.fit(features, labels)

    print(f"Best result parameters: P={best_result['P']}, R={best_result['R']}")

    # visualize_best_result(best_result, model, empty_images, occupied_images,
    #                       '../output/task6/parking_lot_predictions.png')

    final_visualization_parking_spaces(model, best_result['P'], best_result['R'])


def open_close_eyes_recognition_lbp():
    data = train_or_load_lbp_svm('../input/task6/dataset/train/', ['Open_Eyes', 'Closed_Eyes'],
                                 '../output/task6/eye_open_close_results.json')

    best_results = [x for x in data if x['accuracy'] >= 90]
    best_results.sort(key=lambda x: (x['accuracy'], x['processing_time'], x['P'], x['R']), reverse=True)

    print(len(best_results))

    best_result = best_results[0]

    print_best_results(data)
    total_training_time = calculate_total_training_time(data)
    print(f"Total training time took {total_training_time / 60:.2f} minutes and {total_training_time % 60:.2f} seconds")

    path = '../input/task6/dataset/test/'
    open_images = [cv2.imread(path + 'Open_Eyes/' + file) for file in os.listdir(path + 'Open_Eyes/')]
    closed_images = [cv2.imread(path + 'Closed_Eyes/' + file) for file in os.listdir(path + 'Closed_Eyes/')]

    # Train SVM model with the best parameters for displaying predictions on random images
    model = svm.SVC(kernel='linear', C=1.0, probability=True)
    features = []
    labels = []
    for img in open_images:
        img = cv2.resize(img, (80, 80))
        features.append(get_lbp_features(img, best_result['P'], best_result['R']))
        labels.append(0)

    for img in closed_images:
        img = cv2.resize(img, (80, 80))
        features.append(get_lbp_features(img, best_result['P'], best_result['R']))
        labels.append(1)

    model.fit(features, labels)

    # visualize_best_result(best_result, model, open_images, closed_images,
    #                       '../output/task6/eye_open_close_predictions.png')

    print(f"Best result parameters: P={best_result['P']}, R={best_result['R']}")

    final_visualization_open_close_eyes(model, best_result['P'], best_result['R'])


def main():
    # parking_lot_lbp()
    open_close_eyes_recognition_lbp()


if __name__ == "__main__":
    main()
