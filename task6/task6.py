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
                f"Label: {label}, Predicted: {model.predict([get_lbp_features(img, best_result['P'],
                                                                              best_result['R'])])[0]}")
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


def parking_lot_lbp():
    data = train_or_load_lbp_svm('../input/task6/parking_lot/', ['free', 'full'],
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

    visualize_best_result(best_result, model, empty_images, occupied_images,
                          '../output/task6/parking_lot_predictions.png')


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

    visualize_best_result(best_result, model, open_images, closed_images,
                          '../output/task6/eye_open_close_predictions.png')


def main():
    # parking_lot_lbp()
    open_close_eyes_recognition_lbp()


if __name__ == "__main__":
    main()
