import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib


def cv_1():
    cv2.namedWindow("Image", 0)
    # cv2.resizeWindow("Image", 800, 600)

    image_color = cv2.imread("../input/task1/img.png", 0)
    image_color2 = cv2.imread("../input/task1/img.png", 0)

    image_gray = cv2.imread("../input/task1/img.png", 1)
    image_gray2 = cv2.imread("../input/task1/img.png", 1)

    image_gray_resized = cv2.resize(image_gray, (150, 150))

    # image_hc = cv2.hconcat([image_color, image_color2])
    image_hc = cv2.hconcat([image_gray_resized, image_gray_resized])

    cv2.imshow("Image", image_hc)
    cv2.imwrite("../output/task1/img_hc.png", image_hc)
    cv2.waitKey(0)


def cv_2():
    image = cv2.imread("../input/task1/img.png", 1)
    image2 = cv2.imread("../input/task1/img.png", 1)

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image2 = cv2.resize(image, (150, 150))

    plt.subplot(2, 1, 1)
    plt.imshow(image)
    plt.subplot(2, 1, 2)
    plt.imshow(image2)
    plt.show()


def cv_3():
    cv2.namedWindow("Image", 0)
    img = np.zeros((5, 5, 3), dtype=np.uint8)
    print(img.shape)
    img[1, 4] = [255, 255, 255]
    print(img[1, 4])

    cv2.imshow("Image", img)
    cv2.waitKey(0)


def cv_4():
    cam = cv2.VideoCapture(0)
    result = cv2.VideoWriter("../output/task1/output.mp4",
                             cv2.VideoWriter_fourcc(*'X264'),
                             20.0,
                             (640, 480))

    while True:
        ret, frame = cam.read()
        edges = cv2.Canny(frame, 50, 100)
        result.write(edges)
        cv2.imshow("Frame", edges)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    result.release()

    cv2.destroyAllWindows()


def main():
    matplotlib.use("WebAgg")
    print(f"Version of OpenCV: {cv2.__version__}")

    # cv_1()
    cv_2()
    # cv_3()
    # cv_4()


if __name__ == '__main__':
    main()
