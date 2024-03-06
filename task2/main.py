import cv2 as cv
import numpy as np
from pynput.mouse import Button, Controller
from PIL import ImageGrab
from time import sleep

mouse = Controller()


def dart_game():
    # URL: https://www.addictinggames.com/shooting/dart-master
    template_path = '../input/task2/dart_template.png'
    target_image = cv.imread(template_path)
    target_gray = cv.cvtColor(target_image, cv.COLOR_BGR2GRAY)

    while True:
        sleep(0.7)
        img = ImageGrab.grab()
        img = np.array(img)
        img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        template = target_gray

        # Get template dimensions
        w, h = template.shape[::-1]

        # Apply template Matching
        res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

        # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
        top_left = max_loc

        if len(max_loc) == 0:
            print("No match found")
            continue

        if max_val < 0.33:
            print("No match found")
            continue

        # get the center of the rectangle
        center = (top_left[0] + w // 2, top_left[1] + h // 2)
        print(f"Center: {center}, Max Val: {max_val}")

        # click the center of the rectangle using mouse
        mouse.position = center
        mouse.click(Button.left, 1)


def duck_game():
    # URL: https://duckhuntjs.com/index.html?title=&waves=3&ducks=10&points=100&bullets=100&radius=60&speed=3&time=30
    duck1_gray = cv.imread('../input/task2/duck1.png', cv.IMREAD_GRAYSCALE)
    duck2_gray = cv.imread('../input/task2/duck2.png', cv.IMREAD_GRAYSCALE)
    duck1_gray_mirrored = cv.flip(duck1_gray, 1)
    duck2_gray_mirrored = cv.flip(duck2_gray, 1)

    while True:
        for duck in [duck1_gray, duck2_gray, duck1_gray_mirrored, duck2_gray_mirrored]:
            sleep(0.7)
            img = ImageGrab.grab()
            img = np.array(img)
            img = cv.cvtColor(img, cv.COLOR_RGB2BGR)
            img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

            template = duck

            # Get template dimensions
            w, h = template.shape[::-1]

            # Apply template Matching
            res = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
            min_val, max_val, min_loc, max_loc = cv.minMaxLoc(res)

            # If the method is TM_SQDIFF or TM_SQDIFF_NORMED, take minimum
            top_left = max_loc

            if len(max_loc) == 0:
                print("No match found")
                continue

            if max_val < 0.38:
                print("No match found")
                continue

            # get the center of the rectangle
            center = (top_left[0] + w // 2, top_left[1] + h // 2)
            print(f"Center: {center}, Max Val: {max_val}")

            # click the center of the rectangle using mouse
            mouse.position = center
            mouse.click(Button.left, 1)


def main():
    # dart_game()
    duck_game()


if __name__ == "__main__":
    main()
