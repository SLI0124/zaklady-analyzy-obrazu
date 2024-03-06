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


def main():
    dart_game()


if __name__ == "__main__":
    main()
