import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import matplotlib

# matplotlib.use("WebAgg")

# https://docs.opencv.org/4.x/d4/dbd/tutorial_filter_2d.html
# https://docs.opencv.org/4.x/d4/d13/tutorial_py_filtering.html
# https://docs.opencv.org/4.x/d2/d2c/tutorial_sobel_derivatives.html


kernel_1 = np.array([
    [0, -1, 0],
    [-1, 5, -1],
    [0, -1, 0]])

kernel_2 = np.array([
    [1, 1, 1],
    [1, 1, 1],
    [1, 1, 1]]) / 9

kernel_3 = np.ones([11, 11])
kernel_3 /= 11 * 11

dx_kernel = np.array([[-1, 0, 1]])
dy_kernel = dx_kernel.T

sobel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]])

sobel_y = np.array([
    [-1, -2, -1],
    [0, 0, 0],
    [1, 2, 1]])
