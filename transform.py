import os
import cv2 as cv
import numpy as np


def fft(img):
    img = np.fft.fft2(img)
    img = np.fft.fftshift(img)
    return img


def mag(img):
    absvalue = np.abs(img)
    magnitude = 20 * np.log(absvalue)
    return magnitude


def binarize_images(origin, destiny, iterator):
    i = 0
    images = os.listdir(f"./{origin}/{j}")
    for this_img in images:
        img = cv.imread(f'./{origin}/{iterator}/{this_img}')
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

        img = cv.dilate(img, np.ones((2, 2)))

        threshold, img = cv.threshold(
            img, 0, 255, cv.THRESH_OTSU
        )
        cv.imwrite(f'./{destiny}/{iterator}/{i}.png', img)
        i += 1


def fourier(origin, destiny, iterator):
    i = 0
    images = os.listdir(f"./{origin}/{k}")
    for this_img in images:
        img = cv.imread(f'./{origin}/{iterator}/{this_img}', cv.COLOR_BGRA2GRAY)

        img = fft(img)
        img = mag(img)

        cv.imwrite(f'./{destiny}/{iterator}/{i}.png', img)
        i += 1


# for j in range(6):
#     binarize_images("ds", "binarized", j)
#     fourier("ds", "fourier", j)

for k in range(6):
    fourier("binarized", "binarized_fourier", k)
