import cv2
from PIL import Image
import pytesseract
import numpy as np
import os


def get_max_temp(filepathy1):
    temp_im = Image.open(filepathy1)
    max_temp_reading_image = temp_im.crop((278, 42, 318, 64))
    max_temp_reading_image.save("main_data_set/temp_ocr.jpg")
    img = cv2.imread("main_data_set/temp_ocr.jpg")
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray, img_bin = cv2.threshold(gray, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    gray = cv2.bitwise_not(img_bin)
    kernel = np.ones((2, 1), np.uint8)
    img = cv2.erode(gray, kernel, iterations=1)
    img = cv2.dilate(img, kernel, iterations=1)
    pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract'
    max_temp_reading = pytesseract.image_to_string(img)
    max_temp_reading = str(max_temp_reading[:-1]+"."+max_temp_reading[-1])
    os.remove("main_data_set/temp_ocr.jpg")
    try:
        tempi = float(max_temp_reading)
    except ValueError:
        tempi = 77.7
    return tempi


def max_temp_pixel_value(filepathy2):
    im = Image.open(filepathy2)
    coordinate = x, y = 310, 70
    xfi = list(im.getpixel(coordinate))
    xfi.reverse()
    return xfi
