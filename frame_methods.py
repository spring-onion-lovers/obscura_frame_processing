import pytesseract
import cv2 as cv
import numpy as np
from skimage import restoration
from scipy.signal import convolve2d as conv2

rng = np.random.default_rng()

tesseract_config = '--psm 3'

def draw_on_image(frame: np.ndarray, data_row):
    left, top, width, height = data_row[['left', 'top', 'width', 'height']]
    frame = cv.rectangle(frame, (left, top), (left + width, top + height), (0, 255, 0), 2)
    frame = cv.putText(frame, str(data_row['text']), (left, top - 10), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return frame


def ocr_frame(frame: np.ndarray, psm):

    if(psm < 0 or psm > 13):
        raise ValueError("psm must be between 0 and 13")

    tcfg = f'--psm {psm}'

    return pytesseract.image_to_data(frame, config=tcfg, output_type=pytesseract.Output.DATAFRAME)


def convert_to_grayscale(frame: np.ndarray):
    return cv.cvtColor(frame, cv.COLOR_BGR2GRAY)


def convert_to_bgr(frame: np.ndarray):
    return cv.cvtColor(frame, cv.COLOR_GRAY2BGR)


def otsu_thresh(image: np.ndarray):
    # Apply otsu thresholding
    # Otsu's thresholding after Gaussian filtering
    # blur = cv.GaussianBlur(image, (3, 3), 0)
    ret3, th3 = cv.threshold(image, 0, 255, cv.THRESH_BINARY + cv.THRESH_OTSU)
    return th3


def adaptive_gaussian_thresh(image: np.ndarray):
    return cv.adaptiveThreshold(image, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY, 11, 2)


def deblur_image(image: np.ndarray):
    psf = np.ones((5, 5)) / 25
    astro = conv2(image, psf, 'same')
    astro += 0.1 * astro.std() * rng.standard_normal(astro.shape)

    deconvolved, _ = restoration.unsupervised_wiener(astro, psf)
    return deconvolved

    # kernel = np.array([[0, -1, 0],
    #                    [-1, 5, -1],
    #                    [0, -1, 0]])
    # return cv.filter2D(image, -1, kernel)
