import numpy as np
import cv2
from skimage.feature import peak_local_max

def myHoughLines(img_hough, nLines, rhoScale, thetaScale):
    """
    Args:
        - img_hough: hough accumulator
        - nLine: number of lines to return
    Returns:
        - rhos: (nLines, 1) vectors that contains the row coordinates of peaks in img_hough
        - thetas: (nLines, 1) vector that contains the column coordinates of peaks in img_hough
    """
    # Create a structuring element for dilation
    kernel_size = (10, 10)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)

    # Dilation for NMS
    dilated = cv2.dilate(img_hough, kernel)

    # Identify peaks in the original image where it equals the dilated image
    maxima = (img_hough == dilated)

    # Find peaks considering only maxima
    peaks = peak_local_max(img_hough, num_peaks=nLines, indices=True, labels=maxima.astype(np.uint8))

    # Extract rho and theta values for the peaks using scales
    rhos = [rhoScale[idx[0]] for idx in peaks]
    thetas = [thetaScale[idx[1]] for idx in peaks]

    return rhos, thetas