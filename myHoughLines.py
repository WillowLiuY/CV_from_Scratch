import numpy as np
import cv2  # For cv2.dilate function

def myHoughLines(img_hough, nLines):
    # YOUR CODE HERE
    """
    Args:
        - img_hough: hough accumulator
        - nLine: number of lines to return
    Returns:
        - rhos: (nLines, 1) vectors that contains the row coordinates of peaks in img_hough
        - thetas: (nLines, 1) vector that contains the column coordinates of peaks in img_hough
    """
    