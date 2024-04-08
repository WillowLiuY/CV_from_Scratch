import numpy as np
from scipy import signal    # For signal.gaussian function

from myImageFilter import myImageFilter

def myEdgeFilter(img0, sigma):
    """
    Args:
        img0: greyscale image.
        sigma: std of the Gaussian smoothing kernel.
    Returns:
        img1: the edge magnitutude image
    """
    # 1. Define the Gaussian kernel
    hsize = int(2 * np.ceil(3*sigma) + 1)
    gs_kernel = signal.windows.gaussian(hsize, std=sigma).reshape(hsize, 1) # 1D gaussian filter

    gs_kernel_2d = np.outer(gs_kernel, gs_kernel) # 2D gaussian kernel using outer product
    gs_kernel_2d /= np.sum(gs_kernel_2d) # normalize

    # 2. Convolution with the Gaussian kernel to reduce noise and spurious fine edges
    img_smoothed = myImageFilter(img0, gs_kernel_2d)

    # 3. Compute the gradient at x and y directions using Sobel filter
    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]])
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

    img_grad_x, img_grad_y = myImageFilter(img_smoothed, sobel_x), myImageFilter(img_smoothed, sobel_y)

    img_grad_magn = np.sqrt(img_grad_x**2 + img_grad_y**2) # Compute gradient magnitude == detected edges before NMS
    img_grad_dirt = np.rad2deg(np.arctan2(img_grad_y, img_grad_x)) # Compute gradient direction

    # 4. NMS
    img1 = np.zeros_like(img_grad_magn)
    for i in range(1, img_grad_magn.shape[0] - 1):
        for j in range(1, img_grad_magn.shape[1] - 1):
            current_orientation = img_grad_dirt[i, j]
            if (-22.5 <= current_orientation < 22.5) or (157.5 <= current_orientation <= 180) or (-180 <= current_orientation < -157.5):
                neighbors = [img_grad_magn[i, j-1], img_grad_magn[i, j+1]]
            elif (22.5 <= current_orientation < 67.5) or (-157.5 <= current_orientation < -112.5):
                neighbors = [img_grad_magn[i-1, j+1], img_grad_magn[i+1, j-1]]
            elif (67.5 <= current_orientation < 112.5) or (-112.5 <= current_orientation < -67.5):
                neighbors = [img_grad_magn[i-1, j], img_grad_magn[i+1, j]]
            else: # 112.5 to 157.5 and -67.5 to -22.5
                neighbors = [img_grad_magn[i-1, j-1], img_grad_magn[i+1, j+1]]
            
            # Suppress non-maximum
            if img_grad_magn[i, j] >= max(neighbors):
                img1[i, j] = img_grad_magn[i, j]
    return img1