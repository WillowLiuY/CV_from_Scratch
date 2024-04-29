import numpy as np

def myHoughTransform(img_threshold, rhoRes, thetaRes):
    """
    Args:
        - img_threshold: the edge magnitude image
        - rhoRes: scalar, distance resolution of the Hough transform accumulator in pixels
        - thetaRes: scalar, angular resolution of the accumulator in radians
    Returns:
        - img_hough: Hough transform accumulator matrix
        - rhoScale: arrays of rho values
        - thetaScale: arrays of theta values
    """

    # 1.Initialize accumulators
    rho_max = np.hypot(img_threshold.shape[0], img_threshold.shape[1])
    theta_max = np.pi

    rho_num = int(np.ceil(rho_max/rhoRes)) * 2 # Discretize the range of rho and theta
    theta_num = int(np.ceil(theta_max/thetaRes))

    rhoScale = np.linspace(-rho_max, rho_max, rho_num)
    thetaScale = np.linspace(0, theta_max, theta_num)

    img_hough = np.zeros((rho_num, theta_num)) # Initialize the accumulator (img_hough) as a 2D array filled with zeros. 
    
    # 2.Voting
    y_idxs, x_idxs = np.nonzero(img_threshold)
    
    for i in range(len(x_idxs)):
        x = x_idxs[i]
        y = y_idxs[i]
        for j in range(theta_num):
            theta = thetaScale[j]
            rho = x * np.cos(theta) + y * np.sin(theta)
            rho_idx = np.argmin(np.abs(rhoScale - rho))
            img_hough[rho_idx, j] += 1
    
    return img_hough, rhoScale, thetaScale