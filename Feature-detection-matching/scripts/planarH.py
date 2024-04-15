import numpy as np
import cv2

"""
This function compute the planar homography matrix H from a set of matched point pairs.
"""

def computeH(x1, x2):
	"""
	Compute the homography between two sets of points.

	Args:
		- x1, x2: Nx2 matrices of coordinates (x, y) of point pairs between the two images.
	Returns:
		- H2to1: 3x3 matrix for the best (least-square) homography from image 2 to 1.
	"""

	x1 = np.array(x1, dtype=float)
	x2 = np.array(x2, dtype=float)

	N = x1.shape[0] # number of samples

	# 1. Create 2x9 matrix A for each pair (i.e., 2*N x 9 matrix)
	A = np.zeros((2*N, 9))
	for i in range(N):
		x,y = x1[i,0], x1[i,1]
		u,v = x2[i,0], x2[i,1]

		A[2*i] = [-u, -v, -1, 0, 0, 0, u*x, v*x, x]
		A[2*i+1] = [0, 0, 0, -u, -v, -1, u*y, v*y, y]

	# 2. Compute the SVD of A
	_, _, Vt = np.linalg.svd(A)
	h = Vt[-1] # solution to Ah=0 is the last column of V
	H2to1 = h.reshape(3,3)

	return H2to1

def normalize_points(x):
		centroid = np.mean(x, axis=0) # centroid of the points
		x_norm =  x - centroid # Shift the origin of the points to the centroid
		max_dist = np.max(np.sqrt(np.sum(x_norm**2, axis=1)))

		scale = np.sqrt(2)/max_dist

		T = np.array([
			[scale, 0, -scale * centroid[0]],
			[0, scale, -scale * centroid[1]],
        	[0, 0, 1]
		])

		x_homo = np.hstack((x, np.ones((x.shape[0], 1))))
		x_normalized = T.dot(x_homo.T).T[:, :2]
		return x_normalized, T


def computeH_norm(x1, x2):
	x1_norm, T1 = normalize_points(x1)
	x2_norm, T2 = normalize_points(x2)

	H_norm = computeH(x1_norm, x2_norm)

	H2to1 = np.linalg.inv(T1) @ H_norm @ T2
	return H2to1


def computeH_ransac(x1, x2, num_iter = 1000, threshold=5):
	"""
	Compute the best homography using RANSAC.
	"""
	max_inliers = []
	bestH2to1 = None
	N = x1.shape[0]

	for _ in range(num_iter):
		idx = np.random.choice(N, 4, replace=False) # Randomly select 4 pairs
		x1_sample = x1[idx]
		x2_sample = x2[idx]

		H = computeH_norm(x1_sample, x2_sample)
		
		x2_homo = np.hstack((x2, np.ones((N, 1))))
		x1_projected_homo = H @ x2_homo.T # Apply homography H to img2

		# Convert from homogeneous to Cartesian
		x1_projected = x1_projected_homo[:2, :] / x1_projected_homo[2, :]
		x1_projected = x1.projected.T

		# Compute errors between projected x1 and actural x1
		errors = np.linalg.norm(x1-x1_projected, axis=1)
		curr_inliers = errors < threshold

		if sum(curr_inliers) > len(max_inliers):
			max_inliers = curr_inliers
			bestH2to1 = H
	
	inliers = max_inliers.astype(int)
	return bestH2to1, inliers



def compositeH(H2to1, template, img):
	
	#Create a composite image after warping the template image on top
	#of the image using the homography

	#Note that the homography we compute is from the image to the template;
	#x_template = H2to1*x_photo
	#For warping the template to the image, we need to invert it.
	

	#Create mask of same size as template

	#Warp mask by appropriate homography

	#Warp template by appropriate homography

	#Use mask to combine the warped template and the image
	
	return composite_img


