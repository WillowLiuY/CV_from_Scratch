import numpy as np
import cv2
from matchPics import matchPics
from scipy.ndimage import rotate
import matplotlib.pyplot as plt


#Q3.5
#Read the image and convert to grayscale, if necessary
image_path = 'cv_cover.jpg'
image = cv2.imread(image_path)

if image is None:
    raise FileNotFoundError(f"The file {image_path} was not found.")

if len(image.shape) == 3:
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
else:
    image_gray = image

hist_matches = []
angles = range(0, 360, 10)
for i in range(36):
	#Rotate Image
    angle = angles[i]
    rotated_image = rotate(image_gray, angle, reshape=False)
	
	#Compute features, descriptors and Match features
    matches, _, _ = matchPics(image_gray, rotated_image)

	#Update histogram
    hist_matches.append(len(matches))


#Display histogram
plt.bar(angles, hist_matches, width=8, edgecolor='black')
plt.title('Histogram of Matches at Different Rotations')
plt.xlabel('Rotation Angle (degrees)')
plt.ylabel('Number of Matches')
plt.xticks(angles, labels=[str(a) for a in angles], rotation=90)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()
