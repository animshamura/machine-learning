import cv2
import matplotlib.pyplot as plt

# Load images
img1 = cv2.imread('img1.jpg', 0)
img2 = cv2.imread('img2.jpg', 0)

# ORB Detector
orb = cv2.ORB_create()
kp1, des1 = orb.detectAndCompute(img1, None)
kp2, des2 = orb.detectAndCompute(img2, None)

# Brute Force Matcher
bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
matches = bf.match(des1, des2)
matches = sorted(matches, key=lambda x: x.distance)

# Draw matches
img_match = cv2.drawMatches(img1, kp1, img2, kp2, matches[:20], None, flags=2)
plt.imshow(img_match)
plt.title("ORB Feature Matching")
plt.show()
