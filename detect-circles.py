import cv2
import numpy as np

# Load image
image = cv2.imread('shapes.jpg', 0)

params = cv2.SimpleBlobDetector_Params()

params.filterByArea = True

params.minArea = 100


params.filterByCircularity = True
params.minCircularity = 0.9


params.filterByConvexity = True
params.minConvexity = 0.2


params.filterByInertia = True
params.minInertiaRatio = 0.01


detector = cv2.SimpleBlobDetector_create(params)

# Detect blobs
keypoints = detector.detect(image)

# Draw blobs on our image as red circles
blank = np.zeros((1, 1))
blobs = cv2.drawKeypoints(image, keypoints, blank, (255,155,100), cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS) 

number_of_blobs = len(keypoints)
text = "Number of Circular Blobs: " + str(len(keypoints))

cv2.putText(blobs, text, (20, 550), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,100), 2)

# Show blobs
cv2.imshow("Filtering Circular Blobs Only", blobs)

cv2.imwrite('detectCircle.jpg', image)

cv2.waitKey()

# Close all windows
cv2.destroyAllWindows()
