import cv2
import numpy as np

# Load the image
image = cv2.imread('curves.png')  # Replace with the path to your image file

# Convert BGR to HSV
hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Define the range of blue color in HSV
lower_blue = np.array([100, 50, 50])
upper_blue = np.array([140, 255, 255])

# Threshold the HSV image to get only blue colors
mask = cv2.inRange(hsv, lower_blue, upper_blue)

# Bitwise-AND mask and original image
blue_curve = cv2.bitwise_and(image, image, mask=mask)

# Convert the blue curve image to grayscale
gray_curve = cv2.cvtColor(blue_curve, cv2.COLOR_BGR2GRAY)

# Find contours in the grayscale image
contours, _ = cv2.findContours(gray_curve, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

# Find the contour with the maximum area (assuming it's the curve)
max_contour = max(contours, key=cv2.contourArea)

# Extract x and y values from the contour
x_values = max_contour[:, 0, 0]
y_values = max_contour[:, 0, 1]

# Print x and y values
for x, y in zip(x_values, y_values):
    print(f"X: {x}, Y: {y}")

# Optionally, you can visualize the blue curve
cv2.drawContours(image, [max_contour], -1, (0, 255, 0), 2)
cv2.imshow('Blue Curve', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

