import numpy as np
import cv2

def distanceFromOrange(image):
    orange = np.array([0, 144, 255]) #bgr orange
    
    distances = np.linalg.norm(image - orange, axis=-1) #calculates distances

    return distances

def find_least_indices(arr, n):
    flattened_arr = np.ravel(arr) #flattens array
    sorted_indices = np.argsort(flattened_arr) #sort the array
    
    least_indices = sorted_indices[:n] #gets top n indices

    least_indices_2d = np.unravel_index(least_indices, arr.shape) #gets points of the top n indices
    
    return least_indices_2d

def get_largest_contour(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # converting mask to gray; findContours needs this
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # gets the contours

    largest_contour = max(contours, key=cv2.contourArea) # gets the largest contours

    # gets the largest contour
    largest_contour_info = cv2.moments(largest_contour)
    cX = int(largest_contour_info["m10"] / largest_contour_info["m00"])
    cY = int(largest_contour_info["m01"] / largest_contour_info["m00"])

    return largest_contour, (cX, cY)

image = cv2.imread(r'NoteImages\Note2.jpg') # importing image
original_image = image.copy() # creating a copy so we can see later

distances = distanceFromOrange(image) # getting distances

least_indices = find_least_indices(distances, 200000) # getting points of least distances

mask = np.zeros((original_image.shape[1], original_image.shape[0], 3), dtype=np.uint8) # creates blank mask

for y, x in zip(*least_indices):
    cv2.circle(image, (x, y), 1, (0, 255, 0), -1) # plotting points
    cv2.circle(mask, (x, y), 1, (255, 255, 255), -1) # creating mask

largest_contour, (cX, cY) = get_largest_contour(mask)

# plots largest contour and largest contour's center
cv2.drawContours(image, [largest_contour], -1, (255, 0, 0), -1)
cv2.circle(image, (cX, cY), 30, (0, 0, 255), -1)

# displays image
cv2.imshow("Original Image", cv2.resize(original_image, (int(original_image.shape[1]/3), int(original_image.shape[0]/3))))
cv2.imshow("Plotted Image", cv2.resize(image, (int(image.shape[1]/3), int(image.shape[0]/3))))
cv2.waitKey(0)
cv2.destroyAllWindows()