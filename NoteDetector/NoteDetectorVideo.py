import numpy as np
import cv2

def distanceFromOrange(image):
    orange = np.array([0, 144, 255]) # BGR orange
    
    distances = np.sqrt(np.sum((orange - image) ** 2, axis=-1)) # Calculates distances

    return distances

def find_least_indices(arr, n):
    flattened_arr = np.ravel(arr) # Flattens array
    sorted_indices = np.argsort(flattened_arr) # Sorts the array
    
    least_indices = sorted_indices[:n] # Gets top n indices

    least_indices_2d = np.unravel_index(least_indices, arr.shape) # Gets points of the top n indices
    
    return least_indices_2d

def get_largest_contour(mask):
    mask = cv2.cvtColor(mask, cv2.COLOR_BGR2GRAY) # Converting mask to gray; findContours needs this
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) # Gets the contours

    largest_contour = max(contours, key=cv2.contourArea) # Gets the largest contours

    # Gets the largest contour
    largest_contour_info = cv2.moments(largest_contour)
    cX = int(largest_contour_info["m10"] / largest_contour_info["m00"])
    cY = int(largest_contour_info["m01"] / largest_contour_info["m00"])

    return largest_contour, (cX, cY)

# Capture video from default camera (0)
cap = cv2.VideoCapture(2)

while True:
    ret, frame = cap.read()  # Read a frame from the video stream

    if not ret:
        print("Failed to capture frame")
        break

    original_frame = frame.copy()  # Create a copy to display later

    distances = distanceFromOrange(frame)  # Get distances

    least_indices = find_least_indices(distances, 24500)  # Get points of least distances

    mask = np.zeros((original_frame.shape[0], original_frame.shape[1], 3), dtype=np.uint8)  # Create blank mask

    for y, x in zip(*least_indices):
        cv2.circle(mask, (x, y), 1, (255, 255, 255), -1)  # Create mask

    kernel = np.ones((8, 8), np.uint8)
    mask = cv2.erode(mask, kernel, iterations = 1)

    largest_contour, (cX, cY) = get_largest_contour(mask)

    # Plot largest contour and its center
    cv2.drawContours(frame, [largest_contour], -1, (255, 0, 0), -1)
    cv2.circle(frame, (cX, cY), 30, (0, 0, 255), -1)

    # Display frames
    cv2.imshow("Original Frame", frame)
    cv2.imshow("Processed Frame", mask)

    # Break the loop if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture
cap.release()
cv2.destroyAllWindows()
