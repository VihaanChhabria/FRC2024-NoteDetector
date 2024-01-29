# FRC2024-NoteDetector
 Detects where notes are on image using color detection in Python.

This code performs the following tasks:

* Calculates the distances of pixels in an image from a predefined orange color.

* Identifies the points with the least distances from the orange color.

* Constructs a mask using these points.

* Finds the largest contour in the mask and its centroid.

* Draws the identified points and contour on the original image.

* Displays the original image with the plotted points and contour.

Overall, the code aims to locate and highlight regions in the image that are closest to the specified orange color.
