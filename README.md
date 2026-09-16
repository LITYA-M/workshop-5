#  Workshop -5 License-Plate-Detection
##  Aim
To detect a vehicle license plate from an image using OpenCV Haar Cascade Classifier and blur the detected license plate region to protect the vehicle owner's privacy.

## Software Required
Python 3.x

Jupyter Notebook / JupyterLab

OpenCV (cv2)

Matplotlib

Haar Cascade license plate classifier

Input image: car_plate.jpg

Cascade file: haarcascade_licence_plate_rus_16stages.xml

Install Required Libraries

pip install opencv-python numpy matplotlib

##  Algorithm
Import OpenCV, NumPy, and Matplotlib.

Read the input vehicle image using cv2.imread().

Load the license plate Haar Cascade XML classifier.

Use detectMultiScale() to detect possible license plate regions.

For every detected plate, obtain its coordinates (x, y, w, h).

Draw a rectangle around the detected license plate to verify detection.

Extract the detected license plate as a Region of Interest (ROI).

Apply median blur to the ROI using cv2.medianBlur().

Replace the original license plate region with the blurred ROI.

Display the final image.

## Program
Developed By :  LITYA M
Reg No: 212225230152
```
import numpy as np
import cv2
import matplotlib.pyplot as plt

%matplotlib inline

# Read image
img = cv2.imread('car_plate.jpg')

# Display image
def display(img):
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.figure(figsize=(12, 8))
    plt.imshow(img)
    plt.axis('off')
    plt.show()

display(img)

# Load license plate Haar Cascade
plate_cascade = cv2.CascadeClassifier(
    'haarcascade_licence_plate_rus_16stages.xml'
)

# Detect license plate and draw rectangle
def detect_plate(img):
    img_copy = img.copy()

    plates = plate_cascade.detectMultiScale(
        img_copy,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:
        cv2.rectangle(
            img_copy,
            (x, y),
            (x + w, y + h),
            (255, 0, 0),
            3
        )

    return img_copy

# Display detected plate
result = detect_plate(img)
display(result)

# Detect and blur license plate
def detect_and_blur_plate(img):
    img_copy = img.copy()

    plates = plate_cascade.detectMultiScale(
        img_copy,
        scaleFactor=1.1,
        minNeighbors=5
    )

    for (x, y, w, h) in plates:
        roi = img_copy[y:y+h, x:x+w]

        # Apply median blur
        blurred_roi = cv2.medianBlur(roi, 25)

        # Replace original ROI with blurred ROI
        img_copy[y:y+h, x:x+w] = blurred_roi

    return img_copy

# Display final result
result = detect_and_blur_plate(img)
display(result)
```
## Output
<img width="1162" height="662" alt="image" src="https://github.com/user-attachments/assets/8825e8f8-b098-4fcb-a806-02a1dce1424b" />
<img width="1165" height="655" alt="image" src="https://github.com/user-attachments/assets/1ddb1cee-604a-476c-a433-d1586e3e6500" />
<img width="1160" height="647" alt="image" src="https://github.com/user-attachments/assets/a4790aac-744b-41d4-930f-96862ae8ee34" />


## Result
Thus, the vehicle license plate was successfully detected using the Haar Cascade Classifier in OpenCV and the detected license plate region was successfully blurred using a median blur filter.










..




..




..
