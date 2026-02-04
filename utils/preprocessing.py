import numpy as np
import cv2
 
def preprocess_image(img, target_size=(224, 224)):
    kernel_filter = np.array([[0, -1, 0],
                              [-1, 5, -1],
                              [0, -1, 0]])
    # Resize
    img = cv2.resize(img, target_size)
    # Grayscale
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Histogram Equalization
    img = cv2.equalizeHist(img)
    # Noise Reduction
    img = cv2.medianBlur(img, 5)
    # Sharpening
    img = cv2.filter2D(img, -1, kernel_filter)
    # Normalization
    img = img / 255.0
    return np.asarray(img)