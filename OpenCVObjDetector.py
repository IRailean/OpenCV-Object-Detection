#!/usr/bin/env python
# coding: utf-8

# In[43]:




from __future__ import division
import cv2
from matplotlib import pyplot as plt
from math import cos, sin
import numpy as np

green = (0, 255, 0)

def show(image):
    plt.figure(figsize = [10, 10])
    plt.imshow(image, interpolation = 'nearest')
    
def overlay_mask(mask, image):
    # Make mask RGB
    rgb_mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2RGB)
    
    img = cv2.addWeighted(rgb_mask, 0.5, image, 0.5, 0)
    return img

def find_biggest_contour(image):
    # Make copy of image
    image = image.copy()
    
    contours, hierarchy = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    contour_sizes = [(cv2.contourArea(contour), contour) for contour in contours]
    biggest_contour = max(contour_sizes, key = lambda x : x[0])[1]
    
    # Return the biggest contour
    mask = np.zeros(image.shape, np.uint8)
    cv2.drawContours(mask, [biggest_contour], -1, 255, -1)
    
    return biggest_contour, mask
    
def circled_contour(image, contour):
    # Get bounding ellipse
    image_with_ellipse = image.copy()
    ellipse = cv2.fitEllipse(contour)
    
    # Add it
    cv2.ellipse(image_with_ellipse, ellipse, green, 2, cv2.LINE_AA)
    return image_with_ellipse

def detect_object(image):
    # Convert to RGB color scheme
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Resize image
    max_dim = max(image.shape)
    scale = 700/max_dim
    image = cv2.resize(image, None, fx=scale, fy=scale)
    
    # Blur image
    image_blur = cv2.GaussianBlur(image, (5, 5), 0)
    image_blur_hsv = cv2.cvtColor(image_blur, cv2.COLOR_RGB2HSV)
    cv2.imwrite('img_blur_hsv.jpg', image_blur_hsv)
    # Filters
    min_green1 = np.array([30, 150, 130])
    max_green1 = np.array([45, 256, 256])

    mask1 = cv2.inRange(image_blur_hsv, min_green1, max_green1)
    cv2.imwrite('mask1.jpg', mask1)
    
    min_green2 = np.array([0, 100, 80])
    max_green2 = np.array([180, 256, 240])
    
    mask2 = cv2.inRange(image_blur_hsv, min_green2, max_green2)
    cv2.imwrite('mask2.jpg', mask2)
    # Combine masks
    mask = mask1 + mask2
    cv2.imwrite('mask.jpg', mask)
    
    # Segment
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (15, 15))
    mask_closed = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    mask_clean = cv2.morphologyEx(mask_closed, cv2.MORPH_OPEN, kernel)
    
    # Find biggest apple
    big_apple_contour, mask_apple = find_biggest_contour(mask_clean)
    
    # Overlay
    overlay = overlay_mask(mask_clean, image)
    
    # Circle biggest
    circled = circled_contour(overlay, big_apple_contour)
    show(circled)
    
    
    # Convert color scheme back to RGB and return image
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image

img = cv2.imread('apple.jpg')
after_img = detect_object(img)
cv2.imwrite('new_apple.jpg', after_img)


# In[ ]:




