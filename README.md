# OpenCV-Object-Detection
Detect object in an image using OpenCV Python.  
As example we will use a picture of a green apple:  
  
Initial picture on the left. Processed picture on the right. Green ellipse indicates borders of an apple.  

![alt text](https://i.imgur.com/G097UvJ.jpg)
![alt text](https://i.imgur.com/0wfR6pt.jpg)

## Working with the image  
### First step: Preprocessing  
Firstly, we need to convert image from BGR color scheme to RGB. We do it using **cv2.cvtColor()** function  
Resize image to 700x700 to have standardized representation of image  
To remove noise and redundant details from image we will use **cv2.GaussianBlur()** function with kernel size 5x5 and zero standard deviation  
After that we will convert image to HSV color scheme. At this point image will look like this:  
![alt text](https://i.ibb.co/fStzj4m/img-blur-hsv.jpg)  

### Second step: Applying masks  
We need to distinguish apple from the rest of the image, so that we will use masks.  
We will filter image by color and then by brightness using **cv2.inRange()** function, after that masks will be combined:  
![alt text](https://i.ibb.co/YjkfqBn/mask.jpg)  
### Third step: Perform morphological operations and find apple contour in the image  
Create ellipse shaped kernel of size 15x15 using **cv2.getStructuringElement()** function.  
Perform closing and then opening operations using **cv2.morphologyEx()** with kernel specified above to remove noise and to close small holes inside foreground object.  
![alt text](https://i.ibb.co/rcXmfT5/mask-clean.jpg)

After having preprocessed mask find contours with **cv2.findContours()** and then detect the biggest one.  
As we found contours, we can combine contour, mask and original image.  
Finally, change color scheme back from RGB to BGR and display final image:  
![alt text](https://i.imgur.com/0wfR6pt.jpg)
