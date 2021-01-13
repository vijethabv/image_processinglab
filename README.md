# image_processinglab
program1:Develop a program to display grayscale image using read and write operation.
grayscale image:Grayscale is a range of monochromatic shades from black to white. 
Therefore, a grayscale image contains only shades of gray and no color.
to save image:cv2.imwrite()
to show image:cv2.imshow()
destroy all windows:cv2.destroyAllWindows()
import cv2
image=cv2.imread('ip images1.png')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imwrite('ip images1.png',image)
cv2.imshow("frame1",image)
cv2.imshow("frame2",grey_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
output:
![image](https://user-images.githubusercontent.com/72559755/104423701-6dcbc380-55a4-11eb-921a-714a1cfabcda.png)
![image](https://user-images.githubusercontent.com/72559755/104424001-c9964c80-55a4-11eb-9310-b1069ad1f8e8.png)

2.Develop a program to perform linear transformation on image(scaling and rotation).
import cv2
import numpy as np
FILE_NAME = 'ip images1.png'
try: 
    img = cv2.imread(FILE_NAME) 
   (height, width) = img.shape[:2] 
    res = cv2.resize(img, (int(width / 2), int(height / 2)), interpolation = cv2.INTER_CUBIC) 
    cv2.imwrite('result.jpg', res) 
    cv2.imshow('image',img)
    cv2.imshow('result',res)
    cv2.waitKey(0)
  
except IOError: 
    print ('Error while reading files !!!')
    cv2.waitKey(0)
    cv2.destroyAllWindows(0)
