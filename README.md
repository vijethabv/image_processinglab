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

program2:Develop a program to perform linear transformation on image(scaling and rotation).
linear transformation:is type of gray level transformation that is used for image enhancement
it is a spatial domain method.
it is used for manipulation of an image so that result is more suitable than original for a specific application.
scaling:Image scaling is the process of resizing a digital image.
rotation:Image rotation is a common image processing routine with applications in matching, alignment, and other image-based algorithms. 

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
output:
![image](https://user-images.githubusercontent.com/72559755/104427130-e765b080-55a8-11eb-9bdd-476721a1cf40.png)

import cv2 
import numpy as np 
  
FILE_NAME = 'ip images1.png'
img = cv2.imread(FILE_NAME) 
(rows, cols) = img.shape[:2] 
M = cv2.getRotationMatrix2D((cols / 2, rows / 2), 45, 1) 
res = cv2.warpAffine(img, M, (cols, rows)) 
cv2.imshow('result.jpg', res) 
cv2.waitKey(0)
output:
![image](https://user-images.githubusercontent.com/72559755/104427563-635ff880-55a9-11eb-9393-b2eb2d058c7c.png)

program3:Develop a program to find sum and mean of set of images.
Create n number of images and read from directory and perform the operation.
mean:mean value gives the contribution of individual pixel intensity for the entire image.
sum:adds the value of each pixel in one of the input images with the corresponding pixel 
in the other input image and returns the sum in the corresponding pixel of the output image.

import cv2
import os
path = "D:\imp_for_ip"
imgs=[]
dirs=os.listdir(path)

for file in dirs:
    fpat=path+"\\"+file
    imgs.append(cv2.imread(fpat))
    
i=0
sum_img=[]
for sum_img in imgs:
    read_imgs=imgs[i]
    sum_img=sum_img+read_imgs
    #cv2.imshow(dirs[i],imgs[i])
    i=i+1
print(i)
cv2.imshow('sum',sum_img)
print(sum_img)

cv2.imshow('mean',sum_img/i)
mean=(sum_img/i)
print(mean)

cv2.waitKey()
cv2.destroyAllwindows()
output:
![image](https://user-images.githubusercontent.com/72559755/104432663-3dd5ed80-55af-11eb-9d17-9acd29b68ce7.png)

4.Convert color image into gray and binary image.
import cv2
img = cv2.imread("nature1.png")
grey=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
cv2.imshow("Binary Image",grey)
cv2.waitKey(0)
cv2.destroyAllWindows()
ret, bw_img = cv2.threshold(img,127,255,cv2.THRESH_BINARY)
cv2.imshow("Binary Image",bw_img)
cv2.waitKey(0)
cv2.destroyAllWindows()



