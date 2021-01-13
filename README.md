# image_processinglab
import cv2
image=cv2.imread('ip images1.png')
grey_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) 
cv2.imwrite('ip images1.png',image)
cv2.imshow("frame1",image)
cv2.imshow("frame2",grey_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

![image](https://user-images.githubusercontent.com/72559755/104423701-6dcbc380-55a4-11eb-921a-714a1cfabcda.png)
