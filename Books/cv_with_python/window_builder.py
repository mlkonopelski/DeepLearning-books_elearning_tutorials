import cv2
import numpy as np


#------------------------------------------------------------------------
#   Build SLiders
#------------------------------------------------------------------------

# cv2.namedWindow('tracker')

# fill_value = np.array([255, 255, 255], np.uint8)

# def trackbar_callback(idx, value):
#     fill_value[idx] = value
    
# cv2.createTrackbar('R', 'tracker', 255, 255, lambda v: trackbar_callback(2, v))
# cv2.createTrackbar('G', 'tracker', 255, 255, lambda v: trackbar_callback(1, v))
# cv2.createTrackbar('B', 'tracker', 255, 255, lambda v: trackbar_callback(0, v))

# while True:
#     image = np.full((500, 500, 3), fill_value)
#     cv2.imshow('tracker', image)
#     key = cv2.waitKey(3)
#     if key == 27:
#         break

# cv2.destroyAllWindows()

#------------------------------------------------------------------------
#   Draw Lines and Circles
#------------------------------------------------------------------------
import random 

img = cv2.imread(filename='data/Ex1.png')
w, h = img.shape[0], img.shape[1]

def rand_pt(mult=1.):
    return (random.randrange(int(w*mult)),
            random.randrange(int(h*mult)))

cv2.circle(img, center=rand_pt(), radius=40, color=(255, 0, 0))
cv2.circle(img, center=rand_pt(), radius=5, color=(255, 0, 0), thickness=cv2.FILLED)
cv2.circle(img, center=rand_pt(), radius=40, color=(255, 85, 85), thickness=10)
cv2.circle(img, center=rand_pt(), radius=40, color=(255, 170, 170), thickness=10, lineType=cv2.LINE_AA)

cv2.line(img, pt1=rand_pt(), pt2=rand_pt(), color=(0, 255, 0))
cv2.line(img, pt1=rand_pt(), pt2=rand_pt(), color=(85, 255, 85), thickness=10)
cv2.line(img, pt1=rand_pt(), pt2=rand_pt(), color=(170, 255, 170), thickness=4, lineType=cv2.LINE_AA)

x1, y2 = rand_pt(), rand_pt()
cv2.rectangle(img,
              pt1=x1, 
              pt2=y2, 
              color=(255, 255, 255), 
              thickness=10)
cv2.putText(img, 
            text='Rectange', 
            org=x1, 
            fontFace=cv2.FONT_HERSHEY_SIMPLEX, 
            fontScale=1, 
            color=(255, 255, 255), 
            thickness=10)

cv2.imwrite(filename='data/Ex1_draw1.png', img=img)


#------------------------------------------------------------------------
#   Draw Lines and Circles
#------------------------------------------------------------------------
