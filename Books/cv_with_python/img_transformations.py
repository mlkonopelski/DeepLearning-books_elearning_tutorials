import argparse
import cv2


#------------------------------------------------------------------------
#   Read image as numpy array
#------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--path', default='data/Ex1.png', help='Image path')
params = parser.parse_args()

img = cv2.imread(params.path)

#------------------------------------------------------------------------
#   Check if Image was succsfully loaded
#------------------------------------------------------------------------
assert img is not None
print(f'Image path: {params.path}')
print(f'shape: {img.shape}')
print(f'Type: {img.dtype}')

#------------------------------------------------------------------------
#   Load a image as B&W image even if colour originally
#------------------------------------------------------------------------
img = cv2.imread(params.path, cv2.IMREAD_GRAYSCALE)
print(f'shape: {img.shape}')
print(f'Type: {img.dtype}')

#------------------------------------------------------------------------
#   Resize image
#------------------------------------------------------------------------
target_w, target_h = 128, 256
img_resized = cv2.resize(src=img, dsize=(target_w, target_h))
print(f'Resized shape: {img_resized.shape}')
cv2.imwrite(filename='data/Ex1_resized_fixed.png', img=img_resized)

target_resize = .2
img_resized = cv2.resize(img, (0, 0), img, target_resize, target_resize)
print(f'Resized shape: {img_resized.shape}')
cv2.imwrite(filename='data/Ex1_resized_20p.png', img=img_resized)

target_resize = .2
img_resized = cv2.resize(img, (0, 0), img, target_resize, target_resize, cv2.INTER_NEAREST) # use NN interpolation
print(f'Resized shape: {img_resized.shape}')
cv2.imwrite(filename='data/Ex1_resized_20p_internn.png', img=img_resized)


#------------------------------------------------------------------------
#   IMG flip
#------------------------------------------------------------------------
img_flipped = cv2.flip(src=img, flipCode=0)
cv2.imwrite(filename='data/Ex1_fliped_h.png', img=img_flipped, params=[cv2.IMWRITE_PNG_COMPRESSION, 0]) # [0, 10] - bigger number lowe quality

img_flipped = cv2.flip(src=img, flipCode=1)
cv2.imwrite(filename='data/Ex1_fliped_w.jpg', img=img_flipped, params=[cv2.IMWRITE_JPEG_QUALITY, 0]) # [0, 100] - bigger number better quality

img_flipped = cv2.flip(src=img, flipCode=-1)
cv2.imwrite(filename='data/Ex1_fliped_both.jpg', img=img_flipped)
cv2.imshow(winname='Ex1_fliped_both', mat=img_flipped)
cv2.waitKey(20000)

