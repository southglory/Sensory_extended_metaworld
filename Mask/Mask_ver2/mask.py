import cv2 as cv
import numpy as np

tags = ['5','6','7','8']
for tag in tags:
    img = cv.imread(f'Mask/Mask_ver2/input_image/{tag}.png', cv.IMREAD_COLOR)
    img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    # extract bgr channels
    rgb = img[:,:,0:3]

    # select grayscale range
    mask = cv.inRange(rgb, (30,0,0), (255,20,20)) # select only red

    # save output
    cv.imwrite(f'Mask/Mask_ver2/mask_image/{tag}.png', mask)