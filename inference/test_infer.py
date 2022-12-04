import numpy as np

import inference
import cv2 as cv
from matplotlib import pyplot as plt

nr = inference.NerfRunner("../models/lego_4/params.pkl")
x, y, z = 0, -3.5, 3.5
while (True):
    img = nr.inference([x, y, z], [45, 0, 0])
    cv.imshow("", img)
    key = cv.waitKey(0)
    if (chr(key) == 'a'):
        x -= 0.5
    elif (chr(key) == 'd'):
        x += 0.5
    elif (chr(key) == 'w'):
        y += 0.5
    elif (chr(key) == 's'):
        y -= 0.5
    elif (chr(key) == 'z'):
        z -= 0.5
    elif (chr(key) == 'c'):
        z += 0.5
    elif (chr(key) == 'q'):
        break
    #img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #cv.imwrite("./temp/test.png", img)
    #print("rendered image is saved.")
