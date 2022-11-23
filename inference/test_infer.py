import numpy as np

import inference
import cv2 as cv

nr = inference.NerfRunner("../models/lego_4/params.pkl")
img = nr.inference([0, -3.5, 3.5], [45, 0, 0])
# for i in range(1, 11):
#     print(i)
#     img = nr.inference(1, 1)

# cv.imshow("inference result", img)
# cv.waitKey(0)
img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
cv.imwrite("./temp/test.png", img)
print("rendered image is saved.")
