import inference
import numpy as np
import cv2
from math import *
from scipy.spatial.transform import Rotation

def calcRotationMatrix(theta, axis):
    c = cos(theta / 360 * pi) # theta/2 (degree) ---> radian   
    s = sin(theta / 360 * pi)
    quaternion = [i * s for i in axis] + [c]
    rot = Rotation.from_quat(quaternion)
    return rot.as_matrix()


base_up = [0, 1, 0]
base_front = [0, 0, -1]
rotation = Rotation.from_euler("XYZ", [90, 0, 0], True).as_matrix()

pos = [0, -20, 5]

nr = inference.NerfRunner("../models/cafe/params.pkl")

while (True):
    up = np.matmul(rotation, base_up)
    front = np.matmul(rotation, base_front)
    right = np.cross(front, up)
    img = nr.inference(pos, Rotation.from_matrix(rotation).as_euler("XZY", degrees=True), "XYZ")
    cv2.imshow("demo", cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), (800, 800)))
    key = cv2.waitKey(0)

    match chr(key):
        case 'a':
            pos -= right
        case 'd':
            pos += right
        case 'w':
            pos += front
        case 's':
            pos -= front
        case ' ':
            pos += up
        case 'c':
            pos -= up

        case 'l':
            rot = calcRotationMatrix(-5, up)
            rotation = np.matmul(rot, rotation)
        case 'j':
            rot = calcRotationMatrix(5, up)
            rotation = np.matmul(rot, rotation)
        case 'i':
            rot = calcRotationMatrix(5, right)
            rotation = np.matmul(rot, rotation)
        case 'k':
            rot = calcRotationMatrix(-5, right)
            rotation = np.matmul(rot, rotation)

        case 'q':
            exit(0)