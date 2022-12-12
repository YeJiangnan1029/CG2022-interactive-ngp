import inference
import numpy as np
import cv2
from math import *
from scipy.spatial.transform import Rotation

x, y, z = 0, 5, 0.5
eye_angle = [180, 90]

unit_vec = [1, 0, 0]
base_r = Rotation.from_euler("YXZ", eye_angle+[0], degrees=True).as_matrix()
base_vec = np.matmul(unit_vec, base_r)  # 初始法向量
del base_r, unit_vec

nr = inference.NerfRunner("../models/lego_4/params.pkl")

while (True):
    img = nr.inference([x, y, z], [eye_angle[1], eye_angle[0], 0], "YXZ")
    cv2.imshow("demo",cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB),(800,800)))
    key = cv2.waitKey(0)

    r = Rotation.from_euler(
        "ZXY", [eye_angle[0]-180, 0, 0], degrees=True).as_matrix()
    up_r = Rotation.from_euler(
        "YXZ", [eye_angle[1]-90, 0, 0], degrees=True).as_matrix()
    vec = np.matmul(base_vec, r)
    up = np.matmul(base_vec, up_r)
    if (chr(key) == 'a'):
        x -= vec[0]
        y += vec[1]
    if (chr(key) == 'd'):
        x += vec[0]
        y -= vec[1]
    if (chr(key) == 'w'):
        x += vec[1]
        y += vec[0]
    if (chr(key) == 's'):
        x -= vec[1]
        y -= vec[0]
    if (chr(key) == ' '):
        x -= up[1]
        y -= up[2]
        z -= up[0]
    if (chr(key) == 'c'):
        x += up[1]
        y += up[2]
        z += up[0]
    if (chr(key) == 'q'):
        exit(0)
    if (chr(key) == 'l'):
        eye_angle[0] -= 10
    if (chr(key) == 'j'):
        eye_angle[0] += 10
    if (chr(key) == 'i'):
        eye_angle[1] += 10
    if (chr(key) == 'k'):
        eye_angle[1] -= 10