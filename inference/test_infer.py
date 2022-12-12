import inference
import numpy as np
import cv2 as cv
from math import *
from scipy.spatial.transform import Rotation

x, y, z = 0, -20, 5
eye_angle = [0, 90]

unit_vec = [-1, 0, 0]
base_r = Rotation.from_euler("YXZ", eye_angle + [0], degrees=True).as_matrix()
base_vec = np.matmul(unit_vec, base_r)  # 初始法向量
del base_r, unit_vec

nr = inference.NerfRunner("../models/cafe/params.pkl")

while (True):
    img = nr.inference([x, y, z], [eye_angle[1], eye_angle[0], 0], "YXZ")
    cv.imshow("demo", cv.resize(cv.cvtColor(img, cv.COLOR_BGR2RGB), (200, 200)))
    key = cv.waitKey(0)

    r = Rotation.from_euler(
        "ZXY", [eye_angle[0] - 180, 0, 0], degrees=True).as_matrix()
    up_r = Rotation.from_euler(
        "YXZ", [eye_angle[1] - 90, 0, 0], degrees=True).as_matrix()
    vec = np.matmul(base_vec, r)
    up = np.matmul(base_vec, up_r)
    if chr(key) == 'a':
        x -= vec[0]
        y += vec[1]
    if chr(key) == 'd':
        x += vec[0]
        y -= vec[1]
    if chr(key) == 'w':
        x += vec[1]
        y += vec[0]
    if chr(key) == 's':
        x -= vec[1]
        y -= vec[0]
    if chr(key) == ' ':
        x -= up[1]
        y -= up[2]
        z -= up[0]
    if chr(key) == 'c':
        x += up[1]
        y += up[2]
        z += up[0]
    if chr(key) == 'q':
        exit(0)
    if chr(key) == 'l':
        eye_angle[0] -= 10
    if chr(key) == 'j':
        eye_angle[0] += 10
    if chr(key) == 'i':
        eye_angle[1] += 10
    if chr(key) == 'k':
        eye_angle[1] -= 10
