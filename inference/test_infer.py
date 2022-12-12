import inference
import numpy as np
import cv2
import matplotlib
from matplotlib import pyplot as plt
from math import *
from scipy.spatial.transform import Rotation

x, y, z = 0, 5, 0.5
eye_angle = [180, 90]

unit_vec = [1, 0, 0]
base_r = Rotation.from_euler("YXZ", eye_angle+[0], degrees=True).as_matrix()
base_vec = np.matmul(unit_vec, base_r)  # 初始法向量
del base_r, unit_vec


def kbd_press(event):
    global x, y, z, eye_angle
    r = Rotation.from_euler(
        "ZXY", [eye_angle[0]-180, 0, 0], degrees=True).as_matrix()
    up_r = Rotation.from_euler(
        "YXZ", [eye_angle[1]-90, 0, 0], degrees=True).as_matrix()
    vec = np.matmul(base_vec, r)
    up = np.matmul(base_vec, up_r)
    if (event.key == 'a'):
        x -= vec[0]
        y += vec[1]
    if (event.key == 'd'):
        x += vec[0]
        y -= vec[1]
    if (event.key == 'w'):
        x += vec[1]
        y += vec[0]
    if (event.key == 's'):
        x -= vec[1]
        y -= vec[0]
    if (event.key == ' '):
        x -= up[1]
        y -= up[2]
        z -= up[0]
    if (event.key == 'control'):
        x += up[1]
        y += up[2]
        z += up[0]
    if (event.key == 'q'):
        exit(0)
    if (event.key == 'right'):
        eye_angle[0] -= 10
    if (event.key == 'left'):
        eye_angle[0] += 10
    if (event.key == 'up'):
        eye_angle[1] += 10
    if (event.key == 'down'):
        eye_angle[1] -= 10


nr = inference.NerfRunner("../models/lego_4/params.pkl")
matplotlib.use("TkAgg")
plt.rcParams['keymap.save'] = ''
fig = plt.figure()
fig.canvas.mpl_connect("key_press_event", kbd_press)

while (True):
    img = nr.inference([x, y, z], [eye_angle[1], eye_angle[0], 0], "YXZ")
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.show(block=False)
    print("pos:", [x, y, z], "dir", eye_angle)
    plt.waitforbuttonpress(10)
