import jittor as jt
import numpy as np

from jnerf.runner import Runner
from jnerf.utils.config import init_cfg


class NerfRunner:
    def __init__(self, model_file_path):
        """
        类初始化
        :param model_file_path: 加载的模型文件位置
        """
        self.model_file_path = model_file_path
        init_cfg('./config.py')

        self.runner = Runner()
        self.runner.ckpt_path = model_file_path
        self.runner.load_ckpt(model_file_path)

    def inference(self, pos, dir, euler_mode='XYZ'):
        """
        对外接口，用来调用模型推理
        :param pos: 场景相机的坐标 [x, y, z]
        :param dir: 场景相机的方向 用欧拉角表示 单位是度 [x_angle, y_angle, z_angle]
        :param euler_mode: 欧拉角模式，指定绕三个轴旋转的顺序
        :return: 推理出的图片 维度 (H, W, 3) 数据为np.uint8类型 RGB通道
        """
        print("inferencing...")
        pose = self.get_pose_from_pos_dir(pos, dir, mode=euler_mode)
        print(pose)
        img = self.runner.render_img_with_pose(pose)
        img = (img * 255 + 0.5).clip(0, 255).astype('uint8')
        return img

    def get_pose_from_pos_dir(self, pos, dir, mode='XYZ'):
        assert mode in ['XYZ', 'XZY', 'YXZ', 'YZX', 'ZXY', 'ZYX'], "欧拉角模式不合法"

        euler_x = dir[0] * np.pi / 180
        euler_y = dir[1] * np.pi / 180
        euler_z = dir[2] * np.pi / 180

        rotations = {
            'X': np.array([
                [1, 0, 0],
                [0, np.cos(euler_x), -np.sin(euler_x)],
                [0, np.sin(euler_x), np.cos(euler_x)]
            ]).astype(np.float32),
            'Y': np.array([
                [np.cos(euler_y), 0, np.sin(euler_y)],
                [0, 1, 0],
                [-np.sin(euler_y), 0, np.cos(euler_y)]
            ]).astype(np.float32),
            'Z': np.array([
                [np.cos(euler_z), -np.sin(euler_z), 0],
                [np.sin(euler_z), np.cos(euler_z), 0],
                [0, 0, 1]
            ]).astype(np.float32)
        }

        rotation = rotations[mode[2]] @ rotations[mode[1]] @ rotations[mode[0]]
        pose = np.array([
            [0, 0, 0, pos[0]],
            [0, 0, 0, pos[1]],
            [0, 0, 0, pos[2]],
        ]).astype(np.float32)

        pose[:3, :3] = rotation
        pose = jt.array(pose)

        return pose

    def pose_spherical(self, theta, phi, radius):
        trans_t = lambda t: jt.array(np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0],
            [0, 0, 1, t],
            [0, 0, 0, 1]]).astype(np.float32))
        rot_phi = lambda phi: jt.array(np.array([
            [1, 0, 0, 0],
            [0, np.cos(phi), -np.sin(phi), 0],
            [0, np.sin(phi), np.cos(phi), 0],
            [0, 0, 0, 1]]).astype(np.float32))
        rot_theta = lambda th: jt.array(np.array([
            [np.cos(th), 0, -np.sin(th), 0],
            [0, 1, 0, 0],
            [np.sin(th), 0, np.cos(th), 0],
            [0, 0, 0, 1]]).astype(np.float32))
        c2w = trans_t(radius)
        c2w = rot_phi(phi / 180. * np.pi) @ c2w
        c2w = rot_theta(theta / 180. * np.pi) @ c2w
        c2w = jt.array(np.array([[-1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])) @ c2w
        c2w = c2w[:-1, :]
        return c2w
