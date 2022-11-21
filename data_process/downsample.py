import configargparse
import cv2
# import def_Gaussian as dg
# import time
import os.path
import sys


# import glob

#####################################################################################################################
# 读取文件夹里面的图像数量 并返回filenum
def countFile(dir):
    # 输入文件夹
    tmp = 0
    for item in os.listdir(dir):
        if os.path.isfile(os.path.join(dir, item)):
            tmp += 1
        else:
            tmp += countFile(os.path.join(dir, item))
    return tmp

def config_parser():
    parser = configargparse.ArgumentParser()

    # load keypose dataset flag
    parser.add_argument("--src_dir", type=str, required=True,
                        help='first time running, load keypose data into several dirs')

    parser.add_argument("--des_dir", type=str, required=True,
                        help='input src data directory')

    parser.add_argument("--scale", type=float, required=True,
                        help='output data directory')

    return parser

def downsample_dir(src_dir, des_dir, scale):
    if not os.path.exists(des_dir):
        os.mkdir(des_dir)

    for ori_file in os.listdir(src_dir):
        src_file_path = os.path.join(src_dir, ori_file)
        print(src_file_path)
        if os.path.isdir(src_file_path):
            recur_src_dir = os.path.join(src_file_path)
            des_file_path = os.path.join(des_dir, ori_file)
            recur_des_dir = os.path.join(des_file_path)
            print("directory " + recur_src_dir + " " + recur_des_dir)
            downsample_dir(recur_src_dir, recur_des_dir, scale)

        elif ori_file.split(".")[-1] == "png":
            original_image = cv2.imread(src_file_path)
            origin_shape = original_image.shape
            new_shape = (int(origin_shape[0] / scale), int(origin_shape[1] / scale))
            new_image = cv2.resize(original_image, new_shape, interpolation=cv2.INTER_AREA)
            print(str(origin_shape) + "to" + str(new_shape))

            new_file_path = os.path.join(des_dir, ori_file)

            cv2.imwrite(new_file_path, new_image)





if __name__ == '__main__':
    parser = config_parser()
    args = parser.parse_args()

    src_dir = args.src_dir
    des_dir = args.des_dir
    scale = args.scale

    filenum = countFile(src_dir)  # 返回的是图片的张数
    print(filenum)

    downsample_dir(src_dir, des_dir, scale)


