# -*- coding: utf-8 -*-
# @Time    : 2020/11/18:上午9:03
# @Author  : jingtao sun
# 生成 syn_data/train中的3D_BBOX标签信息
import os
import glob
import cv2
import numpy as np
import time
import datetime


def load_depth(depth_path):
    """
    读取深度图
    """
    # 读取八位深度，原通道的图片
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
        # NOTE: RGB is actually BGR in opencv
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'

    return depth16


def get_pose_matrix(pose_path, inst_id):
    """
    获取当前的位姿的齐次变换矩阵

    """
    pose = []
    with open(pose_path, 'r') as posef:
        for input_line in posef:
            input_line = input_line.split(' ')
            if len(input_line) == 1:
                if int(input_line[0]) == inst_id:
                    for i in range(4):
                        input_line = posef.readline()
                        if input_line[-1:] == '\n':
                            input_line = input_line[:-1]
                        input_line = input_line.split(' ')
                        pose.append(
                            [float(input_line[0]), float(input_line[1]),
                             float(input_line[2]), float(input_line[3])])
                    break
    pose = np.array(pose)
    return pose


def write_scale(scale, bbox_f):
    """
    保存3D_BBOX的八个角点坐标

    """
    bbox_f.write(str(scale[2]) + " " + str(scale[1]) + " " + str(scale[0]) + "\n")
    bbox_f.write(str(scale[2]) + " " + str(scale[1]) + " " + str(-scale[0]) + "\n")
    bbox_f.write(str(scale[2]) + " " + str(-scale[1]) + " " + str(scale[0]) + "\n")
    bbox_f.write(str(scale[2]) + " " + str(-scale[1]) + " " + str(-scale[0]) + "\n")
    bbox_f.write(str(-scale[2]) + " " + str(scale[1]) + " " + str(scale[0]) + "\n")
    bbox_f.write(str(-scale[2]) + " " + str(scale[1]) + " " + str(-scale[0]) + "\n")
    bbox_f.write(str(-scale[2]) + " " + str(-scale[1]) + " " + str(scale[0]) + "\n")
    bbox_f.write(str(-scale[2]) + " " + str(-scale[1]) + " " + str(-scale[0]) + "\n")

class SynDataBbox(object):
    def __init__(self,dataset_root):
        self.root = dataset_root
        self.syn_train_dir = os.path.join(self.root, 'syn_data','train')
        self.model_scale_dir = os.path.join(self.root,'model_scales')
        self.gts_dir = os.path.join(self.root, 'gts', 'train')
        self.syn_cam_intrinsics = np.array([[577.5, 0, 319.5],
                                            [0., 577.5, 239.5],
                                            [0., 0., 1.]])
        pass

    def back_project(self, depth, instance_mask):
        """
        将每一个实例物体的图像点坐标映射到三维

        """
        intrinsics_inv = np.linalg.inv(self.syn_cam_intrinsics)
        non_zero_mask = (depth > 0)
        final_instance_mask = np.logical_and(instance_mask, non_zero_mask)
        idxs = np.where(final_instance_mask)
        grid = np.array([idxs[1], idxs[0]])
        length = grid.shape[1]
        ones = np.ones([1, length])
        uv_grid = np.concatenate((grid, ones), axis=0)  # [3, num_pixel]
        xyz = intrinsics_inv @ uv_grid  # [3, num_pixel]
        xyz = np.transpose(xyz)  # [num_pixel, 3]
        z = depth[idxs[0], idxs[1]]
        pts = xyz * z[:, np.newaxis] / xyz[:, -1:]
        pts[:, 0] = -pts[:, 0]
        pts[:, 1] = -pts[:, 1]
        return pts, idxs

    def compute_cur_scale(self, pose, pts, model_id):
        """
        计算当前的物体的bbox在第一象限的角点坐标
        """
        translation = pose[:3, 3]
        rotation = pose[:3, :3]
        pts = pts - translation
        pts = pts @ rotation
        model_path = os.path.join(self.model_scale_dir, model_id) + ".txt"
        bbox = np.loadtxt(model_path)
        scale = (bbox[1] - bbox[0]) / 2

        minx = np.min(pts[:, 0])
        miny = np.min(pts[:, 1])
        minz = np.min(pts[:, 2])
        maxx = np.max(pts[:, 0])
        maxy = np.max(pts[:, 1])
        maxz = np.max(pts[:, 2])

        x = max(maxx, abs(minx))
        y = max(maxy, abs(miny))
        z = max(maxz, abs(minz))

        x_ratio = x / scale[2]
        y_ratio = y / scale[1]
        z_ratio = z / scale[0]

        ratio = max(x_ratio, y_ratio, z_ratio)
        scale = ratio * scale
        return scale

    def get_and_write_3Dbbox(self, depth_path, meta_path, bbox_path, pose_path, inst_ids, mask_im):
        """
        获取每一帧的bbox
        """
        with open(meta_path, 'r') as f:
            with open(bbox_path, 'w') as bbox_f:
                for line in f:
                    line_info = line.split(' ')
                    # 当前的 instance id
                    inst_id = int(line_info[0])
                    # 当前 instance对应的类别ID
                    cls_id = int(line_info[1])
                    model_id = str(line_info[-1]).replace("\n", "")
                    if not inst_id in inst_ids:
                        continue
                    bbox_f.write(str(inst_id) + "\n")
                    if cls_id == 0:
                        for i in range(8):
                            bbox_f.write("0 0 0\n")
                        continue
                    depth_map = load_depth(depth_path)
                    tmp_mask = (mask_im == inst_id)
                    pts, idxs = self.back_project(depth_map, tmp_mask)
                    pose = get_pose_matrix(pose_path, inst_id)
                    scale = self.compute_cur_scale(pose, pts, model_id)
                    write_scale(scale, bbox_f)

    def get_path(self, full_path):
        """
        得到所有相关的数据保存路径
        """
        image_path = full_path.replace("_color.png", "")
        mask_path = image_path + '_mask.png'
        if not os.path.exists(mask_path):
            raise EOFError("path error!")
        depth_path = image_path + '_depth.png'
        meta_path = image_path + '_meta.txt'
        bbox_path = (image_path + '_bbox.txt').replace(self.syn_train_dir, self.gts_dir)
        pose_path = (image_path + '_pose.txt').replace(self.syn_train_dir,self.gts_dir)
        return depth_path, meta_path, bbox_path, pose_path, mask_path


    def run(self):
        """
        入口函数
        """
        folder_list = [name for name in os.listdir(self.syn_train_dir) if os.path.isdir(os.path.join(self.syn_train_dir, name))]
        begin_time = time.time()
        for folder in folder_list:
            image_list = glob.glob(os.path.join(self.syn_train_dir, folder, '*_color.png'))
            image_list = sorted(image_list)
            for image_full_path in image_list:
                depth_path, meta_path, bbox_path, pose_path, mask_path = self.get_path(image_full_path)
                mask_im = cv2.imread(mask_path)[:, :, 2]
                mask_im = np.array(mask_im)
                # 实例对象的label列表
                inst_ids = np.unique(mask_im)
                self.get_and_write_3Dbbox(depth_path, meta_path, bbox_path, pose_path, inst_ids, mask_im)
        end_time = time.time()
        print("spent time: %s " % (datetime.timedelta(seconds=(end_time - begin_time))))

if __name__ == '__main__':
    SynDataBbox('/media/sunjingtao/data_HD_1/NOCS').run()
