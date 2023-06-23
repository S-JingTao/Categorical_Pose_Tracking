# -*- coding: utf-8 -*-
# @Time    : 2020/11/15:下午3:37
# @Author  : jingtao sun


import os
import random
import copy
import cv2
import pptk
import torch
import numpy as np

import torch.utils.data as data
import _pickle as cPickle
import torchvision.transforms as transforms
from PIL import Image
from mpl_toolkits.mplot3d import Axes3D

import matplotlib.pyplot as plt
from generate_syn_data_3dbbox import load_depth


def random_pick(some_list, probabilities):
    x = random.uniform(0, 1)
    cumulative_probability = 0.0
    for item, item_probability in zip(some_list, probabilities):
        cumulative_probability += item_probability
        if x < cumulative_probability:
            break
    return item


def plot_3D_point_cloud(cloud, fig):
    cloud = cloud.tolist()
    x = []
    y = []
    z = []
    for point in cloud:
        x.append(point[0])
        y.append(point[1])
        z.append(point[2])
    fig = plt.figure(dpi=120)
    ax = fig.add_subplot(111, projection='3d')
    plt.title('point cloud')
    ax.scatter(x, y, z, c='r', marker='.', s=2, linewidth=0, alpha=1, cmap='spectral')

    ax.set_xlabel('X Label')
    ax.set_ylabel('Y Label')
    ax.set_zlabel('Z Label')
    # plt.show()


def get_2dbbox(cloud, cam_intrinsics, cam_scale):
    """
    根据三维边框来确定2d边框
    """
    rmin = 10000
    rmax = -10000
    cmin = 10000
    cmax = -10000
    border_list = [-1, 80, 120, 160, 200, 240, 280, 320, 360,
                   400, 440, 480, 520, 560, 600, 640, 680]
    img_width = 480
    img_length = 640
    cam_cx = cam_intrinsics['cx']
    cam_cy = cam_intrinsics['cy']
    cam_fx = cam_intrinsics['fx']
    cam_fy = cam_intrinsics['fy']
    for tg in cloud:
        p1 = int(tg[0] * cam_fx / tg[2] + cam_cx)
        p0 = int(tg[1] * cam_fy / tg[2] + cam_cy)
        if p0 < rmin:
            rmin = p0
        if p0 > rmax:
            rmax = p0
        if p1 < cmin:
            cmin = p1
        if p1 > cmax:
            cmax = p1
    rmax += 1
    cmax += 1
    if rmin < 0:
        rmin = 0
    if cmin < 0:
        cmin = 0
    if rmax >= 480:
        rmax = 479
    if cmax >= 640:
        cmax = 639

    r_b = rmax - rmin
    for tt in range(len(border_list)):
        if r_b > border_list[tt] and r_b < border_list[tt + 1]:
            r_b = border_list[tt + 1]
            break
    c_b = cmax - cmin
    for tt in range(len(border_list)):
        if c_b > border_list[tt] and c_b < border_list[tt + 1]:
            c_b = border_list[tt + 1]
            break
    center = [int((rmin + rmax) / 2), int((cmin + cmax) / 2)]
    rmin = center[0] - int(r_b / 2)
    rmax = center[0] + int(r_b / 2)
    cmin = center[1] - int(c_b / 2)
    cmax = center[1] + int(c_b / 2)

    if rmin < 0:
        delt = -rmin
        rmin = 0
        rmax += delt
    if cmin < 0:
        delt = -cmin
        cmin = 0
        cmax += delt
    if rmax > img_width:
        delt = rmax - img_width
        rmax = img_width
        rmin -= delt
    if cmax > img_length:
        delt = cmax - img_length
        cmax = img_length
        cmin -= delt

    if ((rmax - rmin) in border_list) and ((cmax - cmin) in border_list):
        return rmin, rmax, cmin, cmax
    else:
        return 0


def hit_cur_class_result_by_meta(object, frame_path, pose_dict):
    with open('{0}_meta.txt'.format(frame_path), 'r') as file:
        while True:
            each_line = file.readline()
            if not each_line:
                break
            if each_line[-1:] == '\n':
                each_line = each_line[:-1].split(' ')
            if each_line[-1] == object:
                target = pose_dict[int(each_line[0])]
                target_idx = int(each_line[0])

    target = np.array(target)
    target_r = target[:3, :3]
    target_t = target[:3, 3].flatten()
    return target_r, target_t, target_idx


def search_fit(points):
    # 返回整个3D边界框的八个坐标点的最值
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]


def enlarge_bbox(target):
    # 将3D_bbox按照一定的比例进行扩大

    limit = np.array(search_fit(target))
    longest = max(limit[1] - limit[0], limit[3] - limit[2], limit[5] - limit[4])
    longest = longest * 1.3

    scale1 = longest / (limit[1] - limit[0])
    scale2 = longest / (limit[3] - limit[2])
    scale3 = longest / (limit[5] - limit[4])

    target[:, 0] *= scale1
    target[:, 1] *= scale2
    target[:, 2] *= scale3

    return target


def get_cur_depth(frame_path):
    depth_path = '{0}_depth.png'.format(frame_path)
    cur_depth = load_depth(depth_path)
    cur_depth = np.array(cur_depth)
    return cur_depth


def get_target_mask(frame_path, cur_target_index, cur_cmax, cur_cmin, cur_rmax, cur_rmin):
    target_mask = (cv2.imread('{0}_mask.png'.format(frame_path))[:, :, 2] ==
                   cur_target_index)[cur_rmin:cur_rmax, cur_cmin:cur_cmax]

    target = (target_mask.flatten() != False).nonzero()[0]
    if len(target) == 0:
        return 0
    else:
        return target_mask


class LoadDataset(data.Dataset):

    def __init__(self, state, root_path, num_extracted_point, num_category, id_category, num_data):
        """
        初始化函数

        :param state: 是训练还是测试
        :param root_path: 数据集根目录
        :param num_extracted_point: 网络提取的基础点集数目
        :param num_category: 所有类别总数
        :param id_category: 当前的类别号
        """

        def _get_data_list(dir, num_cate, data_state, model):
            object_list = {}
            object_name_list = {}
            for id_cate in range(1, num_cate + 1):
                if data_state == 'syn':
                    cate_path = os.path.join(dir, 'data_list', 'syn_data', 'train', str(id_cate))
                else:
                    file_name = '{0}_{1}'.format(data_state, model)
                    cate_path = os.path.join(dir, 'data_list', 'real_data', file_name, str(id_cate))
                object_name_list[id_cate] = os.listdir(cate_path)
                object_list[id_cate] = {}
                for temp in object_name_list[id_cate]:
                    object_list[id_cate][temp] = []
                    with open(os.path.join(cate_path, temp, 'list.txt'), 'r') as file:
                        while True:
                            each_line = file.readline()
                            if not each_line:
                                break
                            if each_line[-1:] == '\n':
                                each_line = each_line[:-1]
                            object_list[id_cate][temp].append(each_line)
            return object_list, object_name_list

        def _get_coco_data_list(dir):
            data_list = []
            coco_root = os.path.join(dir, 'train2017')
            base_dir = os.path.abspath(os.path.dirname(__file__))
            with open(os.path.join(base_dir, 'train2017.txt'), 'r') as file:
                while True:
                    each_line = file.readline()
                    if not each_line:
                        break
                    if each_line[-1:] == '\n':
                        each_line = each_line[:-1]
                    data_list.append(os.path.join(coco_root, each_line))
            return data_list

        def _get_mesh():
            mesh = []
            base_dir = os.path.abspath(os.path.dirname(__file__))
            with open(os.path.join(base_dir, 'sphere.xyz'), 'r') as file:
                while True:
                    each_line = file.readline()
                    if not each_line:
                        break
                    if each_line[-1:] == '\n':
                        each_line = each_line[:-1]
                    each_line = each_line.split(' ')
                    mesh.append([float(each_line[0]), float(each_line[1]), float(each_line[2])])
            mesh = np.array(mesh) * 0.6
            return mesh

        self.state = state
        self.dir_path = root_path
        self.num_epoint = num_extracted_point
        self.num_cate = num_category
        self.id_cate = id_category
        self.num_data = num_data

        if self.state == 'train':
            self.syn_obj_list, self.syn_obj_name_list = _get_data_list(self.dir_path, self.num_cate, 'syn', self.state)
        self.real_obj_list, self.real_obj_name_list = _get_data_list(self.dir_path, self.num_cate, 'real', self.state)
        self.coco_data_list = _get_coco_data_list(self.dir_path)
        self.mesh = _get_mesh()

        self.cam_mat_real = {
            'cx': 322.52500, 'cy': 244.11084,
            'fx': 591.01250, 'fy': 590.16775,
        }
        self.cam_mat_syn = {
            'cx': 319.5, 'cy': 239.5,
            'fx': 577.5, 'fy': 577.5,
        }
        self.cam_scale = 1.0
        self.scale_size = 1000.0
        self.x_map = np.array([[j for i in range(640)] for j in range(480)])
        self.y_map = np.array([[i for i in range(640)] for j in range(480)])
        # 归一化函数
        self.norm = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    def get_syn_mesh_info(self, object):
        mesh_bbox = []
        with open('{0}/model_pts/{1}.txt'.format(self.dir_path, object), 'r') as file:
            for i in range(8):
                each_line = file.readline()
                if each_line[-1:] == '\n':
                    each_line = each_line[:-1].split(' ')
                mesh_bbox.append([float(each_line[0]), float(each_line[1]), float(each_line[2])])

        mesh_bbox = np.array(mesh_bbox)
        mesh_bbox = enlarge_bbox(copy.deepcopy(mesh_bbox))

        mesh_pts = []
        with open('{0}/model_pts/{1}.xyz'.format(self.dir_path, object), 'r') as file:
            # 元数据有3000组
            for i in range(2800):
                each_line = file.readline()
                if each_line[-1:] == '\n':
                    each_line = each_line[:-1]
                each_line = each_line.split(' ')
                mesh_pts.append([float(each_line[0]), float(each_line[1]), float(each_line[2])])
        mesh_pts = np.array(mesh_pts)
        return mesh_bbox, mesh_pts

    def get_cur_3Dbbox(self, data_type, frame_path, object, cur_index):
        if data_type == 'syn':
            # 合成数据的3D-BBOX是生成的
            target_3Dbbox = []
            (temp, img_num) = os.path.split(frame_path)
            (_, file_num) = os.path.split(temp)
            bbox_path = os.path.join(self.dir_path, 'gts', 'train', file_num, img_num)
            with open('{0}_bbox.txt'.format(bbox_path), 'r') as file:
                while True:
                    each_line = file.readline()
                    if not each_line:
                        break
                    if each_line[-1:] == '\n':
                        each_line = each_line[:-1].split(' ')
                    if len(each_line) == 1 and int(each_line[0]) == cur_index:
                        for i in range(8):
                            bbox_line = file.readline()
                            if bbox_line[-1:] == '\n':
                                bbox_line = bbox_line[:-1].split(' ')
                            target_3Dbbox.append([float(bbox_line[0]), float(bbox_line[1]), float(bbox_line[2])])
            target_3Dbbox = np.array(target_3Dbbox)
        elif data_type == 'real':
            # 真实数据的3D_BBOX保存在model_scale中
            target_3Dbbox = []
            with open('{0}/model_scales/{1}.txt'.format(self.dir_path, object), 'r') as file:
                for i in range(8):
                    each_line = file.readline()
                    if each_line[-1:] == '\n':
                        each_line = each_line[:-1].split(' ')
                    target_3Dbbox.append([float(each_line[0]), float(each_line[1]), float(each_line[2])])
            target_3Dbbox = np.array(target_3Dbbox)
        else:
            raise ValueError('data_type error!!')

        target_3Dbbox = enlarge_bbox(copy.deepcopy(target_3Dbbox))
        return target_3Dbbox

    def get_cur_pose(self, object, frame_path, data_type):
        index_list = []
        pose_dict = {}

        (temp, img_num) = os.path.split(frame_path)
        (_, file_num) = os.path.split(temp)

        if self.state == 'train':
            if data_type == 'syn':
                pose_path = os.path.join(self.dir_path, 'gts', 'train', file_num, img_num)
            elif data_type == 'real':
                pose_path = os.path.join(self.dir_path, 'gts', 'real_train', file_num, img_num)
            else:
                raise ValueError('data_type error!!')

            with open('{0}_pose.txt'.format(pose_path), 'r') as file:
                while True:
                    each_line = file.readline()
                    if not each_line:
                        break
                    if each_line[-1:] == '\n':
                        each_line = each_line[:-1].split(' ')
                    if len(each_line) == 1:
                        index = int(each_line[0])
                        index_list.append(index)
                        pose_dict[index] = []
                        for i in range(4):
                            pose_line = file.readline()
                            if pose_line[-1:] == '\n':
                                pose_line = pose_line[:-1].split(' ')
                            pose_dict[index].append([float(pose_line[0]), float(pose_line[1]),
                                                     float(pose_line[2]), float(pose_line[3])])
        elif self.state == 'test':
            if data_type == 'syn':
                pose_path = os.path.join(self.dir_path, 'gts', 'val',
                                         'results_val_{0}_{1}'.format(file_num, img_num))
            elif data_type == 'real':
                pose_path = os.path.join(self.dir_path, 'gts', 'real_test',
                                         'results_real_test_{0}_{1}'.format(file_num, img_num))
            else:
                raise ValueError('data_type error!!')

            with open('{0}.pkl'.format(pose_path), 'rb') as file:
                nocs_data = cPickle.load(file)
                for idx in range(nocs_data['gt_RTs'].shape[0]):
                    idx += 1
                    index_list.append(idx)
                    pose_dict[idx] = nocs_data['gt_RTs'][idx - 1]
                    pose_dict[idx][:3, :3] = \
                        pose_dict[idx][:3, :3] / np.cbrt(np.linalg.det(pose_dict[idx][:3, :3]))
                    z_180_RT = np.zeros((4, 4), dtype=np.float32)
                    z_180_RT[:3, :3] = np.diag([-1, -1, 1])
                    z_180_RT[3, 3] = 1
                    pose_dict[idx] = z_180_RT @ pose_dict[idx]
                    pose_dict[idx][:3, 3] = pose_dict[idx][:3, 3] * 1000

        else:
            raise ValueError('train/test state error!')

        cur_r, cur_t, cur_index = hit_cur_class_result_by_meta(object, frame_path, pose_dict)
        return cur_r, cur_t, cur_index

    def get_cur_2Dbbox(self, cur_3Dbbox, cur_r, cur_t, data_type):
        # 增加噪声
        noise_trans = 0.05
        temp = cur_3Dbbox - (np.array([random.uniform(-noise_trans, noise_trans) for i in range(3)]) * 3000.0)
        temp = np.dot(temp, cur_r.T) + cur_t
        temp[:, 0] *= -1.0
        temp[:, 1] *= -1.0

        # 不增加噪声
        cur_3Dbbox = np.dot(cur_3Dbbox, cur_r.T) + cur_t
        cur_3Dbbox[:, 0] *= -1.0
        cur_3Dbbox[:, 1] *= -1.0

        if data_type == 'syn':
            cam_intrinsics = self.cam_mat_syn
        elif data_type == 'real':
            cam_intrinsics = self.cam_mat_real
        else:
            raise ValueError('data type error!!')

        rmin, rmax, cmin, cmax = get_2dbbox(cur_3Dbbox, cam_intrinsics, self.cam_scale)

        return rmin, rmax, cmin, cmax

    def compute_choose_cloud_by_depth(self, cam_mat, choose, target_depth, cur_rmin, cur_rmax,
                                      cur_cmin, cur_cmax, target_t, target_r):
        cam_cx = cam_mat['cx']
        cam_cy = cam_mat['cy']
        cam_fx = cam_mat['fx']
        cam_fy = cam_mat['fy']

        depth_masked = target_depth.flatten()[choose][:, np.newaxis].astype(np.float32)
        v_map_masked = self.x_map[cur_rmin:cur_rmax, cur_cmin:cur_cmax].flatten()[choose][:, np.newaxis].astype(
            np.float32)
        u_map_masked = self.y_map[cur_rmin:cur_rmax, cur_cmin:cur_cmax].flatten()[choose][:, np.newaxis].astype(
            np.float32)

        zw = depth_masked / self.cam_scale
        xw = (u_map_masked - cam_cx) * zw / cam_fx
        yw = (v_map_masked - cam_cy) * zw / cam_fy
        cloud = np.concatenate((-xw, -yw, zw), axis=1)

        cloud = np.dot(cloud - target_t, target_r)
        return cloud

    def get_choose_clouds(self, data_type, target_depth, cur_rmin, cur_rmax,
                          cur_cmin, cur_cmax, target_t, target_r, cur_limit):
        if data_type == 'syn':
            cam_mat = self.cam_mat_syn
        elif data_type == 'real':
            cam_mat = self.cam_mat_real
        else:
            raise ValueError('data type error!!')

        choose_before = (target_depth.flatten() > -1000.0).nonzero()[0]
        cloud = self.compute_choose_cloud_by_depth(cam_mat, choose_before, target_depth, cur_rmin, cur_rmax,
                                                   cur_cmin, cur_cmax, target_t, target_r)
        # cloud = np.dot(cloud, r.T) + t

        choose_temp = (cloud[:, 0] > cur_limit[0]) * (cloud[:, 0] < cur_limit[1]) * (cloud[:, 1] > cur_limit[2]) * (
                cloud[:, 1] < cur_limit[3]) * (cloud[:, 2] > cur_limit[4]) * (cloud[:, 2] < cur_limit[5])

        choose_after = ((target_depth.flatten() != 0.0) * choose_temp).nonzero()[0]
        # 将密集点云进行降采样
        if len(choose_after) == 0:
            return 0
        if len(choose_after) > self.num_epoint:
            c_mask = np.zeros(len(choose_after), dtype=int)
            c_mask[:self.num_epoint] = 1
            np.random.shuffle(c_mask)
            choose_after = choose_after[c_mask.nonzero()]
        else:
            choose_after = np.pad(choose_after, (0, self.num_epoint - len(choose_after)), 'wrap')

        final_cloud = self.compute_choose_cloud_by_depth(cam_mat, choose_after, target_depth, cur_rmin, cur_rmax,
                                                         cur_cmin, cur_cmax, target_t, target_r)
        choose_after = np.array([choose_after])

        # cloud = np.dot(cloud, r.T) + t

        return final_cloud, choose_after

    def get_cur_frame_info(self, object, frame_path, data_type):
        # 获取当前图像的位姿
        cur_target_r, cur_target_t, cur_target_index = self.get_cur_pose(object, frame_path, data_type)
        cur_target_3Dbbox = self.get_cur_3Dbbox(data_type, frame_path, object, cur_target_index)
        cur_rmin, cur_rmax, cur_cmin, cur_cmax = self.get_cur_2Dbbox(cur_target_3Dbbox, cur_target_r,
                                                                     cur_target_t, data_type)
        cur_depth = get_cur_depth(frame_path)
        cur_limit = search_fit(cur_target_3Dbbox)
        cur_img = Image.open('{0}_color.png'.format(frame_path))
        cur_img = np.array(cur_img)

        target_img = np.transpose(cur_img[:, :, :3], (2, 0, 1))[:, cur_rmin:cur_rmax, cur_cmin:cur_cmax] / 255.0
        target_depth = cur_depth[cur_rmin:cur_rmax, cur_cmin:cur_cmax]
        target_mask = get_target_mask(frame_path, cur_target_index, cur_cmax, cur_cmin, cur_rmax, cur_rmin)

        # target_img = cur_img[:, :, :3][cur_rmin: cur_rmax, cur_cmin: cur_cmax, :] # 只取原图像

        target_clouds, choose = self.get_choose_clouds(data_type, target_depth, cur_rmin, cur_rmax,
                                                       cur_cmin, cur_cmax, cur_target_t, cur_target_r, cur_limit)
        # plot_3D_point_cloud(target_clouds / self.scale_size)
        target_clouds = target_clouds / self.scale_size
        cur_target_3Dbbox = cur_target_3Dbbox / self.scale_size
        if data_type == 'syn':
            syn_mesh_bbox, syn_mesh_pts = self.get_syn_mesh_info(object)
            target_clouds = target_clouds + np.random.normal(loc=0.0, scale=0.003, size=target_clouds.shape)
            return target_img, target_clouds, cur_target_3Dbbox, syn_mesh_pts, syn_mesh_bbox, target_mask, choose
        elif data_type == 'real':
            return target_img, target_clouds, cur_target_3Dbbox, target_mask, choose
        else:
            raise ValueError('data type error!')

    def reset_scale(self, f_bbox, n_bbox):
        res = f_bbox / n_bbox
        res_bbox = f_bbox
        return res_bbox, res[0][0]

    def __len__(self):
        return self.num_data

    def __getitem__(self, item):
        cur_data_type = random_pick(['syn', 'real'], [0.9, 0.1])
        if cur_data_type == 'syn':
            try:
                # 合成图像中随机选择两个同一object的数据
                choose_obj = random.sample(self.syn_obj_name_list[self.id_cate], 1)[0]
                choose_frames = random.sample(self.syn_obj_list[self.id_cate][choose_obj], 2)
                first_img, first_clouds, first_3Dbbox, mesh_pts, mesh_bbox, first_mask, first_choose \
                    = self.get_cur_frame_info(choose_obj, choose_frames[0], cur_data_type)
                second_img, second_clouds, second_3Dbbox, _, _, second_mask, second_choose \
                    = self.get_cur_frame_info(choose_obj, choose_frames[1], cur_data_type)

                # plt.figure()
                # plt.subplot(2, 2, 1)
                # plt.imshow(first_img)
                # plt.subplot(2, 2, 2)
                # plt.imshow(first_mask)
                # plt.subplot(2, 2, 3)
                # plt.imshow(second_img)
                # plt.subplot(2, 2, 4)
                # plt.imshow(second_mask)
                #
                # fig1 = plt.figure(dpi=120)
                # plot_3D_point_cloud(first_clouds, fig1)
                # fig2 = plt.figure(dpi=120)
                # plot_3D_point_cloud(second_clouds, fig2)
                # plt.show()

                # 统一化规模尺寸，以第一帧的作为标准
                unified_3Dbbox, scale_bbox = self.reset_scale(first_3Dbbox, second_3Dbbox)
                mesh_bbox, scale_mesh = self.reset_scale(unified_3Dbbox, mesh_bbox)

                second_clouds = second_clouds * scale_bbox
                mesh_pts = mesh_pts * scale_mesh

            except:
                raise IndexError('dataset index list error!')
        else:
            try:
                # 合成图像中随机选择两个同一object的数据
                choose_obj = random.sample(self.real_obj_name_list[self.id_cate], 1)[0]
                choose_frames = random.sample(self.real_obj_list[self.id_cate][choose_obj], 2)
                first_img, first_clouds, first_3Dbbox, first_mask, first_choose \
                    = self.get_cur_frame_info(choose_obj, choose_frames[0], cur_data_type)
                second_img, second_clouds, second_3Dbbox, second_mask, second_choose \
                    = self.get_cur_frame_info(choose_obj, choose_frames[1], cur_data_type)

                if (np.array(first_3Dbbox) == np.array(second_3Dbbox)).all():
                    unified_3Dbbox = first_3Dbbox
                else:
                    raise SyntaxError(" real 3dBBOX error!")
                # plt.figure(0)
                # plt.subplot(2, 2, 1)
                # plt.imshow(first_img)
                # plt.subplot(2, 2, 2)
                # plt.imshow(first_mask)
                # plt.subplot(2, 2, 3)
                # plt.imshow(second_img)
                # plt.subplot(2, 2, 4)
                # plt.imshow(second_mask)
                #
                # fig1 = plt.figure(1,dpi=120)
                # plot_3D_point_cloud(first_clouds, fig1)
                # fig2 = plt.figure(2,dpi=120)
                # plot_3D_point_cloud(second_clouds, fig2)
                # plt.show()
            except:
                raise IndexError('dataset index list error!')
        # 真实类别标签是从0开始的
        # import pdb
        # pdb.set_trace()
        class_gt = np.array([self.id_cate - 1])
        # 获取 14个方向上的
        #self.get_center_point_of_fourteen_plane(unified_3Dbbox)

        return self.norm(torch.from_numpy(first_img.astype(np.float32))), \
               torch.LongTensor(first_choose.astype(np.int32)), \
               torch.from_numpy(first_clouds.astype(np.float32)), \
               self.norm(torch.from_numpy(second_img.astype(np.float32))), \
               torch.LongTensor(second_choose.astype(np.int32)), \
               torch.from_numpy(second_clouds.astype(np.float32)), \
               torch.from_numpy(self.mesh.astype(np.float32)), \
               torch.LongTensor(class_gt.astype(np.int32))


if __name__ == '__main__':
    a = LoadDataset(state='train', root_path='/media/sjt/data_HD_1/NOCS', num_extracted_point=500,
                    num_category=6, id_category=2, num_data=1000)
    a.__getitem__(11)
    print('q')

    A = []
    B = []
    for i in range(100):
        temp = random_pick(some_list=['syn', 'real'], probabilities=[0.3, 0.1])
        if temp == 'syn':
            A.append(temp)
        else:
            B.append(temp)
    print(len(A))
    print(len(B))
    # print(random_pick(some_list=['syn','real'],probabilities=[0.3, 0.7]))
    # a.__init__(state='train', root_path='/media/sunjingtao/data_HD_1/NOCS', num_extracted_point=500, num_category=6, id_category=5)
