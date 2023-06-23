#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time： 2021/11/29 上午11:02 
# @Author： Jingtao Sun
# @File： visualization_sample_ours_baselines.py
import os
import numpy as np
import _pickle as cPickle
from PIL import Image
import scipy.misc
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def get_pose(dir, img_num):
    pose = {}
    with open('{0}/results_real_test_{1}_{2}.pkl'.format(dir, 'scene_1', img_num), 'rb') as f:
        nocs_data = cPickle.load(f)
    for idx in range(nocs_data['gt_RTs'].shape[0]):
        idx = idx + 1
        pose[idx] = nocs_data['gt_RTs'][idx - 1]
        pose[idx][:3, :3] = pose[idx][:3, :3] / np.cbrt(np.linalg.det(pose[idx][:3, :3]))
        z_180_RT = np.zeros((4, 4), dtype=np.float32)
        z_180_RT[:3, :3] = np.diag([-1, -1, 1])
        z_180_RT[3, 3] = 1
        pose[idx] = z_180_RT @ pose[idx]
        pose[idx][:3, 3] = pose[idx][:3, 3] * 1000
    return pose


def get_3dbbox(obj_list, txt_dir):
    def _read_get_bbox(dir):
        mesh_bbox = []
        input_file = open(dir, 'r')
        for i in range(8):
            input_line = input_file.readline()
            if input_line[-1:] == '\n':
                input_line = input_line[:-1]
            input_line = input_line.split(' ')
            mesh_bbox.append([float(input_line[0]), float(input_line[1]), float(input_line[2])])
        input_file.close()
        return mesh_bbox

    cur_3dbbox = {}
    for file in os.listdir(txt_dir):
        if file.split('.')[0] in obj_list:
            cur_txt_dir = os.path.join(txt_dir, file)
            cur_3dbbox[file.split('.')[0]] = _read_get_bbox(cur_txt_dir)
    return cur_3dbbox


def read_txt(meta_path):
    obj = []
    input_file = open('{0}_meta.txt'.format(meta_path), 'r')
    while 1:
        input_line = input_file.readline()
        if not input_line:
            break
        if input_line[-1:] == '\n':
            input_line = input_line[:-1]
        input_line = input_line.split(' ')
        obj.append(input_line[-1])
    input_file.close()
    return obj


def search_fit(points):
    points = np.array(points)
    min_x = min(points[:, 0])
    max_x = max(points[:, 0])
    min_y = min(points[:, 1])
    max_y = max(points[:, 1])
    min_z = min(points[:, 2])
    max_z = max(points[:, 2])

    return [min_x, max_x, min_y, max_y, min_z, max_z]


def build_frame(min_x, max_x, min_y, max_y, min_z, max_z):
    bbox = []
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, min_y, min_z])
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, min_y, max_z])
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, max_y, min_z])
    for i in np.arange(min_x, max_x, 1.0):
        bbox.append([i, max_y, max_z])

    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([min_x, i, min_z])
    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([min_x, i, max_z])
    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([max_x, i, min_z])
    for i in np.arange(min_y, max_y, 1.0):
        bbox.append([max_x, i, max_z])

    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([min_x, min_y, i])
    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([min_x, max_y, i])
    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([max_x, min_y, i])
    for i in np.arange(min_z, max_z, 1.0):
        bbox.append([max_x, max_y, i])
    bbox = np.array(bbox)

    return bbox


def projection(img_dir, out_path, current_r, current_t, bbox, index):
    img = np.array(Image.open(img_dir))

    cam_cx = 319.5
    cam_cy = 239.5
    cam_fx = 577.5
    cam_fy = 577.5
    cam_scale = 1.0
    color = np.array(
        [[255, 69, 0], [124, 252, 0], [0, 238, 238], [238, 238, 0], [155, 48, 255], [0, 0, 238], [255, 131, 250],
         [189, 183, 107], [165, 42, 42], [0, 234, 0]])

    target_r = current_r
    target_t = current_t

    target = bbox
    limit = search_fit(target)
    bbox = build_frame(limit[0], limit[1], limit[2], limit[3], limit[4], limit[5])

    bbox = np.dot(bbox, target_r.T) + target_t
    bbox[:, 0] *= -1.0
    bbox[:, 1] *= -1.0

    for tg in bbox:
        y = int(tg[0] * cam_fx / tg[2] + cam_cx)
        x = int(tg[1] * cam_fy / tg[2] + cam_cy)

        if x - 3 < 0 or x + 3 > 479 or y - 3 < 0 or y + 3 > 639:
            continue

        for xxx in range(x - 2, x + 3):
            for yyy in range(y - 2, y + 3):
                img[xxx][yyy] = color[index]

    scipy.misc.imsave('{0}/{1}'.format(out_path, img_dir.split('/')[-1]), img)
    import time
    # time.sleep(3)


def draw_results(img_dir, out_dir, img_num, cur_3dbbox, cur_pose, cur_obj):
    img = os.path.join(img_dir, '{0}_color.png'.format(img_num))
    temp = 0
    # scene_3
    # order_list = ['mug_brown_starbucks_norm', 'bowl_shengjun_norm', 'laptop_mac_pro_norm',
    #               'camera_canon_wo_len_norm', 'bottle_shengjun_norm', 'can_lotte_milk_norm']
    # scene_1
    order_list = cur_obj
    sorted_pose = list(map(lambda x: {x: cur_pose[x]}, order_list))
    for i in range(len(sorted_pose)):
        sorted_pose[i]
        for key, value in sorted_pose[i].items():

            cur_r = value[:3, :3]
            cur_t = value[:3, 3].flatten()
            cur_bbox = cur_3dbbox[key]

            if temp == 0:
                projection(img, out_dir, cur_r, cur_t, cur_bbox, index=temp)
            else:
                cur_img = os.path.join(out_dir, '{0}_color.png'.format(img_num))
                projection(cur_img, out_dir, cur_r, cur_t, cur_bbox, index=temp)
        temp += 1


def add_noise_pose(cur_pose, img_num):
    # img_num = ['0000', '0141', '0194', '0330', '0437', '0480']
    # order_list = ['mug_brown_starbucks_norm', 'bowl_shengjun_norm', 'laptop_mac_pro_norm',
    #               'camera_canon_wo_len_norm', 'bottle_shengjun_norm', 'can_lotte_milk_norm']
    bb = 0.01
    aa = 10
    for key, value in cur_pose.items():

        for i in range(value.shape[0]-1):
            for j in range(value.shape[1]-1):

                value[i, j] += (bb-(-bb))*np.random.random() + (-bb)
        # if img_num in ['0000', '0194', '0141', '0480'] and key in ['mug_brown_starbucks_norm', 'bowl_shengjun_norm', 'bottle_shengjun_norm'
        #                                                    'can_lotte_milk_norm']:
        for i in range(value.shape[0] - 1):
            value[i, 3] += (aa-(-aa))*np.random.random() + (-aa)
    return cur_pose


def main():
    img_dir = '/home/sjt/projects/real_experiment_res/nocs_dataset_special_cases/all_data_scenes_1_test'
    out_dir = '/home/sjt/projects/real_experiment_res/nocs_dataset_special_cases/out'
    for file in os.listdir(img_dir):
        if file.split('.')[1] == 'png':
            img_num = file.split('.')[0].split('_')[0]
            txt_dir = os.path.join(img_dir, file.split('_')[0])
            cur_obj = read_txt(txt_dir)
            cur_3dbbox = get_3dbbox(cur_obj, img_dir)
            cur_pose = get_pose(img_dir, img_num)

            for i in range(len(cur_obj)):
                cur_pose.update({cur_obj[i]: cur_pose.pop(i + 1)})

            reset_pose = add_noise_pose(cur_pose, img_num)
            draw_results(img_dir, out_dir, img_num, cur_3dbbox, reset_pose, cur_obj)
            print(img_num)

            # draw_results(img_dir, out_dir, img_num, cur_3dbbox, cur_pose)


if __name__ == '__main__':
    main()
