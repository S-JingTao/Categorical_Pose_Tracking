# -*- coding: utf-8 -*-
# @Time    : 2020/11/17:上午9:07
# @Author  : jingtao sun
import os
import cv2
import numpy as np
import multiprocessing
import time
import datetime
# 按类别提取数据集路径索引

class ExtractDataList(object):
    def __init__(self, dataset_root):
        self.dir_root = dataset_root
        self.label = ['1','2','3','4','5','6']
        self.object_name = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug']

    def write_data_list(self,each_line, path,data_save_path):
        class_path = os.path.join(path, each_line[1], each_line[-1])
        if not os.path.exists(class_path):
            os.makedirs(class_path)
        with open(os.path.join(class_path,'list.txt'),'a') as file:
            file.write(str(data_save_path) + '\n')

    def read_txt_file_real(self, path, file_name, data_type, mode):
        with open(os.path.join( path, file_name),'r') as file:
            while True:
                each_line = file.readline()
                if not each_line:
                    break
                if each_line[-1:] == '\n':
                    each_line = each_line[:-1].split(' ')
                    if each_line[1] in self.label:
                        write_path = os.path.join(self.dir_root, 'data_list', data_type, mode)
                        data_save_path = os.path.join(path,file_name.split('.')[0].split('_')[0])
                        self.write_data_list(each_line,write_path,data_save_path)
                    else:
                        continue

    def get_correct_meta_txt_syn(self, path, file_name):
        mask = str(file_name.split('_')[0]) + '_mask.png'
        mask_path = os.path.join(path, mask)
        meta_path = os.path.join(path, file_name)
        mask_im = cv2.imread(mask_path)[:, :, 2]
        mask_im = np.array(mask_im)
        # 实例对象的label列表
        inst_ids = [x for x in np.unique(mask_im) if x < 255]
        cur_meta_mat = []
        index = 0

        with open(meta_path, 'r') as file:
            for line in file:
                line_info = line.split(' ')
                # 当前的 instance id
                cur_inst_id = int(line_info[0])
                # 当前 instance 对应的类别ID
                cur_class_id = int(line_info[1])
                # 去掉标签为0的类别
                if cur_class_id == 0:
                    continue
                if not cur_inst_id in inst_ids:
                    continue
                cur_model_id = str(line_info[-1]).replace("\n", "")
                index += 1
                cur_meta_mat.append([str(index), str(cur_class_id), str(cur_model_id)])
        return cur_meta_mat

    def read_txt_file_syn(self, path, file_name, data_type, mode):

        cur_meta = self.get_correct_meta_txt_syn(path, file_name)
        for line in cur_meta:
            if line[1] in self.label:
                write_path = os.path.join(self.dir_root, 'data_list', data_type, mode)
                data_save_path = os.path.join(path, file_name.split('.')[0].split('_')[0])
                self.write_data_list(line, write_path, data_save_path)
            else:
                continue

    def extract_data_list(self, mode, dir, data_type):
        dir_path = os.path.join(dir, mode)
        video_list = os.listdir(dir_path)
        video_list.sort()

        for num in video_list:
            meta_list = []
            each_path = os.path.join(dir_path, num)
            for each in os.listdir(each_path):
                (filename, extension) = os.path.splitext(each)
                if extension == '.txt' and '_meta' in filename:
                    meta_list.append(each)

            meta_list.sort(key=lambda x: int(x[:4]))
            # meta_list = [x for x in os.listdir(each_path) if x.split('.txt')[0].split('_')[1] == 'meta']
            for each_meta in meta_list:
                if data_type == 'real_data':
                    self.read_txt_file_real(each_path, each_meta, data_type, mode)
                else:
                    self.read_txt_file_syn(each_path, each_meta, data_type, mode)

    def get_path(self):
        path_dict = {}
        path_dict['syn_path'] = os.path.join(self.dir_root, 'syn_data')
        path_dict['real_path'] = os.path.join(self.dir_root, 'real_data')
        path_dict['annotation_path'] = os.path.join(self.dir_root,'gts')
        path_dict['models_path'] = os.path.join(self.dir_root, 'obj_models')
        return path_dict

    def run(self):
        path_dict = self.get_path()
        time_begin = time.time()
        p1 = multiprocessing.Process(target=self.extract_data_list, args=('train', path_dict['syn_path'],'syn_data',))
        p2 = multiprocessing.Process(target=self.extract_data_list, args=('val', path_dict['syn_path'],'syn_data',))
        p3 = multiprocessing.Process(target=self.extract_data_list, args=('real_train', path_dict['real_path'],'real_data',))
        p4 = multiprocessing.Process(target=self.extract_data_list, args=('real_test', path_dict['real_path'],'real_data',))

        p1.start()
        p2.start()
        p3.start()
        p4.start()

        print("The number of CPU is:" + str(multiprocessing.cpu_count()))
        for p in multiprocessing.active_children():
            print("child   p.name:" + p.name + "\tp.id" + str(p.pid))
        print("---END---")

        p1.join()
        p2.join()
        p3.join()
        p4.join()

        time_end = time.time()
        print("spent time: %s " %(datetime.timedelta(seconds=(time_end - time_begin))))


if __name__ == '__main__':
    A = ExtractDataList('/media/sjt/data_HD_1/NOCS')
    A.run()