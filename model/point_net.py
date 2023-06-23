#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time： 2022/1/7 下午9:05
# @Author： Jingtao Sun

import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


class PointNet(nn.Module):
    def __init__(self, point_num):
        super(PointNet, self).__init__()

        self.inputTransform = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((point_num, 1)),
        )
        self.inputFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 9),
        )
        self.mlp1 = nn.Sequential(
            nn.Conv2d(1, 64, (1, 3)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.featureTransform = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.MaxPool2d((point_num, 1)),
        )
        self.featureFC = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 64 * 64),
        )
        self.mlp2 = nn.Sequential(
            nn.Conv2d(64, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )
        self.fc = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            # nn.Dropout(p=0.7,inplace=True),对于ShapeNet数据集来说,用dropout反而准确率会降低
            nn.Linear(256, 16),
            nn.Softmax(dim=1),
        )
        self.inputFC[4].weight.data = torch.zeros(3 * 3, 256)
        self.inputFC[4].bias.data = torch.eye(3).view(-1)

    def forward(self, x):  # [B, N, XYZ]
        '''
            B:batch_size
            N:point_num
            K:k_classes
            XYZ:input_features
        '''
        batch_size = x.size(0)  # batchsize大小
        x = x.unsqueeze(1)  # [B, 1, N, XYZ]

        t_net = self.inputTransform(x)  # [B, 1024, 1,1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.inputFC(t_net)  # [B, 3*3]
        t_net = t_net.view(batch_size, 3, 3)  # [B, 3, 3]

        x = x.squeeze(1)  # [B, N, XYZ]

        x = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(x, t_net)])  # [B, N, XYZ]# 因为mm只能二维矩阵之间，故逐个乘再拼起来

        x = x.unsqueeze(1)  # [B, 1, N, XYZ]

        x = self.mlp1(x)  # [B, 64, N, 1]

        t_net = self.featureTransform(x)  # [B, 1024, 1, 1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.featureFC(t_net)  # [B, 64*64]
        t_net = t_net.view(batch_size, 64, 64)  # [B, 64, 64]

        x = x.squeeze(3).permute(0, 2, 1)  # [B, N, 64]

        x = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(x, t_net)])  # [B, N, 64]

        x = x.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, N, 1]

        x = self.mlp2(x)  # [B, N, 64]

        x, _ = torch.max(x, 2)  # [B, 1024, 1]

        x = self.fc(x.squeeze(2))  # [B, K]
        return x


if __name__ == '__main__':
    # img_f = Image.open("/home/sjt/ICK-Tracking/results/0000_color.png")
    # depth_f = np.array(load_depth("/home/sjt/ICK-Tracking/results/0000_depth.png"))
    # mask_f = cv2.imread("/home/sjt/ICK-Tracking/results/0000_mask.png")[:, :, 0] == 255
    # cam_cx_2 = 319.5
    # cam_cy_2 = 239.5
    # cam_fx_2 = 577.5
    # cam_fy_2 = 577.5
    # img_f = np.transpose(np.array(img_f), (2, 0, 1))
    # img = img_f * (~mask_f)
    # ori_depth = depth_f * (~mask_f)
    from utils.depth_image2point_cloud import Depth2PointCloud

    intrinsic_matrix = [
        [577.5, 0, 319.5],
        [0, 577.5, 239.5],
        [0, 0, 1]
    ]
    import random
    cur = Depth2PointCloud('/home/sjt/ICK-Tracking/results/0000_color.png',
                           '/home/sjt/ICK-Tracking/results/0000_color.png', 'pc1.ply', intrinsic_matrix,
                           scalingfactor=10000)
    cur.calculate()
    cur_points = np.array(random.sample(cur.read_ply(), 4096))

    pre = Depth2PointCloud('/home/sjt/ICK-Tracking/results/0001_color.png',
                           '/home/sjt/ICK-Tracking/results/0001_color.png', 'pc1.ply', intrinsic_matrix,
                           scalingfactor=10000)
    pre.calculate()
    pre_points = np.array(random.sample(pre.read_ply(), 2048))
    cur_points = torch.from_numpy(cur_points.astype(np.float32))
    pre_points = torch.from_numpy(pre_points.astype(np.float32))
    # Tensor装成Variable作为模型的输入
    cur_points, pre_points = Variable(cur_points).cuda(), Variable(pre_points).cuda()
    cur_points = cur_points.view(1,cur_points.size()[0], cur_points.size()[1])

    model = PointNet(4096)
    # model.train()
    model.cuda()
    model.train()
    res = model(cur_points)


