# -*- coding: utf-8 -*-
# @Time    : 2021/12/1:上午10:34
# @Author  : jingtao sun
import torch
import torch.nn as nn
import numpy as np
from torch.autograd import Variable


def gather_nd(params, indices):
    """
    按索引找到对应的列
    """

    orig_shape = list(indices.shape)
    num_samples = np.prod(orig_shape[:-1])
    m = orig_shape[-1]
    n = len(params.shape)

    if m <= n:
        out_shape = orig_shape[:-1] + list(params.shape)[m:]
    else:
        raise ValueError(
            f'the last dimension of indices must less or equal to the rank of params. Got indices:{indices.shape}, params:{params.shape}. {m} > {n}'
        )

    indices = indices.reshape((num_samples, m)).transpose(0, 1).tolist()
    output = params[indices]
    return output.reshape(out_shape).contiguous()


def projection_convert(point_cloud, index):
    # index == points or knn-points
    if index == "points":
        batch_size = point_cloud.size()[0]
        vector_norm = torch.sqrt(torch.sum(point_cloud * point_cloud, -1))

        # AXIS-1
        v1, id1 = torch.topk(vector_norm, k=1)
        batch_indices = torch.reshape(torch.arange(batch_size), (-1, 1)).cuda()
        indices1 = torch.cat((batch_indices, id1), dim=1)
        axis1 = gather_nd(point_cloud, indices1)
        axis1 = axis1 / (torch.norm(axis1, dim=-1, keepdim=True) + 1e-7)

        # AXIS-2
        v2, id2 = torch.topk(-vector_norm, k=1)
        batch_indices = torch.reshape(torch.arange(batch_size), (-1, 1)).cuda()
        indices2 = torch.cat((batch_indices, id2), dim=1)
        axis2 = gather_nd(point_cloud, indices2)
        axis2 = axis2 / (torch.norm(axis2, dim=-1, keepdim=True) + 1e-7)

        # AXIS-3
        axis3 = axis1 + 1.5 * axis2
        axis3 = axis3 / (torch.norm(axis3, dim=-1, keepdim=True) + 1e-7)

        all = torch.norm(point_cloud, dim=-1, keepdim=True) + 1e-7

        new_f1 = torch.sum(point_cloud * axis1.unsqueeze(1), dim=-1, keepdim=True) / all  # (bs,num_point,1)
        new_f2 = torch.sum(point_cloud * axis2.unsqueeze(1), dim=-1, keepdim=True) / all  # (bs,num_point,1)
        new_f3 = torch.sum(point_cloud * axis3.unsqueeze(1), dim=-1, keepdim=True) / all  # (bs,num_point,1)
        new_f4 = vector_norm.unsqueeze(2)
        new_feat = torch.cat([new_f1, new_f2, new_f3, new_f4], dim=2)
    elif index == "knn_points":
        batch_size = point_cloud.size()[0]
        point_num = point_cloud.size()[1]
        center_points = point_cloud[:, :, 0, :]
        center_points = center_points.unsqueeze(2)
        point_cloud = point_cloud - center_points
        vector_norm = torch.sqrt(torch.sum(point_cloud * point_cloud, -1))

        # AXIS-1
        v1, id1 = torch.topk(vector_norm, k=1)
        batch_indices = torch.reshape(torch.arange(batch_size), (-1, 1, 1)).cuda().repeat(1, point_num, 1)
        point_num_indices = torch.reshape(torch.arange(point_num), (1, -1, 1)).cuda().repeat(batch_size, 1, 1)
        indices1 = torch.cat((batch_indices, point_num_indices, id1), dim=2)
        axis1 = gather_nd(point_cloud, indices1)
        axis1 = axis1 / (torch.norm(axis1, dim=-1, keepdim=True) + 1e-7)

        # AXIS-2
        axis2 = torch.mean(point_cloud, dim=2)
        axis2 = axis2 / (torch.norm(axis2, dim=-1, keepdim=True) + 1e-7)

        # AXIS-3
        axis3 = axis1 + 1.5 * axis2
        axis3 = axis3 / (torch.norm(axis3, dim=-1, keepdim=True) + 1e-7)

        all = torch.norm(point_cloud, dim=-1, keepdim=True) + 1e-7

        new_f1 = torch.sum(point_cloud * axis1.unsqueeze(2), dim=-1, keepdim=True) / all  # (bs,num_point,1)
        new_f2 = torch.sum(point_cloud * axis2.unsqueeze(2), dim=-1, keepdim=True) / all  # (bs,num_point,1)
        new_f3 = torch.sum(point_cloud * axis3.unsqueeze(2), dim=-1, keepdim=True) / all  # (bs,num_point,1)
        new_f4 = vector_norm.unsqueeze(3)
        new_feat = torch.cat([new_f1, new_f2, new_f3, new_f4], dim=3)
    else:
        raise ValueError("index is error!")

    return new_feat


class PointWiseFeat(nn.Module):
    def __init__(self, point_num):
        super(PointWiseFeat, self).__init__()

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

        self.inputFC[4].weight.data = torch.zeros(3 * 3, 256)
        self.inputFC[4].bias.data = torch.eye(3).view(-1)

    def forward(self, points):  # [B, N, XYZ]
        batch_size = points.size(0)
        points = points.unsqueeze(1)  # [B, 1, N, XYZ]

        t_net = self.inputTransform(points)  # [B, 1024, 1,1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.inputFC(t_net)  # [B, 3*3]
        t_net = t_net.view(batch_size, 3, 3)  # [B, 3, 3]

        points = points.squeeze(1)  # [B, N, XYZ]

        points = torch.stack(
            [x_item.mm(t_item) for x_item, t_item in zip(points, t_net)])

        points = points.unsqueeze(1)  # [B, 1, N, XYZ]

        points = self.mlp1(points)  # [B, 64, N, 1]

        t_net = self.featureTransform(points)  # [B, 1024, 1, 1]
        t_net = t_net.squeeze()  # [B, 1024]
        t_net = self.featureFC(t_net)  # [B, 64*64]
        t_net = t_net.view(batch_size, 64, 64)  # [B, 64, 64]

        points = points.squeeze(3).permute(0, 2, 1)  # [B, N, 64]

        points = torch.stack([x_item.mm(t_item) for x_item, t_item in zip(points, t_net)])  # [B, N, 64]

        points = points.permute(0, 2, 1).unsqueeze(-1)  # [B, 64, N, 1]

        pwe_out = self.mlp2(points)  # [B, N, 64]

        return pwe_out


class RotationInvariantFeat(nn.Module):
    def __init__(self, knn_num):
        super(RotationInvariantFeat, self).__init__()
        self.knn_num = knn_num

        self.mlp_1 = nn.Sequential(
            nn.Conv2d(4, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.mlp_2 = nn.Sequential(
            nn.Conv2d(192, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
        )

        self.transform = nn.Sequential(
            nn.Conv2d(64, 1, (1, 1)),
            nn.BatchNorm2d(1),
            nn.ReLU(inplace=True),

        )

    def knn_search(self, points):
        batch_size = points.size()[0]
        points_num = points.size()[1]
        # 计算距离矩阵

        def _get_distance_mat(mat_a, mat_b):
            r_A = torch.sum(mat_a * mat_a, dim=2, keepdim=True)
            r_B = torch.sum(mat_b * mat_b, dim=2, keepdim=True)
            m = torch.matmul(mat_a, mat_b.transpose(2, 1))
            dis = r_A - 2 * m + r_B.transpose(2, 1)
            return dis

        dis = _get_distance_mat(points, points)

        dis_v, points_id = torch.topk(-dis, k=self.knn_num, sorted=True)
        batch_indices = torch.reshape(torch.arange(batch_size), (-1, 1, 1, 1)).repeat(1, points_num, self.knn_num,
                                                                                      1).cuda()
        indices = torch.cat([batch_indices, points_id.unsqueeze(3)], dim=3)

        knn_points = gather_nd(points, indices)

        return knn_points

    def graph_conv(self, points):
        batch_size = points.size()[0]
        point_num = points.size()[1]
        new_point = points.view(batch_size * point_num * self.knn_num, 4)

        for i in range(5):
            shape = [new_point.size()[-1], 64]
            int_range = np.sqrt(6.0 / (shape[0] + shape[1]))
            init = nn.init.uniform_(torch.empty(shape[0], shape[1]), a=-int_range, b=int_range)
            init_weight = Variable(init)
            init_bias = Variable(torch.zeros([64], dtype=torch.float))
            new_point = nn.ReLU(inplace=False)(torch.matmul(new_point.cpu(), init_weight) + init_bias)
        new_point = new_point.view(batch_size, point_num, self.knn_num, 64)
        new_point = nn.MaxPool2d((self.knn_num, 1), stride=(1, 1))(new_point)

        return new_point

    def forward(self, points):
        emb_up = self.knn_search(points)
        emb_up = projection_convert(emb_up, index='knn_points')
        emb_up = self.graph_conv(emb_up)
        emb_up = emb_up.view(1, 64, emb_up.size()[1], 1).cuda()

        emb_down = projection_convert(points, index="points")
        emb_down = emb_down.view(1, emb_down.size()[2], emb_down.size()[1], 1)
        emb_down = self.mlp_1(emb_down)

        emb = torch.cat((emb_up, emb_down), 1)
        rie_out = self.mlp_2(emb)
        return rie_out


class IckNet(nn.Module):
    def __init__(self, num_point, num_cates):
        """
        num_cates：当前目标物体类别
        num_point: 前一帧典型模板特征总数
        """
        super(IckNet, self).__init__()
        self.num_point = num_point
        self.num_cates = num_cates
        self.knn_num = 16

        self.rie_feat = RotationInvariantFeat(self.knn_num)
        self.pwe_feat = PointWiseFeat(self.num_point)

        # MLP(3072-1024-256-128-64)
        self.mlp = nn.Sequential(
            nn.Conv2d(3072, 1024, 1),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),

            nn.Conv2d(1024, 256, 1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 128, 1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 64, 1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        self.max_pool = nn.MaxPool2d((int(self.num_point * 2), 1), stride=1)

        self.main_linear = nn.Sequential(
            nn.Linear(64, 3),
            nn.ReLU(inplace=True),
        )

    def forward(self, cur_cloud, pre_cloud):
        cur_cloud = cur_cloud.view(1, cur_cloud.size()[0], cur_cloud.size()[1])
        pre_cloud = pre_cloud.view(1, pre_cloud.size()[0], pre_cloud.size()[1])

        pre_pwe_emb = self.pwe_feat(pre_cloud)
        pre_rie_emb = self.rie_feat(pre_cloud)

        cur_rie_emb = self.rie_feat(cur_cloud)

        cat_1 = torch.cat((pre_pwe_emb, pre_rie_emb), 1)

        # cur_rie_int = torch.tensor(cur_rie_emb,dtype=torch.int64)
        cur_rie_emb = self.max_pool(cur_rie_emb).contiguous()
        cur_rie_emb = cur_rie_emb.repeat(1, 1, self.num_point, 1).contiguous()

        cat_2 = torch.cat((cur_rie_emb, cat_1), 1)

        cat = self.mlp(cat_2)

        cat = cat.view(1, self.num_point, 64).contiguous()
        key_points = self.main_linear(cat)

        return key_points


if __name__ == '__main__':
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
    # cur_points = cur_points.view(1,cur_points.size()[0], cur_points.size()[1])
    model = IckNet(2048, 4)
    # model.train()
    model.cuda()
    model.train()
    res = model(cur_points, pre_points)
    print("final")
    print(res)
