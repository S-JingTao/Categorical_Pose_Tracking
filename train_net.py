# -*- coding: utf-8 -*-
# @Time    : 2021/11/15:下午2:40
# @Author  : jingtao sun

import numpy as np
import argparse
import torch.utils.data
import torch.optim as optim
from torch.autograd import Variable

from dataset.nocs_dataset import LoadDataset
from model.network import IckNet


def default_hyparams():
    """
    获取公共配置参数
    """
    parsers = argparse.ArgumentParser()
    parsers.add_argument('--nocs_dataset_dir', type=str, default='/media/sjt/data_HD_1/NOCS',
                         help='dataset root dir')
    parsers.add_argument('--resume', type=str, default='', help='resume model')
    parsers.add_argument('--category', type=int, default=5, help='object category to train')
    parsers.add_argument('--num_points', type=int, default=500, help='number of previous canonical points')
    parsers.add_argument('--num_cates', type=int, default=6, help='all number of categories')
    parsers.add_argument('--workers', type=int, default=3, help='number of data loading workers')
    parsers.add_argument('--out_file', type=str, default='models/', help='save dir')
    parsers.add_argument('--lr', default=0.0001, help='learning rate')
    parsers.add_argument('--epoch', type=int, default=100, help='epoch')
    parsers.add_argument('--batch', type=int, default=15, help='batch size')
    parsers.add_argument('--cates_list', type=str, default='bottle,bowl,camera,can,laptop,mug',
                         help='all category list')

    return parsers.parse_args()


def load_dataset(hyparams):
    """
    下载数据
    """
    train_data = LoadDataset('train', hyparams.nocs_dataset_dir,
                             hyparams.num_points, hyparams.num_cates,
                             hyparams.category, 7000)
    val_data = LoadDataset('test', hyparams.nocs_dataset_dir,
                           hyparams.num_points, hyparams.num_cates,
                           hyparams.category, 3000)
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=hyparams.batch,
                                               shuffle=True, num_workers=hyparams.workers)
    val_loader = torch.utils.data.DataLoader(val_data, batch_size=hyparams.batch,
                                             shuffle=True, num_workers=hyparams.workers)
    return train_loader, val_loader


def save_model(model, sign, hyparams, epoch=0, test_dis=0):
    """
    模型保存
    """
    cate_list = hyparams.cates_list.split(',')
    if sign == 'train':
        # 保存训练阶段临时模型
        torch.save(model.state_dict(),
                   '{0}/cur_model_of_{1}.pth'.format(hyparams.out_file,
                                                     cate_list[hyparams.category - 1]))
    elif sign == 'val':
        # 保存最佳验证模型
        torch.save(model.state_dict(),
                   '{0}/model_{1}_{2}_of_{3}.pth'.format(hyparams.out_file, epoch,
                                                         test_dis, cate_list[hyparams.category - 1]))
        print('cur_epoch: %d, cur best val_model saved!' % epoch)
    else:
        raise ValueError('status error!')


def main():
    """
    Train主函数入口
    """
    # 加载预设置的参数
    params = default_hyparams()

    # 加载NOCS数据集
    train_loader, val_loader = load_dataset(params)

    # 加载模型框架
    model = IckNet(num_point=params.num_points, num_cates=params.category)

    # 将模型设置在GPU下运行
    model.cuda()

    # 如果已经存在model, 加载已经存在的模型
    if params.resume != '':
        model.load_state_dict(params.load('{0}/{1}'.format(params.out_file, params.resume)))

    # 创建损失函数
    # criterion = Loss()

    # 构建优化器
    # optimizer = optim.Adam(model.parameters(), lr=params.lr)

    best_test = np.Inf
    for epoch in range(1, params.epoch):
        # 训练阶段
        model.train()
        train_dis_avg = 0.0
        train_count = 0
        # optimizer.zero_grad()

        for i, data in enumerate(train_loader, 0):
            first_img, first_choose, first_cloud, \
            second_img, second_choose, second_cloud, mesh, class_gt = data

            first_img, first_choose, first_cloud, \
            second_img, second_choose, second_cloud, \
            mesh, class_gt = Variable(first_img).cuda(), Variable(first_choose).cuda(), \
                             Variable(first_cloud).cuda(), Variable(second_img).cuda(), \
                             Variable(second_choose).cuda(), Variable(second_cloud).cuda(), \
                             Variable(mesh).cuda(), Variable(class_gt).cuda()
            # 写入模型
            model(first_cloud, second_cloud)
            # 计算损失
            # 反向传播，计算梯度
            # loss.backward()
            # train_dis_avg += loss.item()
            train_count += 1
            if train_count != 0 and train_count % 8 == 0:
                # 更新每一层网络的参数
                # optimizer.step()
                # optimizer.zero_grad()
                train_dis_avg = 0.0
            if train_count != 0 and train_count % 100 == 0:
                # 保存当前的临时模型
                save_model(model, "train", params, epoch, test_dis=0)
        # 测试阶段
        # 固定BN和dropout层的权值，防止在测试阶段变化
        optimizer.zero_grad()
        model.eval()
        val_score = []
        for i, data in enumerate(val_loader, 0):
            first_img, first_choose, first_cloud, \
            second_img, second_choose, second_cloud, mesh, class_gt = data

            first_img, first_choose, first_cloud, \
            second_img, second_choose, second_cloud, \
            mesh, class_gt = Variable(first_img).cuda(), Variable(first_choose).cuda(), \
                             Variable(first_cloud).cuda(), Variable(second_img).cuda(), \
                             Variable(second_choose).cuda(), Variable(second_cloud).cuda(), \
                             Variable(mesh).cuda(), Variable(class_gt).cuda()
        # 导入模型
        # 验证泛化效果

        test_dis = np.mean(np.array(val_score))
        if test_dis < best_test:
            best_test = test_dis
            save_model(model, 'val', params, epoch, test_dis=test_dis)


if __name__ == '__main__':
    main()
