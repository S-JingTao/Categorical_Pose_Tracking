# -*- coding: utf-8 -*-
# @Time    : 2021/11/8 17:15
# @Author  : jingtao sun
# @File    : compute_initial_pose_size.py
# get initial pose at first frame via RGB-D camera
import os
import time
import numpy as np
import skimage.io
import tensorflow as tf
from PIL import Image
import scipy.misc

from inference import mrcnn as modellib
from inference.mrcnn import visualize

from inference.samples.coco import coco
from cass_net import Model, to_device


class InferenceConfig(coco.CocoConfig):
    # Set batch size to 1 since we'll be running inference on
    # one image at a time. Batch size = GPU_COUNT * IMAGES_PER_GPU
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1


def load_mask_net(model_path):
    config = InferenceConfig()
    model = modellib.MaskRCNN(mode="inference", model_dir=model_path, config=config)
    # Load weights trained on MS-COCO
    coco_model_dir = os.path.join(model_path, 'mask_rcnn_coco.h5')
    model.load_weights(coco_model_dir, by_name=True)
    return model


def visualized(r, first_image, class_names):
    # Visualize results
    visualize.display_instances(first_image, r['rois'], r['masks'], r['class_ids'],
                                class_names, r['scores'])


def get_mask_ff(first_image, root_dir, class_names):
    mask_model_dir = os.path.join(root_dir, 'models')
    model = load_mask_net(mask_model_dir)
    start_time = time.clock()
    results = model.detect([first_image], verbose=1)
    print("_________-")
    elapsed = (time.clock() - start_time)
    print("Time used:", elapsed)

    r = results[0]
    visualized(r, first_image, class_names)
    each_mask_dict = {}
    for i in range(len(r['class_ids'])):
        each_class_name = class_names[r['class_ids'][i]]
        given_class = ['bottle', 'bowl', 'camera', 'can', 'laptop', 'mug', 'cup']
        if each_class_name in given_class:
            each_mask = r['masks'][:, :, i]
            each_mask = np.where(each_mask == False, 255, 2)
            save_dir = os.path.join(root_dir, "real_time_data_stream", "results", 'first_mask',
                                    "first_mask_%s.png") % each_class_name
            scipy.misc.imsave(save_dir, each_mask)
            each_mask_dict[each_class_name] = each_mask
    return each_mask_dict


def download_save_mask(dir):
    mask_dict = {}
    for img in os.listdir(dir):
        each_mask = np.array(Image.open(os.path.join(dir, img)))
        each_class_id = img.split(".")[0].split('_')[2]
        mask_dict[each_class_id] = each_mask
    return mask_dict


def load_cass_model():
    from eval import opt
    model = to_device(Model(opt)).eval()

    pass


def get_pose_size_ff(mask_dict, data_dir, first_frame, init_num):
    first_img = first_frame / 255.0
    first_mask = mask_dict
    first_depth = np.load(os.path.join(data_dir, 'depth', '%d.npy') % init_num)
    cass_model = load_cass_model()
    pass


def main():
    root_dir = "/home/sjt/SSCKR-Tracking"
    class_names = ['BG', 'person', 'bicycle', 'car', 'motorcycle', 'airplane',
                   'bus', 'train', 'truck', 'boat', 'traffic light',
                   'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird',
                   'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear',
                   'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie',
                   'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
                   'kite', 'baseball bat', 'baseball glove', 'skateboard',
                   'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup',
                   'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple',
                   'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                   'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed',
                   'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
                   'keyboard', 'cell phone', 'microwave', 'oven', 'toaster',
                   'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors',
                   'teddy bear', 'hair drier', 'toothbrush']

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    session = tf.Session(config=config)
    init_num = 10
    _data_stream_dir = os.path.join(root_dir, "real_time_data_stream")
    # because of previous frame is  fuzzy, the begin frame is ten frame
    first_frame = skimage.io.imread(os.path.join(_data_stream_dir, 'color', '%d.png') % init_num)
    # mask_dict = get_mask_ff(first_frame, root_dir, class_names)
    mask_dict = download_save_mask(os.path.join(_data_stream_dir, 'results', 'first_mask'))
    get_pose_size_ff(mask_dict, _data_stream_dir, first_frame, init_num)


if __name__ == '__main__':
    main()
