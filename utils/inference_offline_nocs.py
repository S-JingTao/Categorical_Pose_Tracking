import os
import numpy as np
from inference.libs.tracker import tracker
import cv2


def load_depth(depth_path):
    depth = cv2.imread(depth_path, -1)

    if len(depth.shape) == 3:
        depth16 = np.uint16(depth[:, :, 1] * 256) + np.uint16(depth[:, :, 2])
        depth16 = depth16.astype(np.uint16)
    elif len(depth.shape) == 2 and depth.dtype == 'uint16':
        depth16 = depth
    else:
        assert False, '[ Error ]: Unsupported depth type.'

    return depth16


def depth_png_to_npy():
    # epth.png convert to depth.npy
    depth_path = '/home/sjt/projects/now_work/6-PACK/test_real_robot/test_data_nocs/depth'
    for png in os.listdir(depth_path):
        each_png_path = os.path.join(depth_path, png)
        print(each_png_path)
        _name = png.split(".")[0]
        depth = np.array(load_depth(each_png_path))
        # depth =np.array(cv2.imread(depth_path, -1))
        np.save(os.path.join(depth_path, _name), depth)


# start and end frame
st = 0
ed = 10
# category id
category_id = 4
tracker = tracker(category_id)

# provide the initial pose and scale of the object
current_r = np.array([[0.14951887434063071, 0.029788069104260347, 0.9883100612434088],
                      [-0.45034972151522173, 0.8918988517542368, 0.04125004934100991],
                      [-0.8802438494798317, -0.4512528217956242, 0.1467707609655745]])

current_t = np.array([-91.96180820614123, -35.684646967447634, 1060.269680069195])

# bbox = [[0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0],
#         [0.0, 0.0, 0.0]]
bbox = [[35.0, 68.0, 35.0],
        [35.0, 68.0, -35.0],
        [35.0, -68.0, 35.0],
        [35.0, -68.0, -35.0],
        [-35.0, 68.0, 35.0],
        [-35.0, 68.0, -35.0],
        [-35.0, -68.0, 35.0],
        [-35.0, -68.0, -35.0]]
# depth_png_to_npy()
# import pdb
# pdb.set_trace()
img_path = '/home/sjt/projects/temp/temp_ori/color/{0}.png'
depth_path = '/home/sjt/projects/temp/temp_ori/depth/{0}.npy'

img_initial_dir = img_path.format(st)
depth_initial_dir = depth_path.format(st)
current_r_begin, current_t_begin = tracker.init_estimation(img_initial_dir, depth_initial_dir, current_r, current_t,
                                                           bbox)
# current_r_begin, current_t_begin = current_r,current_t
print('-----------------')
print("begin pose:")
print(current_r_begin)
print(current_t_begin)

for i in range(st + 1, ed):
    img_dir = img_path.format(i)
    depth_dir = depth_path.format(i)
    current_r, current_t = tracker.next_estimation(img_dir, depth_dir, current_r_begin, current_t_begin)
    print('%d frame' % i)
    print('--------------R:')
    print(current_r)
    print('--------------T:')
    print(current_t)
