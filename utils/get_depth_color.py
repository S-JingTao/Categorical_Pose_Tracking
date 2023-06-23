# -*- coding: utf-8 -*-
# @Time    : 2021/7/7 10:39
# @Author  : jingtao sun
# @File    : get_depth_color.py
import os
import pyrealsense2 as rs
import numpy as np
import cv2


def show_color_depth(color_image, depth_colormap):
    # Stack both images horizontally
    images_stack = np.hstack((color_image, depth_colormap))
    # Show images
    cv2.namedWindow('RealSense', cv2.WINDOW_AUTOSIZE)
    cv2.imshow('RealSense', images_stack)


def get_show_realsense_params(depth_frame, color_frame):
    dprofile = depth_frame.get_profile()
    cprofile = color_frame.get_profile()

    cvsprofile = rs.video_stream_profile(cprofile)
    dvsprofile = rs.video_stream_profile(dprofile)

    color_intrin = cvsprofile.get_intrinsics()
    print(color_intrin)
    depth_intrin = dvsprofile.get_intrinsics()
    # print(color_intrin)
    extrin = dprofile.get_extrinsics_to(cprofile)
    print(extrin)


def save_data(saved_color, saved_depth, count, save_path):
    # 彩色图片保存为png格式
    if not os.path.exists(os.path.join(save_path, "color")):
        os.makedirs(os.path.join(save_path, "color"))
    if not os.path.exists(os.path.join(save_path, "depth")):
        os.makedirs(os.path.join(save_path, "depth"))

    cv2.imwrite(os.path.join(save_path, "color", "{}.png".format(count)), saved_color)
    # 深度信息由采集到的float16直接保存为npy格式
    np.save(os.path.join(save_path, "depth", "{}".format(count)), saved_depth)


def main():
    # # Configure depth and color streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    # Start streaming
    pipeline.start(config)
    # depth align to color
    align = rs.align(rs.stream.color)
    save_path = '/home/sjt/SSCKR-Tracking/real_time_data_stream'

    try:
        count = 0
        while True:
            # Wait for a coherent pair of frames: depth and color
            frames = pipeline.wait_for_frames()

            frames = align.process(frames)
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            if not depth_frame or not color_frame:
                continue

            # # 获取图像，realsense刚启动的时候图像会有一些失真，我们保存第100帧图片。
            # for i in range(100):
            #     data = pipeline.wait_for_frames()
            #     depth_frame = data.get_depth_frame()
            #     color_frame = data.get_color_frame()
            # if not depth_frame or not color_frame:
            #     continue

            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            get_show_realsense_params(depth_frame, color_frame)

            # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
            depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)

            # save_data(color_image, depth_image, count, save_path)
            count += 1
            int(count)
            if count > 500:
                break

            # real-time display
            # show_color_depth(color_image, depth_colormap)

            key = cv2.waitKey(1)
            # Press esc or 'q' to close the image window
            if key & 0xFF == ord('q') or key == 27:
                cv2.destroyAllWindows()
                break

    finally:
        # Stop streaming
        pipeline.stop()


if __name__ == '__main__':
    main()
