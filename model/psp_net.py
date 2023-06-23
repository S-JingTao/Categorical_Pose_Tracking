import tensorflow as tf
import numpy as np
import math
import sys
import os
import tensorflow.contrib.slim as slim

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, '../utils'))
import tf_util
from transform_nets import input_transform_net, feature_transform_net


def placeholder_inputs(batch_size, num_point):
    pointclouds_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    labels_pl = tf.placeholder(tf.int32, shape=(batch_size))
    normal_pl = tf.placeholder(tf.float32, shape=(batch_size, num_point, 3))
    return pointclouds_pl, labels_pl, normal_pl


def my_conv(points, normals, is_training, KNN=16, unique=False, bn_decay=None):
    # Extract local feature for every point
    # Input: [bs,N,3]
    points_shape = tf.shape(points)
    batch_size = points_shape[0]
    point_num = 1024

    # --------------------------------------------------------------- #
    # calculate KNN points for every point: Input:[batch_size,num_point,3],Return:[batch_size,num_point,K,2]
    # The last number in dimension, 2, means (batch_idx, point_idx)
    # To this end, we first calculate distance matrix
    D = batch_distance_matrix_general(points, points)
    if unique:
        prepare_for_unique_top_k(D, points)
    distances, point_indices = tf.nn.top_k(-D, k=KNN, sorted=True)
    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1, 1)), (1, point_num, KNN, 1))
    indices = tf.concat([batch_indices, tf.expand_dims(point_indices, axis=3)], axis=3)
    # --------------------------------------------------------------- #
    # Find KNN points using above indices
    knn_points = tf.gather_nd(points, indices)  # [batch_size, point_num, K, 3]

    # Calculate key point weight for each poitn
    # normal_of_knn_points = tf.gather_nd(normals, indices)  # [batch_size, point_num, K, 3]
    # first_knn_points = tf.expand_dims(normal_of_knn_points[:, :, 0, :], 2)
    # print('first knn points shape ', np.shape(first_knn_points))
    # fenzi = tf.reduce_sum(normal_of_knn_points * first_knn_points, axis=-1,
    #                       keep_dims=True)  # [batch_size, point_num, K, 1]
    # fenmu = tf.norm(normal_of_knn_points, axis=-1, keep_dims=True) \
    #         * tf.norm(first_knn_points, axis=-1, keep_dims=True)  # [batch_size, point_num, K, 1]
    # cosine = fenzi / fenmu  # [batch_size, point_num, K, 1]
    # degree = tf.reduce_sum(1 - cosine * cosine, axis=2, keep_dims=True) / (KNN - 1)  # [batch_size, point_num, 1, 1]
    # print('degree shape ', np.shape(degree))

    # --------------------------------------------------------------- #
    # [batch_size,num_point,KNN,4]
    new_coordinates = convert_to_new_coordinate_for_local(knn_points)
    # --------------------------------------------------------------- #
    with tf.variable_scope('my_conv') as sc:
        # --------------------------------------------------------------- #
        '''
        # Type1: new feature comes from linear combination of old feature
        # Apply convolution operation across local neighboring points
        new_coordinates_t = tf.transpose(new_coordinates, (0, 2, 3, 1))  # [bs,KNN,4,num_point]
        dw_conv_w1 = tf.Variable(tf.random_uniform((KNN, 1, point_num, 1)), dtype=tf.float32,name='dw_conv_w1')
        feature_across_local_points = tf.nn.depthwise_conv2d(new_coordinates_t, dw_conv_w1, [1, 1, 1, 1],
                                                             padding='VALID',name='dw_conv1')  # [bs,1,4,num_point]
        feature_across_local_points = tf.nn.relu(feature_across_local_points)
        feature_across_local_points_t = tf.transpose(feature_across_local_points, [0, 3, 1, 2])  # [bs,1024,1,4]
        local_feature = tf_util.conv2d(feature_across_local_points_t, 64, [1, 1],
                                       padding='VALID',stride=[1,1],
                                       bn=True,is_training=is_training,
                                       scope='local_conv1',bn_decay=bn_decay)  # [bs,1024,1,64], point-wise fc
        local_feature = tf_util.conv2d(local_feature, 64, [1, 1],
                                       padding='VALID',stride=[1,1],
                                       bn=True,is_training=is_training,
                                       scope='local_conv2',bn_decay=bn_decay)
        '''

        # --------------------------------------------------------------- #
        # Type2: Graph CNN: f(p,t+1) = sum_(q in N(p)){w1*f(q,t)} + bias
        def glorot(shape, name=None):
            """Glorot & Bengio (AISTATS 2010) init."""
            init_range = np.sqrt(6.0 / (shape[0] + shape[1]))
            initial = tf.random_uniform(shape, minval=-init_range, maxval=init_range, dtype=tf.float32)
            return tf.Variable(initial, name=name)

        def zeros(shape, name=None):
            """All zeros."""
            initial = tf.zeros(shape, dtype=tf.float32)
            return tf.Variable(initial, name=name)

        new_coordinates_reshape = tf.reshape(new_coordinates, [batch_size * point_num * KNN, 4])

        local_feature = new_coordinates_reshape
        for i in range(5):
            weight = glorot([local_feature.get_shape().as_list()[-1], 64], name='g_cnn_w{:d}'.format(i))
            bias = zeros([64], name='g_cnn_bias{:d}'.format(i))
            local_feature = tf.nn.relu(tf.matmul(local_feature, weight) + bias)

        local_feature = tf.reshape(local_feature, [batch_size, point_num, KNN, 64])  # [bs,1024,KNN,64]
        # local_feature = tf.reduce_mean(local_feature, axis=2, keep_dims=True)  # [bs,1024,1,64]
        local_feature = tf_util.max_pool2d(local_feature, [1, KNN], stride=[1, 1], padding='VALID',
                                           scope='local_maxpool')  # [bs,1024,1,64]

        print('local_feature_after_pooling shape ', np.shape(local_feature))
        local_feature = tf_util.conv2d(local_feature, 64, [1, 1],
                                       padding='VALID', stride=[1, 1],
                                       bn=True, is_training=is_training,
                                       scope='local_conv2', bn_decay=bn_decay)
        # --------------------------------------------------------------- #
        return local_feature, degree  # [bs,1024,1,64]


def convert_to_new_coordinate(points):
    # Convert the 3d coordinates to 4d coordinates
    # Input points shape: [batch_size,num_point,3]
    # Return: [batch_size,numpoint,4]

    batch_size = tf.shape(points)[0]
    # Calculate 3 axises
    # Axis1 corresponds to the vector with largest norm
    vector_norm = tf.sqrt(tf.reduce_sum(points * points, -1))
    print(np.shape(vector_norm))
    v1, id1 = tf.nn.top_k(vector_norm, k=1)
    batch_indices = tf.reshape(tf.range(batch_size), (-1, 1))
    indices1 = tf.concat([batch_indices, id1], axis=1)
    axis1 = tf.gather_nd(points, indices1)
    axis1 = axis1 / (tf.norm(axis1, axis=-1, keep_dims=True) + 1e-7)  # (bs,3)
    # return axis1

    # ---------------------------------------------------------------#
    # Axis2 corresponds to mean vector
    # axis2 = tf.reduce_mean(points,1)
    # axis2 = axis2 / (tf.norm(axis2,axis=-1,keep_dims=True) + 1e-7)#(bs,3)
    # ---------------------------------------------------------------#

    v2, id2 = tf.nn.top_k(-vector_norm, k=1)
    batch_indices = tf.reshape(tf.range(batch_size), (-1, 1))
    indices2 = tf.concat([batch_indices, id2], axis=1)
    axis2 = tf.gather_nd(points, indices2)
    axis2 = axis2 / (tf.norm(axis2, axis=-1, keep_dims=True) + 1e-7)  # (bs,3)

    # Axis3 is sum of axis1 and axis2
    axis3 = axis1 + 1.5 * axis2
    axis3 = axis3 / (tf.norm(axis3, axis=-1, keep_dims=True) + 1e-7)  # (bs,3)

    fenmu = tf.norm(points, axis=-1, keep_dims=True) + 1e-7
    new_c1 = tf.reduce_sum(points * tf.expand_dims(axis1, 1), axis=-1, keep_dims=True) / fenmu  # (bs,num_point,1)
    new_c2 = tf.reduce_sum(points * tf.expand_dims(axis2, 1), axis=-1, keep_dims=True) / fenmu  # (bs,num_point,1)
    new_c3 = tf.reduce_sum(points * tf.expand_dims(axis3, 1), axis=-1, keep_dims=True) / fenmu  # (bs,num_point,1)
    new_c4 = tf.expand_dims(vector_norm, 2)
    new_c = tf.concat([new_c1, new_c2, new_c3, new_c4], axis=2)
    return new_c


def batch_distance_matrix_general(A, B):
    r_A = tf.reduce_sum(A * A, axis=2, keep_dims=True)
    r_B = tf.reduce_sum(B * B, axis=2, keep_dims=True)
    m = tf.matmul(A, tf.transpose(B, perm=(0, 2, 1)))
    D = r_A - 2 * m + tf.transpose(r_B, perm=(0, 2, 1))
    return D


def find_duplicate_columns(A):
    N = A.shape[0]
    P = A.shape[1]
    indices_duplicated = np.fill((N, 1, P), 1, dtype=np.int32)
    for idx in range(N):
        _, indices = np.unique(A[idx], return_index=True, axis=0)
        indices_duplicated[idx, :, indices] = 0
    return indices_duplicated


def prepare_for_unique_top_k(D, A):
    indices_duplicated = tf.py_func(find_duplicate_columns, [A], tf.int32)
    D += tf.reduce_max(D) * tf.cast(indices_duplicated, tf.float32)


def convert_to_new_coordinate_for_local(points):
    # Convert the 3d coordinates to 4d coordinates
    # Input points shape: [batch_size,num_point,K,3]
    # Return: [batch_size,num_point,K,4]
    points_shape = np.shape(points)
    batch_size = points_shape[0]
    point_num = points_shape[1]

    # Convert to local coordinate, whose origin located at the mean points
    # ------------------------------------------------------ #
    # Type1: substract mean vector
    # mean_point = tf.reduce_mean(points,axis=2,keep_dims=True) # [batch_size,num_point,1,3]
    # points = points - mean_point
    # ------------------------------------------------------ #
    # Type2: substract central vector
    central_point = points[:, :, 0, :]  # [batch_size,num_point,3]
    central_point = tf.expand_dims(central_point, 2)  # [batch_size,num_point,1,3]
    points = points - central_point
    # return [batch_size, point_num, 1, 3]
    print('hahahahaha hahahahaha ', np.shape(points))
    # ------------------------------------------------------ #

    # Calculate 3 axises
    # Axis1 corresponds to the vector with largest norm
    vector_norm = tf.sqrt(tf.reduce_sum(points * points, -1))  # [batch_size,num_point,KNN]
    print('haha vector norm ', np.shape(vector_norm))
    v1, id1 = tf.nn.top_k(vector_norm, k=1)

    batch_indices = tf.tile(tf.reshape(tf.range(batch_size), (-1, 1, 1)), (1, point_num, 1))
    point_num_indices = tf.tile(tf.reshape(tf.range(point_num), (1, -1, 1)), (batch_size, 1, 1))

    indices1 = tf.concat([batch_indices, point_num_indices, id1], axis=2)
    axis1 = tf.gather_nd(points, indices1)
    axis1 = axis1 / (tf.norm(axis1, axis=-1, keep_dims=True) + 1e-7)  # (bs,num_point,3)

    # Axis2 corresponds to mean vector
    axis2 = tf.reduce_mean(points, axis=2)
    axis2 = axis2 / (tf.norm(axis2, axis=-1, keep_dims=True) + 1e-7)  # (bs,num_point,3)

    # Axis3 is sum of axis1 and axis2
    axis3 = axis1 + 1.5 * axis2
    axis3 = axis3 / (tf.norm(axis3, axis=-1, keep_dims=True) + 1e-7)  # (bs,num_point,3)

    fenmu = tf.norm(points, axis=-1, keep_dims=True) + 1e-7

    new_c1 = tf.reduce_sum(points * tf.expand_dims(axis1, 2), axis=-1, keep_dims=True) / fenmu  # (bs,num_point,KNN,1)
    new_c2 = tf.reduce_sum(points * tf.expand_dims(axis2, 2), axis=-1, keep_dims=True) / fenmu  # (bs,num_point,KNN,1)
    new_c3 = tf.reduce_sum(points * tf.expand_dims(axis3, 2), axis=-1, keep_dims=True) / fenmu  # (bs,num_point,KNN,1)
    new_c4 = tf.expand_dims(vector_norm, 3)
    new_c = tf.concat([new_c1, new_c2, new_c3, new_c4], axis=3)
    return new_c  # [batch_size,num_point,KNN,4]


def get_model(point_cloud, normals, is_training=True, bn_decay=None):
    """ Classification PointNet, input is BxNx3, output Bx40 """
    batch_size = point_cloud.get_shape()[0].value
    num_point = point_cloud.get_shape()[1].value
    end_points = {}

    # ------------------------------------------------------ #
    # Extract local feature for every point    ->    local_feature
    # Key point detection, containing edge, corner     ->    points_weight  [bs,1024,1,1]
    local_feature, points_weight = my_conv(point_cloud, normals, is_training=is_training, KNN=25, bn_decay=bn_decay)
    # ------------------------------------------------------ #

    # ------------------------------------------------------ #
    point_cloud = convert_to_new_coordinate(point_cloud)

    # with tf.variable_scope('transform_net1') as sc:
    #	transform = input_transform_net(point_cloud, is_training, bn_decay, K=4)
    # point_cloud_transformed = tf.matmul(point_cloud, transform)
    point_cloud_transformed = point_cloud
    input_image = tf.expand_dims(point_cloud_transformed, -1)  # [bs,N,4,1]

    net = tf_util.conv2d(input_image, 64, [1, 4],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv1', bn_decay=bn_decay)  # [bs,N,1,64]
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv2', bn_decay=bn_decay)
    print('hahahahha1 ', np.shape(net))

    # ------------------------------------------------------ #
    # Combine original feature and new local feature
    net = tf.concat([net, local_feature], axis=3)
    net = tf_util.conv2d(net, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv_conbine', bn_decay=bn_decay)
    # ------------------------------------------------------ #

    # with tf.variable_scope('transform_net2') as sc:
    #	transform = feature_transform_net(net, is_training, bn_decay, K=64)
    # end_points['transform'] = transform
    # net_transformed = tf.matmul(tf.squeeze(net, axis=[2]), transform)
    # net_transformed = tf.expand_dims(net_transformed, [2])

    net_transformed = net
    net = tf_util.conv2d(net_transformed, 64, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv3', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 128, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv4', bn_decay=bn_decay)
    net = tf_util.conv2d(net, 1024, [1, 1],
                         padding='VALID', stride=[1, 1],
                         bn=True, is_training=is_training,
                         scope='conv5', bn_decay=bn_decay)

    # ---------------------------------------------------- #
    # add weight and emphasize key point before maxpooling
    net = net + points_weight
    # ---------------------------------------------------- #

    # Symmetric function: max pooling
    net = tf_util.max_pool2d(net, [num_point, 1], padding='VALID', scope='maxpool')
    # net = tf_util.avg_pool2d(net, [num_point,1],padding='VALID', scope='maxpool')

    net = tf.reshape(net, [batch_size, -1])
    net = tf_util.fully_connected(net, 512, bn=True, is_training=is_training,
                                  scope='fc1', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp1')
    net = tf_util.fully_connected(net, 256, bn=True, is_training=is_training,
                                  scope='fc2', bn_decay=bn_decay)
    net = tf_util.dropout(net, keep_prob=0.7, is_training=is_training,
                          scope='dp2')
    net = tf_util.fully_connected(net, 40, activation_fn=None, scope='fc3')

    return net, end_points


def get_loss(pred, label, end_points, reg_weight=0.001):
    """ pred: B*NUM_CLASSES,
        label: B, """
    loss = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=pred, labels=label)
    classify_loss = tf.reduce_mean(loss)
    tf.summary.scalar('classify loss', classify_loss)

    # Enforce the transformation as orthogonal matrix
    '''
    transform = end_points['transform'] # BxKxK
    K = transform.get_shape()[1].value
    mat_diff = tf.matmul(transform, tf.transpose(transform, perm=[0,2,1]))
    mat_diff -= tf.constant(np.eye(K), dtype=tf.float32)
    mat_diff_loss = tf.nn.l2_loss(mat_diff)
    tf.summary.scalar('mat loss', mat_diff_loss)'''

    return classify_loss  # + mat_diff_loss * reg_weight


if __name__ == '__main__':
    with tf.Graph().as_default():
        inputs = tf.zeros((32, 1024, 3))
        outputs = get_model(inputs, tf.constant(True))
        print(outputs)
