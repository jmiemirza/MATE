###  Generate Various Common Corruptions ###
import os
import h5py
import json
import numpy as np
from numpy import random
import open3d as o3d
import argparse
from pygem import FFD, RBF
import glob


# np.random.seed(2021)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default="modelnet40")
    parser.add_argument('--main_path', type=str, default="./data/")
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--corrupted_dataset_path', type=str)

    return parser.parse_args()


def load_mesh(filepath):
    return o3d.io.read_triangle_mesh(filepath)


def export_mesh(mesh, filepath):
    o3d.io.write_triangle_mesh(filepath, mesh)


def load_pcd(filepath):
    return o3d.io.read_point_cloud(filepath)


def export_pcd(pcd, filepath):
    o3d.io.write_point_cloud(filepath, pcd)


def mesh_to_pcd(mesh, number_of_points=2048):
    return mesh.sample_points_uniformly(number_of_points=number_of_points)


def normalize(new_pc):
    new_pc[:, 0] -= (np.max(new_pc[:, 0]) + np.min(new_pc[:, 0])) / 2
    new_pc[:, 1] -= (np.max(new_pc[:, 1]) + np.min(new_pc[:, 1])) / 2
    new_pc[:, 2] -= (np.max(new_pc[:, 2]) + np.min(new_pc[:, 2])) / 2
    leng_x, leng_y, leng_z = np.max(new_pc[:, 0]) - np.min(new_pc[:, 0]), np.max(new_pc[:, 1]) - np.min(
        new_pc[:, 1]), np.max(new_pc[:, 2]) - np.min(new_pc[:, 2])
    if leng_x >= leng_y and leng_x >= leng_z:
        ratio = 2.0 / leng_x
    elif leng_y >= leng_x and leng_y >= leng_z:
        ratio = 2.0 / leng_y
    else:
        ratio = 2.0 / leng_z
    new_pc *= ratio
    return new_pc


def appendSpherical_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    xy = xyz[:, 0] ** 2 + xyz[:, 1] ** 2
    ptsnew[:, 3] = np.sqrt(xy + xyz[:, 2] ** 2)
    ptsnew[:, 4] = np.arctan2(np.sqrt(xy), xyz[:, 2])  # for elevation angle defined from Z-axis down
    # ptsnew[:,4] = np.arctan2(xyz[:,2], np.sqrt(xy)) # for elevation angle defined from XY-plane up
    ptsnew[:, 5] = np.arctan2(xyz[:, 1], xyz[:, 0])
    return ptsnew


def appendCart_np(xyz):
    ptsnew = np.hstack((xyz, np.zeros(xyz.shape)))
    ptsnew[:, 3] = ptsnew[:, 0] * np.sin(ptsnew[:, 1]) * np.cos(ptsnew[:, 2])
    ptsnew[:, 4] = ptsnew[:, 0] * np.sin(ptsnew[:, 1]) * np.sin(ptsnew[:, 2])
    ptsnew[:, 5] = ptsnew[:, 0] * np.cos(ptsnew[:, 1])
    return ptsnew


def lidar_pose(severity):
    """generate a random LiDAR pose"""
    theta = 2 * np.pi * severity / 5
    delta = np.pi / 5
    angle_x = 5. / 8. * np.pi
    angle_y = 0
    angle_z = np.random.uniform(theta - delta, theta + delta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(-R[:, 2] * 5, 1)  # select the third column, reshape into (3, 1)-vector
    pose = np.concatenate([np.concatenate([R, t], 1), [[0, 0, 0, 1]]], 0)
    matrix = np.concatenate([np.concatenate([R.T, -np.dot(R.T, t)], 1), [[0, 0, 0, 1]]], 0)
    return matrix, pose


def random_pose(severity):
    """generate a random camera pose"""

    theta = 2 * np.pi * severity / 5
    delta = np.pi / 5
    angle_x = np.random.uniform(2. / 3. * np.pi, 5. / 6. * np.pi)
    angle_y = 0
    angle_z = np.random.uniform(theta - delta, theta + delta)
    Rx = np.array([[1, 0, 0],
                   [0, np.cos(angle_x), -np.sin(angle_x)],
                   [0, np.sin(angle_x), np.cos(angle_x)]])
    Ry = np.array([[np.cos(angle_y), 0, np.sin(angle_y)],
                   [0, 1, 0],
                   [-np.sin(angle_y), 0, np.cos(angle_y)]])
    Rz = np.array([[np.cos(angle_z), -np.sin(angle_z), 0],
                   [np.sin(angle_z), np.cos(angle_z), 0],
                   [0, 0, 1]])
    R = np.dot(Rz, np.dot(Ry, Rx))
    # a rotation matrix with arbitrarily chosen yaw, pitch, roll
    # Set camera pointing to the origin and 1 unit away from the origin
    t = np.expand_dims(-R[:, 2] * 3., 1)  # select the third column, reshape into (3, 1)-vector

    matrix = np.concatenate([np.concatenate([R.T, -np.dot(R.T, t)], 1), [[0, 0, 0, 1]]], 0)
    return matrix


def get_points(data):
    if isinstance(data, o3d.cpu.pybind.geometry.TriangleMesh):
        return np.asarray(data.vertices)
    elif isinstance(data, o3d.cpu.pybind.geometry.PointCloud):
        return np.asarray(data.points)
    else:
        raise Exception("Wrong input data format: should be pointcloud or mesh")


def set_points(data, points):
    if isinstance(data, o3d.cpu.pybind.geometry.TriangleMesh):
        data.vertices = o3d.utility.Vector3dVector(points)
        return data
    elif isinstance(data, o3d.cpu.pybind.geometry.PointCloud):
        data.points = o3d.utility.Vector3dVector(points)
        return data
    else:
        raise Exception("Wrong input data format: should be pointcloud or mesh")


def get_default_camera_extrinsic():
    return np.array([[1, 0, 0, 1],
                     [0, 1, 0, 0],
                     [0, 0, 1, 2],
                     [0, 0, 0, 1]])


def get_default_camera_intrinsic(width=1920, height=1080):
    return {
        "width": width,
        "height": height,
        "fx": 365,
        "fy": 365,
        "cx": width / 2 - 0.5,
        "cy": height / 2 - 0.5
    }


def core_distortion(points, n_control_points=[2, 2, 2], displacement=None):
    """
        Ref: http://mathlab.github.io/PyGeM/tutorial-1-ffd.html
    """
    # the size of displacement matrix: 3 * control_points.shape
    if displacement is None:
        displacement = np.zeros((3, *n_control_points))

    ffd = FFD(n_control_points=n_control_points)
    ffd.box_length = [2., 2., 2.]
    ffd.box_origin = [-1., -1., -1.]
    ffd.array_mu_x = displacement[0, :, :, :]
    ffd.array_mu_y = displacement[1, :, :, :]
    ffd.array_mu_z = displacement[2, :, :, :]
    new_points = ffd(points)

    return new_points


def distortion(points, direction_mask=np.array([1, 1, 1]), point_mask=np.ones((5, 5, 5)), severity=0.5):
    n_control_points = [5, 5, 5]
    # random
    displacement = np.random.rand(3, *n_control_points) * 2 * severity - np.ones((3, *n_control_points)) * severity
    displacement *= np.transpose(np.tile(direction_mask, (5, 5, 5, 1)), (3, 0, 1, 2))
    displacement *= np.tile(point_mask, (3, 1, 1, 1))

    points = core_distortion(points, n_control_points=n_control_points, displacement=displacement)

    return points


def distortion_2(points, severity=(0.4, 3), func='gaussian_spline'):
    rbf = RBF(func=func)
    xv = np.linspace(-1, 1, severity[1])
    yv = np.linspace(-1, 1, severity[1])
    zv = np.linspace(-1, 1, severity[1])
    z, y, x = np.meshgrid(zv, yv, xv)
    mesh = np.array([x.ravel(), y.ravel(), z.ravel()]).T
    rbf.original_control_points = mesh
    alpha = np.random.uniform(-np.pi, np.pi, mesh.shape[0])
    gamma = np.random.uniform(-np.pi, np.pi, mesh.shape[0])
    distance = np.ones(mesh.shape[0]) * severity[0]
    displacement_x = distance * np.cos(alpha) * np.sin(gamma)
    displacement_y = distance * np.sin(alpha) * np.sin(gamma)
    displacement_z = distance * np.cos(gamma)
    displacement = np.array([displacement_x, displacement_y, displacement_z]).T
    rbf.deformed_control_points = mesh + displacement
    new_points = rbf(points)
    return new_points


def occlusion_1(mesh, type, severity, window_width=1080, window_height=720, n_points=None, downsample_ratio=None):
    points = get_points(mesh)
    points = normalize(points)
    set_points(mesh, points)
    if type == 'occlusion':
        camera_extrinsic = random_pose(severity)
    elif type == 'lidar':
        camera_extrinsic, pose = lidar_pose(severity)
    camera_intrinsic = get_default_camera_intrinsic(window_width, window_height)
    pcd = core_occlusion(mesh, type, camera_extrinsic=camera_extrinsic, camera_intrinsic=camera_intrinsic,
                         window_width=window_width, window_height=window_height, n_points=n_points,
                         downsample_ratio=downsample_ratio)

    points = get_points(pcd)
    if points.shape[0] < n_points:
        index = np.random.choice(points.shape[0], n_points)
        points = points[index]
    if type == 'lidar':
        return points[:n_points, :], pose
    else:
        return points[:n_points, :]


def shuffle_data(data):
    idx = np.arange(data.shape[0])
    np.random.shuffle(idx)
    return data[idx, ...]


def core_occlusion(mesh, type, camera_extrinsic=None, camera_intrinsic=None, window_width=1080, window_height=720,
                   n_points=None, downsample_ratio=None):
    if camera_extrinsic is None:
        camera_extrinsic = get_default_camera_extrinsic()

    if camera_intrinsic is None:
        camera_intrinsic = get_default_camera_intrinsic()

    camera_parameters = o3d.camera.PinholeCameraParameters()
    camera_parameters.extrinsic = camera_extrinsic
    camera_parameters.intrinsic.set_intrinsics(**camera_intrinsic)

    viewer = o3d.visualization.Visualizer()
    viewer.create_window(width=window_width, height=window_height)
    viewer.add_geometry(mesh)

    control = viewer.get_view_control()
    control.convert_from_pinhole_camera_parameters(camera_parameters)
    # viewer.run()

    depth = viewer.capture_depth_float_buffer(do_render=True)

    viewer.destroy_window()
    pcd = o3d.geometry.PointCloud.create_from_depth_image(depth, camera_parameters.intrinsic,
                                                          extrinsic=camera_parameters.extrinsic)

    if downsample_ratio is not None:
        ratio = int((1 - downsample_ratio) / downsample_ratio)
        pcd = pcd.uniform_down_sample(ratio)
    elif n_points is not None:
        # print(np.asarray(pcd.points).shape[0])
        ratio = int(np.asarray(pcd.points).shape[0] / n_points)
        if ratio > 0:
            # if type == 'occlusion':
            set_points(pcd, shuffle_data(np.asarray(pcd.points)))
            pcd = pcd.uniform_down_sample(ratio)

    return pcd


### Transformation ###
'''
Rotate the point cloud
'''


def rotation(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [2.5, 5, 7.5, 10, 15][severity - 1]
    theta = np.random.uniform(c - 2.5, c + 2.5) * np.random.choice([-1, 1]) * np.pi / 180.
    gamma = np.random.uniform(c - 2.5, c + 2.5) * np.random.choice([-1, 1]) * np.pi / 180.
    beta = np.random.uniform(c - 2.5, c + 2.5) * np.random.choice([-1, 1]) * np.pi / 180.

    matrix_1 = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
    matrix_2 = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
    matrix_3 = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

    new_pc = np.matmul(pointcloud, matrix_1)
    new_pc = np.matmul(new_pc, matrix_2)
    new_pc = np.matmul(new_pc, matrix_3).astype('float32')
    return normalize(new_pc), label


'''
Shear the point cloud
'''


def shear(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [0.05, 0.1, 0.15, 0.2, 0.25][severity - 1]
    a = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    b = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    d = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    e = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    f = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])
    g = np.random.uniform(c - 0.05, c + 0.05) * np.random.choice([-1, 1])

    matrix = np.array([[1, 0, b], [d, 1, e], [f, 0, 1]])
    new_pc = np.matmul(pointcloud, matrix).astype('float32')
    return normalize(new_pc), label


'''
Scale the point cloud
'''


def scale(pointcloud, label, severity):
    # TODO
    N, C = pointcloud.shape
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    a = b = d = 1
    r = np.random.randint(0, 3)
    t = np.random.choice([-1, 1])
    if r == 0:
        a += c * t
        b += c * (-t)
    elif r == 1:
        b += c * t
        d += c * (-t)
    elif r == 2:
        a += c * t
        d += c * (-t)

    matrix = np.array([[a, 0, 0], [0, b, 0], [0, 0, d]])
    new_pc = np.matmul(pointcloud, matrix).astype('float32')
    return normalize(new_pc), label


### Noise ###
'''
Add Uniform noise to point cloud 
'''


def uniform_noise(pointcloud, label, severity):
    # TODO
    N, C = pointcloud.shape
    c = [0.01, 0.02, 0.03, 0.04, 0.05][severity - 1]
    jitter = np.random.uniform(-c, c, (N, C))
    new_pc = (pointcloud + jitter).astype('float32')
    return normalize(new_pc), label


'''
Add Gaussian noise to point cloud 
'''


def gaussian_noise(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [0.01, 0.015, 0.02, 0.025, 0.03][severity - 1]
    jitter = np.random.normal(size=(N, C)) * c
    new_pc = (pointcloud + jitter).astype('float32')
    new_pc = np.clip(new_pc, -1, 1)
    return new_pc, label


'''
Add noise to the edge-length-2 cude
'''


def background_noise(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [N // 45, N // 40, N // 35, N // 30, N // 20][severity - 1]

    jitter = np.random.uniform(-1, 1, (c, C))
    new_pc = np.concatenate((pointcloud, jitter), axis=0).astype('float32')
    label_background = [777777] * jitter.shape[0]
    label = np.concatenate((label, label_background), axis=0).astype('float32')
    return normalize(new_pc), label


'''
Upsampling
'''


def upsampling(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [N // 5, N // 4, N // 3, N // 2, N][severity - 1]
    index = np.random.choice(ORIG_NUM, c, replace=False)
    add = pointcloud[index] + np.random.uniform(-0.05, 0.05, (c, C))
    new_pc = np.concatenate((pointcloud, add), axis=0).astype('float32')
    label_add = [777777] * add.shape[0]
    label = np.concatenate((label, label_add), axis=0).astype('float32')
    return normalize(new_pc), label


'''
Add impulse noise
'''


def impulse_noise(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [N // 30, N // 25, N // 20, N // 15, N // 10][severity - 1]
    index = np.random.choice(ORIG_NUM, c, replace=False)
    pointcloud[index] += np.random.choice([-1, 1], size=(c, C)) * 0.1
    return normalize(pointcloud), label


### Point Number Modification ###
'''
Cutout several part in the point cloud
'''


def cutout(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [(2, 30), (3, 30), (5, 30), (7, 30), (10, 30)][severity - 1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        label = np.delete(label, idx.squeeze())
    return pointcloud, label


'''
Uniformly sampling the point cloud
'''


def uniform_sampling(pointcloud, severity):
    N, C = pointcloud.shape
    c = [N // 15, N // 10, N // 8, N // 6, N // 2, 3 * N // 4][severity - 1]
    index = np.random.choice(ORIG_NUM, ORIG_NUM - c, replace=False)
    return pointcloud[index]


'''
Density-based up-sampling the point cloud
'''


def density_inc(pointcloud, label , severity):
    N, C = pointcloud.shape
    c = [(1, 100), (2, 100), (3, 100), (4, 100), (5, 100)][severity - 1]
    # idx = np.random.choice(N,c[0])
    temp = []
    temp_label = []
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        temp.append(pointcloud[idx.squeeze()])
        temp_label.append(label[idx.squeeze()])
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        # label = np.delete(label, idx.squeeze(), axis=0)

    idx = np.random.choice(pointcloud.shape[0], int(round((ORIG_NUM / 2) - c[0] * c[1])))
    temp.append(pointcloud[idx.squeeze()])
    temp_label.append(label[idx.squeeze()])
    pointcloud = np.concatenate(temp)
    return pointcloud, label


'''
Density-based sampling the point cloud
'''


def density(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [(1, 100), (2, 100), (3, 100), (4, 100), (5, 100)][severity - 1]
    for _ in range(c[0]):
        i = np.random.choice(pointcloud.shape[0], 1)
        picked = pointcloud[i]
        dist = np.sum((pointcloud - picked) ** 2, axis=1, keepdims=True)
        idx = np.argpartition(dist, c[1], axis=0)[:c[1]]
        idx_2 = np.random.choice(c[1], int((3 / 4) * c[1]), replace=False)
        idx = idx[idx_2]
        pointcloud = np.delete(pointcloud, idx.squeeze(), axis=0)
        label = np.delete(label, idx.squeeze(), axis=0)
    print(pointcloud.shape)
    return pointcloud, label


def occlusion(severity):
    ## severity here does not stand for real severity ##
    pointcloud = []
    f_0 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_0_id2file.json")
    f_1 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_1_id2file.json")
    lsit_0 = json.load(f_0)
    lsit_1 = json.load(f_1)
    f_0.close()
    f_1.close()

    for item in lsit_0 + lsit_1:
        folder = item.split('/')[0]
        mesh = item.split('/')[1][:-3] + 'off'
        # print(mesh)
        original_data = load_mesh("./data/ModelNet40/" + folder + "/test/" + mesh)
        new_pc = occlusion_1(original_data, 'occlusion', severity, n_points=1024)

        theta = -np.pi / 2.
        gamma = 0
        beta = np.pi

        matrix_1 = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

        new_pc = np.matmul(new_pc, matrix_1)
        new_pc = np.matmul(new_pc, matrix_2)
        new_pc = normalize(np.matmul(new_pc, matrix_3).astype('float32'))

        pointcloud.append(new_pc)

    pointcloud = np.stack(pointcloud, axis=0)

    np.save(args.corrupted_dataset_path + "/data_occlusion_" + str(severity) + ".npy", pointcloud)
    return


def simulate_lidar(pointcloud, pose, severity):
    pose = pose.transpose()
    #####################################
    # simplify the rotation to I matrix #
    pose[:3, :3] = 0
    pose[0, 0] = pose[1, 1] = pose[2, 2] = 1
    # Translate the point cloud #
    pose[3, [0, 1, 2]] = -pose[3, [0, 1, 2]]
    #####################################

    pointcloud_new = np.concatenate([pointcloud, np.ones((pointcloud.shape[0], 1))], axis=1)
    pointcloud_new = np.dot(pointcloud_new, pose)

    pointcloud_new = appendSpherical_np(pointcloud_new[:, :3])
    delta = 1. * np.pi / 180.
    cur = np.min(pointcloud_new[:, 4])

    new_pc = []

    while cur + delta < np.max(pointcloud_new[:4]):
        pointcloud_new[(pointcloud_new[:, 4] >= cur + delta / 4) & (
                pointcloud_new[:, 4] < cur + delta * 3 / 4), 4] = cur + delta / 2.
        new_pc.append(
            pointcloud_new[(pointcloud_new[:, 4] >= cur + delta / 4) & (pointcloud_new[:, 4] < cur + delta * 3 / 4)])
        cur += delta
    new_pc = np.concatenate(new_pc, axis=0)
    # pointcloud = np.dot(pointcloud,np.linalg.inv(pose))
    new_pc = appendCart_np(new_pc[:, 3:])
    new_pc = np.concatenate([new_pc[:, 3:], np.ones((new_pc.shape[0], 1))], axis=1)
    new_pc = np.dot(new_pc, np.linalg.inv(pose))
    index = np.random.choice(new_pc.shape[0], 768)
    new_pc = new_pc[index]
    return new_pc[:, :3]


def lidar(severity):
    ## severity here does not stand for real severity ##
    pointcloud = []
    f_0 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_0_id2file.json")
    f_1 = open("./data/modelnet40_ply_hdf5_2048/ply_data_test_1_id2file.json")
    lsit_0 = json.load(f_0)
    lsit_1 = json.load(f_1)
    f_0.close()
    f_1.close()

    for item in lsit_0 + lsit_1:
        folder = item.split('/')[0]
        mesh = item.split('/')[1][:-3] + 'off'
        original_data = load_mesh("./data/ModelNet40/" + folder + "/test/" + mesh)
        new_pc, pose = occlusion_1(original_data, 'lidar', severity, n_points=1024)

        new_pc = simulate_lidar(new_pc, pose, severity)

        theta = -np.pi / 2.
        gamma = 0
        beta = np.pi

        matrix_1 = np.array([[1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]])
        matrix_2 = np.array([[np.cos(gamma), 0, np.sin(gamma)], [0, 1, 0], [-np.sin(gamma), 0, np.cos(gamma)]])
        matrix_3 = np.array([[np.cos(beta), -np.sin(beta), 0], [np.sin(beta), np.cos(beta), 0], [0, 0, 1]])

        new_pc = np.matmul(new_pc, matrix_1)
        new_pc = np.matmul(new_pc, matrix_2)
        new_pc = np.matmul(new_pc, matrix_3).astype('float32')

        pointcloud.append(new_pc)

    pointcloud = np.stack(pointcloud, axis=0)

    np.save(args.corrupted_dataset_path + "/data_lidar_" + str(severity) + ".npy", pointcloud)
    return


def ffd_distortion(pointcloud,label, severity):
    N, C = pointcloud.shape
    c = [0.1, 0.2, 0.3, 0.4, 0.5][severity - 1]
    new_pc = distortion(pointcloud, severity=c)
    return normalize(new_pc), label


def rbf_distortion(pointcloud,label, severity):
    N, C = pointcloud.shape
    c = [(0.025, 5), (0.05, 5), (0.075, 5), (0.1, 5), (0.125, 5)][severity - 1]
    new_pc = distortion_2(pointcloud, severity=c, func='multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32'), label


def rbf_distortion_inv(pointcloud, label, severity):
    N, C = pointcloud.shape
    c = [(0.025, 5), (0.05, 5), (0.075, 5), (0.1, 5), (0.125, 5)][severity - 1]
    new_pc = distortion_2(pointcloud, severity=c, func='inv_multi_quadratic_biharmonic_spline')
    return normalize(new_pc).astype('float32'), label


def create_mesh(pcd):
    meshclass = o3d.geometry.TriangleMesh()
    test = meshclass.create_from_point_cloud_poisson(pcd=pcd)


def create_data_partnet(args, data, label, corruption, severity):
    data_corr = MAP[corruption](data, label, severity)
    return data_corr


MAP = {
    # 'uniform': uniform_noise,
    #    'gaussian': gaussian_noise,
    #    'background': background_noise,

       # 'impulse': impulse_noise,
       # 'upsampling': upsampling,
       # 'shear': shear,
       # 'rotation': rotation,
       # 'cutout': cutout,
       # 'density': density,
       'density_inc': density_inc,
    'distortion': ffd_distortion,
    # 'distortion_rbf': rbf_distortion,
    # 'distortion_rbf_inv': rbf_distortion_inv,

    # 'occlusion': occlusion,
    # 'lidar': lidar,
    # 'original': None,
}


if __name__ == "__main__":
    args = get_args()

    cat = ['02958343', '04379243', '02954340', '02773838', '03790512', '03642806', '03467517',
           '04099429',
           '03948459', '03624134', '03001627', '02691156', '03636649', '03797390', '04225987',
           '03261776']
    args.debug = False
    from pathlib import Path

    args.dataset_path = './data/partnet/clean'

    for sev in [5]:
        for cor in MAP.keys():
            for c in cat:
                if cor == 'density_inc':
                    ORIG_NUM = 4096
                else:
                    ORIG_NUM = 2048

                path_to_save = Path(f'partnet_c/{cor}_{sev}/{c}')
                path_to_save.mkdir(exist_ok=True, parents=True)
                all_pc = glob.glob(os.path.join(args.dataset_path, f'{c}/*.txt'))
                all_pc = all_pc[:1] if args.debug else all_pc
                for pc in all_pc:
                    txt_file = pc.strip().split('/')[-1]
                    txt_file = os.path.join(f'./{path_to_save}', f'{txt_file}')
                    pc_current = np.loadtxt(pc)
                    current_xyz = pc_current[:, :3]
                    global_coords = pc_current[:, 3:-1]
                    label = pc_current[:, -1]
                    choice = np.random.choice(len(label), ORIG_NUM, replace=True)
                    data_corr = create_data_partnet(args, data=current_xyz[choice, ], label=label[choice], corruption=cor, severity=5)
                    concat = np.column_stack((data_corr[0], data_corr[1]))
                    np.savetxt(txt_file, concat, fmt='%10.6f')
            print("Done with Corruption: {} with Severity: {}".format(cor, 5))




