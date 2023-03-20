import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
import cv2
import numpy as np

corruptions = [
    # 'clean',
    'uniform',
    'gaussian',
    'background', 'impulse', 'upsampling',
    'distortion_rbf', 'distortion_rbf_inv', 'density',
    'density_inc', 'shear', 'rotation',
    'cutout',
    'distortion', 'occlusion', 'lidar',
    # 'mixed_corruptions_2_0', 'mixed_corruptions_2_1', 'mixed_corruptions_2_2'
]
corruptions_shapenet = [
    # 'clean',
    'add_global', 'add_local', 'dropout_global', 'dropout_local', 'jitter', 'rotate', 'scale'
]

def test_net(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    # _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    args.dataset_name = 'modelnet'
    # if args.dataset_name == 'modelnet':
    root = config.tta_dataset_path
    root = os.path.join(root, f'{args.dataset_name}_c')
    args.split = 'test'
    args.corruption = 'distortion'
    args.severity = 5

    for args.corruption in corruptions_shapenet:
        print(args.corruption)
        # if args.corruption == 'clean':
        root = config.tta_dataset_path  # being lazy - 1
        args.corruption_path = root
        if args.dataset_name == 'modelnet':
            root = os.path.join(root, f'{args.dataset_name}_c')

            if args.corruption == 'clean':
                inference_dataset = tta_datasets.ModelNet_h5(args, root)
            else:
                inference_dataset = tta_datasets.ModelNet_h5(args, root)

        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=10)

        # else:
        #     inference_dataset = tta_datasets.ModelNet40C(args, root)
        #     tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

        base_model = builder.model_builder(config.model)
        # base_model.load_model_from_ckpt(args.ckpts)
        builder.load_model(base_model, args.ckpts, logger=logger)

        if args.use_gpu:
            base_model.to(args.local_rank)

        #  DDP
        if args.distributed:
            raise NotImplementedError()

        test(base_model, tta_loader, args, config, logger=logger)


def load_base_model(args, config, logger, load_part_seg=False):
    base_model = builder.model_builder(config.model)
    base_model.load_model_from_ckpt(args.ckpts, load_part_seg)
    if args.use_gpu:
        base_model.to(args.local_rank)
    if args.distributed:
        if args.sync_bn:
            base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
            print_log('Using Synchronized BatchNorm ...', logger=logger)
        base_model = nn.parallel.DistributedDataParallel(base_model, device_ids=[
            args.local_rank % torch.cuda.device_count()], find_unused_parameters=True)
        print_log('Using Distributed Data parallel ...', logger=logger)
    else:
        print_log('Using Data parallel ...', logger=logger)
        base_model = nn.DataParallel(base_model).cuda()
    return base_model


def test_net_partnet(args, config):
    logger = get_logger(args.log_name)
    print_log('Tester start ... ', logger=logger)
    # _, test_dataloader = builder.dataset_builder(args, config.dataset.test)
    args.dataset_name = 'modelnet'
    # if args.dataset_name == 'modelnet':
    root = config.tta_dataset_path
    root = os.path.join(root, f'{args.dataset_name}_c')
    args.split = 'test'
    args.corruption = 'distortion'
    args.severity = 5

    for args.corruption in corruptions_shapenet:
        print(args.corruption)
        root = './data/shapenet_c'

        # root = os.path.join(root, f'{args.dataset_name}_c')
        inference_dataset = tta_datasets.ShapeNetC(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle, num_workers=10)
        base_model = load_base_model(args, config, logger, load_part_seg=True)
        # base_model = builder.model_builder(config.model)

        # base_model.load_model_from_ckpt(args.ckpts)
        # builder.load_model(base_model, args.ckpts, logger=logger)

        if args.use_gpu:
            base_model.to(args.local_rank)

        #  DDP
        if args.distributed:
            raise NotImplementedError()

        test_partnet(base_model, tta_loader, args, config, logger=logger)
# visualization

def save_recon(args, idx, pt, roll, pitch):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    x, z, y = pt.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll, pitch)
    max, min = np.max(pt), np.min(pt)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')
    from pathlib import Path
    Path(f'vis/{args.corruption}/recon').mkdir(exist_ok=True, parents=True)
    plt.savefig(f'vis/{args.corruption}/recon/{idx}.pdf')


def save_org(args, idx, pt, roll, pitch):
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure(figsize=(8, 8))
    x, z, y = pt.transpose(1, 0)
    ax = fig.gca(projection=Axes3D.name, adjustable='box')
    ax.axis('off')
    # ax.axis('scaled')
    ax.view_init(roll, pitch)
    max, min = np.max(pt), np.min(pt)
    ax.set_xbound(min, max)
    ax.set_ybound(min, max)
    ax.set_zbound(min, max)
    ax.scatter(x, y, z, zdir='z', c=y, cmap='jet')
    from pathlib import Path
    Path(f'vis/{args.corruption}/org').mkdir(exist_ok=True, parents=True)
    plt.savefig(f'vis/{args.corruption}/org/{idx}.pdf')


def test(base_model, test_dataloader, args, config, logger=None):
    npoints = config.npoints
    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",  # plane
        "04379243",  # table
        "03790512",  # motorbike
        "03948459",  # pistol
        "03642806",  # laptop
        "03467517",  # guitar
        "03261776",  # earphone
        "03001627",  # chair
        "02958343",  # car
        "04090263",  # rifle
        "03759954",  # microphone
    ]
    with torch.no_grad():
        # for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        reconstructed = list()
        un_masked = list()
        org = list()
        for idx, data in enumerate(test_dataloader):
            # import pdb; pdb.set_trace()
            # if  taxonomy_ids[0] not in useful_cate:
            #     continue
            # if taxonomy_ids[0] == "02691156":
            #     a, b= 90, 135
            # elif taxonomy_ids[0] == "04379243":
            #     a, b = 30, 30
            # elif taxonomy_ids[0] == "03642806":
            #     a, b = 30, -45
            # elif taxonomy_ids[0] == "03467517":
            #     a, b = 0, 90
            # elif taxonomy_ids[0] == "03261776":
            #     a, b = 0, 75
            # elif taxonomy_ids[0] == "03001627":
            #     a, b = 30, -45
            # else:
            #     a, b = 0, 0

            a, b = 90, 135

            dataset_name = config.dataset.test._base_.NAME
            if dataset_name == 'ShapeNet':
                points = data.cuda()
            elif dataset_name == 'ModelNet':
                points = data[0].cuda()
                # points = misc.fps(points, npoints)
            else:
                raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers = base_model(points, vis=True)
            # print(centers.shape)
            # print(points.shape)
            dense_points = dense_points.squeeze().detach().cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(dense_points)
            # from pathlib import Path
            # Path(f'vis/{args.corruption}/recon').mkdir(exist_ok=True, parents=True)
            # o3d.io.write_point_cloud(f"vis/{args.corruption}/recon/{idx}.ply", pcd)

            # save_recon(args, idx, dense_points, a, b)

            final_image = []
            # data_path = f'./vis/{args.exp_name}/{idx}'
            # if not os.path.exists(data_path):
            #     os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # from pathlib import Path
            # Path(f'vis/{args.corruption}/org').mkdir(exist_ok=True, parents=True)
            # o3d.io.write_point_cloud(f"vis/{args.corruption}/org/{idx}.ply", pcd)

            # print(points.shape)
            org.append(points)
            # np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')

            # save_org(args, idx, points, a, b)

            points = misc.get_pointcloud_img(points, a, b, 'groud-truth')
            final_image.append(points[150:650, 150:675, :])

            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)

            vis_points = vis_points.squeeze().detach().cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(vis_points)
            # from pathlib import Path
            # Path(f'vis/{args.corruption}/vis').mkdir(exist_ok=True, parents=True)
            # o3d.io.write_point_cloud(f"vis/{args.corruption}/vis/{idx}.ply", pcd)

            un_masked.append(vis_points)
            # print(vis_points.shape)
            # np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_pointcloud_img(vis_points, a, b, 'visible')

            final_image.append(vis_points[150:650, 150:675, :])

            # dense_points = dense_points.squeeze().detach().cpu().numpy()
            reconstructed.append(dense_points)

            # np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_pointcloud_img(dense_points, a, b, 'reconstructed')
            final_image.append(dense_points[150:650, 150:675, :])

            img = np.concatenate(final_image, axis=1)
            # img_path = os.path.join(data_path, f'plot.jpg')
            from pathlib import Path
            # Path(f'vis/{args.corruption}/org').mkdir(exist_ok=True, parents=True)
            Path(f'modelnet_vis/{args.corruption}/').mkdir(exist_ok=True, parents=True)

            cv2.imwrite(f'modelnet_vis/{args.corruption}/{idx}.png', img)
            # cv2.imwrite(f'vis/{args.corruption}/recon/{idx}.pdf', dense_points)

            if idx > 20:
                break

        from pathlib import Path
        # quit()
        # org = np.stack(org)
        # un_masked = np.stack(un_masked)
        # reconstructed = np.stack(reconstructed)
        # # Path(f'vis/{args.corruption}/').mkdir(exist_ok=True, parents=True)
        # # cv2.imwrite(f'vis/{args.corruption}/{idx}.png', img)
        # print(org.shape)
        # print(un_masked.shape)
        # print(reconstructed.shape)
        # # print()
        #
        # np.save('vis/original.npy', org)
        # np.save('vis/un_masked.npy', un_masked)
        # np.save('vis/reconstructed.npy', reconstructed)
        return
def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y

def test_partnet(base_model, test_dataloader, args, config, logger=None):
    # npoints = config.npoints
    base_model.eval()  # set model to eval mode
    target = './vis'
    useful_cate = [
        "02691156",  # plane
        "04379243",  # table
        "03790512",  # motorbike
        "03948459",  # pistol
        "03642806",  # laptop
        "03467517",  # guitar
        "03261776",  # earphone
        "03001627",  # chair
        "02958343",  # car
        "04090263",  # rifle
        "03759954",  # microphone
    ]
    with torch.no_grad():
        # for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
        reconstructed = list()
        un_masked = list()
        org = list()
        for idx, (data, label, target) in enumerate(test_dataloader):
            points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()

            # import pdb; pdb.set_trace()
            # if  taxonomy_ids[0] not in useful_cate:
            #     continue
            # if taxonomy_ids[0] == "02691156":
            #     a, b= 90, 135
            # elif taxonomy_ids[0] == "04379243":
            #     a, b = 30, 30
            # elif taxonomy_ids[0] == "03642806":
            #     a, b = 30, -45
            # elif taxonomy_ids[0] == "03467517":
            #     a, b = 0, 90
            # elif taxonomy_ids[0] == "03261776":
            #     a, b = 0, 75
            # elif taxonomy_ids[0] == "03001627":
            #     a, b = 30, -45
            # else:
            #     a, b = 0, 0

            a, b = 90, 135

            # dataset_name = config.dataset.test._base_.NAME
            # if dataset_name == 'ShapeNet':
            #     points = data.cuda()
            # elif dataset_name == 'ModelNet':
            #     points = data[0].cuda()
            #     # points = misc.fps(points, npoints)
            # else:
            #     raise NotImplementedError(f'Train phase do not support {dataset_name}')

            # dense_points, vis_points = base_model(points, vis=True)
            dense_points, vis_points, centers = base_model(points, to_categorical(label, 16), vis=True)
            # print(centers.shape)
            # print(points.shape)
            # import open3d as o3d
            # dense_points = misc.fps(dense_points, points.shape[1])
            # print(dense_points.shape)
            dense_points = dense_points.squeeze().detach().cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(dense_points)
            # from pathlib import Path
            # Path(f'vis/{args.corruption}/recon').mkdir(exist_ok=True, parents=True)
            # o3d.io.write_point_cloud(f"vis/{args.corruption}/recon/{idx}.ply", pcd)

            # save_recon(args, idx, dense_points, a, b)

            final_image = []
            # data_path = f'./vis/{args.exp_name}/{idx}'
            # if not os.path.exists(data_path):
            #     os.makedirs(data_path)

            points = points.squeeze().detach().cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(points)
            # from pathlib import Path
            # Path(f'vis/{args.corruption}/org').mkdir(exist_ok=True, parents=True)
            # o3d.io.write_point_cloud(f"vis/{args.corruption}/org/{idx}.ply", pcd)

            # print(points.shape)
            org.append(points)
            # np.savetxt(os.path.join(data_path, 'gt.txt'), points, delimiter=';')

            # save_org(args, idx, points, a, b)

            points = misc.get_pointcloud_img(points, a, b, 'groud-truth')
            final_image.append(points[150:650,150:675,:])

            # centers = centers.squeeze().detach().cpu().numpy()
            # np.savetxt(os.path.join(data_path,'center.txt'), centers, delimiter=';')
            # centers = misc.get_ptcloud_img(centers)
            # final_image.append(centers)

            vis_points = vis_points.squeeze().detach().cpu().numpy()

            # pcd = o3d.geometry.PointCloud()
            # pcd.points = o3d.utility.Vector3dVector(vis_points)
            # from pathlib import Path
            # Path(f'vis/{args.corruption}/vis').mkdir(exist_ok=True, parents=True)
            # o3d.io.write_point_cloud(f"vis/{args.corruption}/vis/{idx}.ply", pcd)

            un_masked.append(vis_points)
            # print(vis_points.shape)
            # np.savetxt(os.path.join(data_path, 'vis.txt'), vis_points, delimiter=';')
            vis_points = misc.get_pointcloud_img(vis_points, a, b, 'visible')

            final_image.append(vis_points[150:650,150:675,:])

            # dense_points = dense_points.squeeze().detach().cpu().numpy()
            reconstructed.append(dense_points)

            # np.savetxt(os.path.join(data_path, 'dense_points.txt'), dense_points, delimiter=';')
            dense_points = misc.get_pointcloud_img(dense_points, a, b, 'reconstructed')
            final_image.append(dense_points[150:650,150:675,:])

            img = np.concatenate(final_image, axis=1)
            # img_path = os.path.join(data_path, f'plot.jpg')
            from pathlib import Path
            # Path(f'vis/{args.corruption}/org').mkdir(exist_ok=True, parents=True)
            Path(f'patnet_vis/{args.corruption}/').mkdir(exist_ok=True, parents=True)

            cv2.imwrite(f'patnet_vis/{args.corruption}/{idx}.png', img)
            # cv2.imwrite(f'vis/{args.corruption}/recon/{idx}.pdf', dense_points)

            if idx > 20:
                break

        from pathlib import Path
        # quit()
        # org = np.stack(org)
        # un_masked = np.stack(un_masked)
        # reconstructed = np.stack(reconstructed)
        # # Path(f'vis/{args.corruption}/').mkdir(exist_ok=True, parents=True)
        # # cv2.imwrite(f'vis/{args.corruption}/{idx}.png', img)
        # print(org.shape)
        # print(un_masked.shape)
        # print(reconstructed.shape)
        # # print()
        #
        # np.save('vis/original.npy', org)
        # np.save('vis/un_masked.npy', un_masked)
        # np.save('vis/reconstructed.npy', reconstructed)
        return