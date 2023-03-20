import torch
import torch.nn as nn
import os
import json
from tools import builder
from utils import misc, dist_utils
import time
from utils.logger import *
from utils.AverageMeter import AverageMeter

from sklearn.svm import LinearSVC
import numpy as np
from torchvision import transforms
from datasets import data_transforms
from torch.utils.data import DataLoader
from pointnet2_ops import pointnet2_utils

# train_transforms = transforms.Compose(
#     [
#         # data_transforms.PointcloudScale(),
#         # data_transforms.PointcloudRotate(),
#         # # data_transforms.PointcloudRotatePerturbation(),
#         # data_transforms.RandomHorizontalFlip(),
#         # data_transforms.PointcloudTranslate(),
#         # data_transforms.PointcloudJitter(),
#         # data_transforms.PointcloudRandomInputDropout(),
#         data_transforms.PointcloudScaleAndTranslate(),
#     ]
# )


class Acc_Metric:
    def __init__(self, acc=0.):
        if type(acc).__name__ == 'dict':
            self.acc = acc['acc']
        else:
            self.acc = acc

    def better_than(self, other):
        if self.acc > other.acc:
            return True
        else:
            return False

    def state_dict(self):
        _dict = dict()
        _dict['acc'] = self.acc
        return _dict


def evaluate_svm(train_features, train_labels, test_features, test_labels):
    clf = LinearSVC()
    clf.fit(train_features, train_labels)
    pred = clf.predict(test_features)
    return np.sum(test_labels == pred) * 1. / pred.shape[0]


def run_net(args, config, train_writer=None, val_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter

    if args.dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif args.dataset_name == 'scanobject':  # for with background
        config.model.cls_dim = 15
    elif args.dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif args.dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif args.dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    if not args.partnet_cls:
        config.model.group_norm = args.group_norm
        logger = get_logger(args.log_name)
        # build dataset
        (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                                   builder.dataset_builder(args, config.dataset.val)
        (_, extra_train_dataloader) = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get(
            'extra_train') else (None, None)
        # build model
        base_model = builder.model_builder(config.model)
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('TRAINING FROM SCRATCH ...', logger=logger)
        print_log(f'\n\n\n\nMASK RATIO :::: {config.model.transformer_config.mask_ratio}\n\n\n\n', logger=logger)
        if args.use_gpu:
            base_model.to(args.local_rank)

        # from IPython import embed; embed()

        # parameter setting
        start_epoch = 0
        best_metrics = Acc_Metric(0.)
        metrics = Acc_Metric(0.)

        # resume ckpts
        if args.resume:
            start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
            best_metrics = Acc_Metric(best_metric)
        elif args.start_ckpts is not None:
            builder.load_model(base_model, args.start_ckpts, logger=logger)

        # DDP
        if args.distributed:
            # Sync BN
            if args.sync_bn:
                base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
                print_log('Using Synchronized BatchNorm ...', logger=logger)
            base_model = nn.parallel.DistributedDataParallel(base_model,
                                                             device_ids=[args.local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=True)
            print_log('Using Distributed Data parallel ...', logger=logger)
        else:
            print_log('Using Data parallel ...', logger=logger)
            base_model = nn.DataParallel(base_model).cuda()
        # optimizer & scheduler
        optimizer, scheduler = builder.build_opti_sche(base_model, config)
        if args.resume:
            builder.resume_optimizer(optimizer, args, logger=logger)

        # trainval
        # training
        base_model.zero_grad()
        for epoch in range(start_epoch, config.max_epoch + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            base_model.train()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['Loss', 'Reconstruction Loss', 'Classification Loss'])

            num_iter = 0

            base_model.train()  # set model to training mode
            n_batches = len(train_dataloader)
            for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
                num_iter += 1
                n_itr = epoch * n_batches + idx

                data_time.update(time.time() - batch_start_time)
                npoints = config.npoints
                points = data[0].cuda()
                label = data[1].cuda()

                if npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                elif npoints == 4096:
                    point_all = 4800
                elif npoints == 8192:
                    point_all = 8192
                else:
                    raise NotImplementedError()

                if points.size(1) < point_all:
                    point_all = points.size(1)

                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                                  2).contiguous()  # (B, N, 3)
                assert points.size(1) == npoints

                # points = train_transforms(points)

                if not args.cyclic:
                    if args.only_cls:  # only supervised classification task
                        rec_loss = torch.tensor(0)
                        ret = base_model.module.classification_only(points,
                                                                    only_unmasked=args.only_unmasked)  # todo new flag for 100% or 10% tokens for classification (for ablation study)

                    else:  # joint  training
                        rec_loss, ret, _ = base_model(points)

                else:
                    rec_loss, ret = base_model(points,
                                               cyclic=args.cyclic)  # todo for cls loss with 100% and recon loss with 10% tokens

                class_loss, acc = base_model.module.get_loss_acc(ret, label)
                loss = class_loss + rec_loss
                try:
                    loss.backward()
                except:
                    loss = loss.mean()
                    rec_loss = rec_loss.mean()
                    class_loss = class_loss.mean()
                    loss.backward()
                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([loss.item() * 1000, rec_loss.item() * 1000, class_loss.item() * 1000])
                else:
                    losses.update([loss.item() * 1000, rec_loss.item() * 1000, class_loss.item() * 1000])

                if args.distributed:
                    torch.cuda.synchronize()

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/RecLoss', rec_loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/ClassLoss', class_loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 20 == 0:
                    print_log(
                        '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses (total/rec/class) = %s, Acc = %s lr = %.6f' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                         ['%.4f' % l for l in losses.val()], acc.item(), optimizer.param_groups[0]['lr']),
                        logger=logger)
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
            epoch_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s, Acc = %s , lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], acc.item(),
                       optimizer.param_groups[0]['lr']), logger=logger)

            if epoch % args.val_freq == 0 and epoch != 0:
                # Validate the current model
                metrics = validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config,
                                   logger=logger)

                # Save ckeckpoints
                if metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                            logger=logger)
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                                    logger=logger)
            if epoch % 25 == 0 and epoch >= 250:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                        args,
                                        logger=logger)
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()

    else:
        config.model.group_norm = args.group_norm
        logger = get_logger(args.log_name)

        train_dataset = PartNormalDataset(root=config.root, npoints=config.npoint, split='trainval',
                                          normal_channel=config.normal, debug=args.debug)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=50,
                                  drop_last=True)
        test_dataset = PartNormalDataset(root=config.root, npoints=config.npoint, split='test',
                                         normal_channel=config.normal, debug=args.debug)
        test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=10)

        # build model
        base_model = builder.model_builder(config.model)
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('TRAINING FROM SCRATCH ...', logger=logger)

        if args.use_gpu:
            base_model.to(args.local_rank)

        # parameter setting
        start_epoch = 0
        best_metrics = Acc_Metric(0.)
        metrics = Acc_Metric(0.)

        # resume ckpts
        if args.resume:
            start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
            best_metrics = Acc_Metric(best_metric)
        elif args.start_ckpts is not None:
            builder.load_model(base_model, args.start_ckpts, logger=logger)

        # DDP
        if args.distributed:
            # Sync BN
            if args.sync_bn:
                base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
                print_log('Using Synchronized BatchNorm ...', logger=logger)
            base_model = nn.parallel.DistributedDataParallel(base_model,
                                                             device_ids=[args.local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=True)
            print_log('Using Distributed Data parallel ...', logger=logger)
        else:
            print_log('Using Data parallel ...', logger=logger)
            base_model = nn.DataParallel(base_model).cuda()
        # optimizer & scheduler
        optimizer, scheduler = builder.build_opti_sche(base_model, config)

        if args.resume:
            builder.resume_optimizer(optimizer, args, logger=logger)

        base_model.zero_grad()
        for epoch in range(start_epoch, config.max_epoch + 1):
            base_model.train()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['Loss', 'Reconstruction Loss', 'Classification Loss'])

            num_iter = 0

            base_model.train()  # set model to training mode
            n_batches = len(train_loader)
            for idx, (points, label) in enumerate(train_loader):
                num_iter += 1
                n_itr = epoch * n_batches + idx

                data_time.update(time.time() - batch_start_time)
                npoints = config.npoint

                points = points.cuda()
                label = label.cuda()

                assert points.size(1) == npoints

                # if args.train_aug:
                #     points = train_transforms(points)

                if not args.cyclic:
                    if args.only_cls:  # only supervised classification task
                        rec_loss = torch.tensor(0)
                        ret = base_model.module.classification_only(points, only_unmasked=False)  # pass 100% tokens

                    else:  # joint  training
                        rec_loss, ret = base_model(points)

                else:
                    rec_loss, ret = base_model(points,
                                               cyclic=args.cyclic)  # todo for cls loss with 100% and recon loss with 10% tokens

                class_loss, acc = base_model.module.get_loss_acc(ret, label)

                loss = class_loss + rec_loss

                try:
                    loss.backward()
                    # print("Using one GPU")
                except:
                    loss = loss.mean()
                    rec_loss = rec_loss.mean()
                    class_loss = class_loss.mean()
                    loss.backward()
                    # print("Using multi GPUs")

                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([loss.item(), rec_loss.item() * 1000, class_loss.item()])
                else:
                    losses.update([loss.item(), rec_loss.item() * 1000, class_loss.item()])

                if args.distributed:
                    torch.cuda.synchronize()

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/RecLoss', rec_loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/ClassLoss', class_loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/TrainAcc', acc.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 20 == 0:
                    print_log(
                        '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses (total/rec/class) = %s, Acc = %s lr = %.6f' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                         ['%.4f' % l for l in losses.val()], acc.item(), optimizer.param_groups[0]['lr']),
                        logger=logger)
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
            epoch_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s, Acc = %s , lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], acc.item(),
                       optimizer.param_groups[0]['lr']), logger=logger)

            if epoch % args.val_freq == 0 and epoch != 0:
                # Validate the current model
                metrics = validate(base_model, extra_train_dataloader=None, test_dataloader=test_loader, epoch=epoch,
                                   val_writer=val_writer, args=args, config=config,
                                   logger=logger)

                # Save ckeckpoints
                if metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                            logger=logger)
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                                    logger=logger)
            if epoch % 25 == 0 and epoch >= 250:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                        args,
                                        logger=logger)
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()


def rotate_batch_with_labels(batch, labels):
    pts_list = []
    for pts, label in zip(batch, labels):
        if label == 1:
            pts = rotate_point_cloud_by_angle(pts, rotation_angle=90)
        elif label == 2:
            pts = rotate_point_cloud_by_angle(pts, rotation_angle=180)
        elif label == 3:
            pts = rotate_point_cloud_by_angle(pts, rotation_angle=270)
        pts_list.append(pts.unsqueeze(0))
    return torch.cat(pts_list)


def rotate_batch(batch, label='rand'):
    if label == 'rand':
        labels = torch.randint(4, (len(batch),), dtype=torch.long)
    elif label == 'expand':
        labels = torch.cat([torch.zeros(len(batch), dtype=torch.long),
                            torch.zeros(len(batch), dtype=torch.long) + 1,
                            torch.zeros(len(batch), dtype=torch.long) + 2,
                            torch.zeros(len(batch), dtype=torch.long) + 3])
        batch = batch.repeat((4, 1, 1, 1))
    else:
        assert isinstance(label, int)
        labels = torch.zeros((len(batch),), dtype=torch.long) + label

    return rotate_batch_with_labels(batch, labels), labels


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          1xNx3 array, original batch of point clouds
        Return:
          1xNx3 array, rotated batch of point clouds
    """
    rotation_angle = torch.tensor(rotation_angle).cuda()
    rotated_data = torch.zeros(batch_data.shape).cuda()
    # for k in range(batch_data.shape[0]):
    cosval = torch.cos(rotation_angle).cuda()
    sinval = torch.sin(rotation_angle).cuda()
    rotation_matrix = torch.tensor([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]]).cuda()
    shape_pc = batch_data[:, 0:3]
    rotated_data[:, 0:3] = torch.matmul(shape_pc.reshape((-1, 3)), rotation_matrix.cuda())
    return rotated_data.cuda()


def run_net_rot_net(args, config, train_writer=None, val_writer=None):
    if args.dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif args.dataset_name == 'scanobject':  # for with background
        config.model.cls_dim = 15
    elif args.dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif args.dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif args.dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    if not args.partnet_cls:
        config.model.group_norm = args.group_norm
        logger = get_logger(args.log_name)
        # build dataset
        (train_sampler, train_dataloader), (_, test_dataloader), = builder.dataset_builder(args, config.dataset.train), \
                                                                   builder.dataset_builder(args, config.dataset.val)
        (_, extra_train_dataloader) = builder.dataset_builder(args, config.dataset.extra_train) if config.dataset.get(
            'extra_train') else (None, None)
        # build model
        base_model = builder.model_builder(config.model)
        # todo base_models returns -- loss_cls, loss_rot, acc_cls * 100, acc_cls_rot * 100
        if args.ckpts is not None:
            base_model.load_model_from_ckpt(args.ckpts)
        else:
            print_log('TRAINING FROM SCRATCH ...', logger=logger)

        if args.use_gpu:
            base_model.to(args.local_rank)

        # from IPython import embed; embed()

        # parameter setting
        start_epoch = 0
        best_metrics = Acc_Metric(0.)
        metrics = Acc_Metric(0.)

        # resume ckpts
        if args.resume:
            start_epoch, best_metric = builder.resume_model(base_model, args, logger=logger)
            best_metrics = Acc_Metric(best_metric)
        elif args.start_ckpts is not None:
            builder.load_model(base_model, args.start_ckpts, logger=logger)

        # DDP
        if args.distributed:
            # Sync BN
            if args.sync_bn:
                base_model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(base_model)
                print_log('Using Synchronized BatchNorm ...', logger=logger)
            base_model = nn.parallel.DistributedDataParallel(base_model,
                                                             device_ids=[args.local_rank % torch.cuda.device_count()],
                                                             find_unused_parameters=True)
            print_log('Using Distributed Data parallel ...', logger=logger)
        else:
            print_log('Using Data parallel ...', logger=logger)
            base_model = nn.DataParallel(base_model).cuda()
        # optimizer & scheduler
        optimizer, scheduler = builder.build_opti_sche(base_model, config)
        if args.resume:
            builder.resume_optimizer(optimizer, args, logger=logger)

        base_model.zero_grad()
        for epoch in range(start_epoch, config.max_epoch + 1):
            if args.distributed:
                train_sampler.set_epoch(epoch)
            base_model.train()

            epoch_start_time = time.time()
            batch_start_time = time.time()
            batch_time = AverageMeter()
            data_time = AverageMeter()
            losses = AverageMeter(['Loss', 'Reconstruction Loss', 'Classification Loss'])

            num_iter = 0

            base_model.train()  # set model to training mode
            n_batches = len(train_dataloader)
            for idx, (taxonomy_ids, model_ids, data) in enumerate(train_dataloader):
                num_iter += 1
                n_itr = epoch * n_batches + idx

                data_time.update(time.time() - batch_start_time)
                npoints = config.dataset.train.others.npoints
                points = data[0].cuda()
                label = data[1].cuda()

                if npoints == 1024:
                    point_all = 1200
                elif npoints == 2048:
                    point_all = 2400
                elif npoints == 4096:
                    point_all = 4800
                elif npoints == 8192:
                    point_all = 8192
                else:
                    raise NotImplementedError()

                if points.size(1) < point_all:
                    point_all = points.size(1)

                fps_idx = pointnet2_utils.furthest_point_sample(points, point_all)  # (B, npoint)
                fps_idx = fps_idx[:, np.random.choice(point_all, npoints, False)]
                points = pointnet2_utils.gather_operation(points.transpose(1, 2).contiguous(), fps_idx).transpose(1,
                                                                                                                  2).contiguous()  # (B, N, 3)

                assert points.size(1) == npoints

                # points = train_transforms(points)
                pts_rot, label_rot = rotate_batch(points)
                pts_rot, label_rot = pts_rot.cuda(), label_rot.cuda()
                # todo base_models returns -- loss_cls, loss_rot, acc_cls * 100, acc_cls_rot * 100
                # todo base_models takes arguments -- pts, pts_rot, gt, gt_rot, cls_only=False

                loss_cls, loss_rot, acc_cls, acc_cls_rot = base_model(points, pts_rot, label, label_rot)
                loss = loss_cls + loss_rot

                try:
                    loss.backward()
                    # print("Using one GPU")
                except:
                    loss = loss.mean()
                    loss_rot = loss_rot.mean()
                    loss_cls = loss_cls.mean()
                    loss.backward()
                    # print("Using multi GPUs")

                # forward
                if num_iter == config.step_per_update:
                    num_iter = 0
                    optimizer.step()
                    base_model.zero_grad()

                if args.distributed:
                    loss = dist_utils.reduce_tensor(loss, args)
                    losses.update([loss.item() * 1000, loss_rot.item() * 1000, loss_cls.item() * 1000])
                else:
                    losses.update([loss.item() * 1000, loss_rot.item() * 1000, loss_cls.item() * 1000])

                if args.distributed:
                    torch.cuda.synchronize()

                if train_writer is not None:
                    train_writer.add_scalar('Loss/Batch/Loss', loss.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/RecLoss', loss_rot.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/ClassLoss', loss_rot.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/TrainAcc', acc_cls.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/TrainAccRot', acc_cls_rot.item(), n_itr)
                    train_writer.add_scalar('Loss/Batch/LR', optimizer.param_groups[0]['lr'], n_itr)

                batch_time.update(time.time() - batch_start_time)
                batch_start_time = time.time()

                if idx % 20 == 0:
                    print_log(
                        '[Epoch %d/%d][Batch %d/%d] BatchTime = %.3f (s) DataTime = %.3f (s) Losses (total/rec/class) = %s, AccCls = %s AccSSL = %s lr = %.6f' %
                        (epoch, config.max_epoch, idx + 1, n_batches, batch_time.val(), data_time.val(),
                         ['%.4f' % l for l in losses.val()], acc_cls.item(), acc_cls_rot.item(),
                         optimizer.param_groups[0]['lr']),
                        logger=logger)
            if isinstance(scheduler, list):
                for item in scheduler:
                    item.step(epoch)
            else:
                scheduler.step(epoch)
            epoch_end_time = time.time()

            if train_writer is not None:
                train_writer.add_scalar('Loss/Epoch/Loss_1', losses.avg(0), epoch)
            print_log('[Training] EPOCH: %d EpochTime = %.3f (s) Losses = %s, AccCls = %s, AccSSL = %s , lr = %.6f' %
                      (epoch, epoch_end_time - epoch_start_time, ['%.4f' % l for l in losses.avg()], acc_cls.item(),
                       acc_cls_rot.item(),
                       optimizer.param_groups[0]['lr']), logger=logger)

            if epoch % args.val_freq == 0 and epoch != 0:
                # Validate the current model
                metrics = validate_ttt(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args,
                                       config,
                                       logger=logger)

                # Save ckeckpoints
                if metrics.better_than(best_metrics):
                    best_metrics = metrics
                    builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-best', args,
                                            logger=logger)
            builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, 'ckpt-last', args,
                                    logger=logger)
            if epoch % 25 == 0 and epoch >= 250:
                builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}',
                                        args,
                                        logger=logger)
            # if (config.max_epoch - epoch) < 10:
            #     builder.save_checkpoint(base_model, optimizer, epoch, metrics, best_metrics, f'ckpt-epoch-{epoch:03d}', args, logger = logger)
        if train_writer is not None:
            train_writer.close()
        if val_writer is not None:
            val_writer.close()


def validate_ttt(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    if not args.partnet_cls:

        print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)

        base_model.eval()  # set model to eval mode
        npoints = config.npoints
        dataset_name = config.dataset.train._base_.NAME
        all_acc = list()
        all_acc_rot = list()
        with torch.no_grad():
            for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
                if dataset_name == 'ShapeNet':
                    points = data.cuda()
                elif dataset_name == 'ModelNet':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                elif dataset_name == 'ScanObjectNN':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                elif dataset_name == 'ShapeNetCore':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                elif dataset_name == 'scanobject':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                else:
                    raise NotImplementedError(f'Train phase do not support {dataset_name}')

                assert points.size(1) == npoints

                # if args.train_aug:
                # points = train_transforms(points)

                pts_rot, label_rot = rotate_batch(points)
                pts_rot, label_rot = pts_rot.cuda(), label_rot.cuda()

                acc_cls, acc_cls_rot = base_model.module.classification_only(points, pts_rot, label, label_rot)

                all_acc.append(acc_cls.cpu())
                all_acc_rot.append(acc_cls_rot.cpu())

            final_acc = np.mean(all_acc)
            final_acc_rot = np.mean(all_acc_rot)
            print_log('[Validation] EPOCH: %d  classification Acc = %.4f  rotation pred Acc = %.4f'
                      % (epoch, final_acc, final_acc_rot), logger=logger)

            # Add testing results to TensorBoard
            if val_writer is not None:
                val_writer.add_scalar('Metric/TestAcc/TestAccRot', final_acc, final_acc_rot, epoch)

        return Acc_Metric(final_acc)


def validate(base_model, extra_train_dataloader, test_dataloader, epoch, val_writer, args, config, logger=None):
    if not args.partnet_cls:

        print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)

        base_model.eval()  # set model to eval mode
        npoints = config.npoints
        dataset_name = config.dataset.train._base_.NAME
        all_acc = list()
        with torch.no_grad():
            for idx, (taxonomy_ids, model_ids, data) in enumerate(test_dataloader):
                if dataset_name == 'ShapeNet':
                    points = data.cuda()
                elif dataset_name == 'ModelNet':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                elif dataset_name == 'ScanObjectNN':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                elif dataset_name == 'scanobject':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                elif dataset_name == 'ShapeNetCore':
                    points = data[0].cuda()
                    points = misc.fps(points, npoints)
                    label = data[1].cuda()
                else:
                    raise NotImplementedError(f'Train phase do not support {dataset_name}')

                assert points.size(1) == npoints

                # if args.train_aug:
                # points = train_transforms(points)

                ret = base_model.module.classification_only(points, only_unmasked=False)
                _, acc = base_model.module.get_loss_acc(ret, label)
                all_acc.append(acc.cpu())
            final_acc = np.mean(all_acc)
            print_log('[Validation] EPOCH: %d  classification Acc = %.4f' % (epoch, final_acc), logger=logger)

            # Add testing results to TensorBoard
            if val_writer is not None:
                val_writer.add_scalar('Metric/TestAcc', final_acc, epoch)

        return Acc_Metric(final_acc)

    else:
        print_log(f"[VALIDATION] Start validating epoch {epoch}", logger=logger)
        base_model.eval()  # set model to eval mode
        npoints = config.npoint
        all_acc = list()

        with torch.no_grad():
            for batch_id, (points, label) in enumerate(test_dataloader):
                points, label = points.cuda(), label.cuda()

                assert points.size(1) == npoints

                # if args.train_aug:
                #     # points = train_transforms(points)

                ret = base_model.module.classification_only(points, only_unmasked=False)
                _, acc = base_model.module.get_loss_acc(ret, label)
                all_acc.append(acc.cpu())
            final_acc = np.mean(all_acc)
            print_log('[Validation] EPOCH: %d  classification Acc = %.4f' % (epoch, final_acc), logger=logger)

        # Add testing results to TensorBoard
        if val_writer is not None:
            val_writer.add_scalar('Metric/TestAcc', final_acc, epoch)

        return Acc_Metric(final_acc)


def test_net():
    pass

