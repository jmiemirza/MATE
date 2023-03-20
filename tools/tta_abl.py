import torch
import torch.nn as nn
from tools import builder
from utils import misc, dist_utils
from utils.logger import *
from utils.AverageMeter import AverageMeter
import os
import datasets.tta_datasets as tta_datasets
from torch.utils.data import DataLoader
import numpy as np
from utils.rotnet_utils import rotate_batch
from pointnet2_ops import pointnet2_utils
from pathlib import Path
from pthflops import count_ops

level = [5]
corruptions_partnet = [
    # 'clean',
    'uniform',
    'gaussian',
    # 'background',
    'impulse',
    # 'upsampling',
    'distortion_rbf',
    'distortion_rbf_inv', 'density',
    # 'density_inc',
    'shear',
    'rotation', 'cutout',
    'distortion',
    # 'occlusion', 'lidar'
]
corruptions_scanobj = [
    # 'clean',
    # 'uniform',
    'gaussian',
    'background',
    'impulse',
    'upsampling',
    'distortion_rbf',
    'distortion_rbf_inv', 'density',
    'density_inc',
    'shear',
    'rotation', 'cutout',
    'distortion',
    'occlusion', 'lidar'
]

corruptions = [
    # 'clean',
    'uniform', 'gaussian', 'background', 'impulse', 'upsampling',
    'distortion_rbf', 'distortion_rbf_inv', 'density',
    'density_inc', 'shear', 'rotation',
    'cutout', 'distortion', 'occlusion', 'lidar',
    # 'mixed_corruptions_2_0', 'mixed_corruptions_2_1', 'mixed_corruptions_2_2'
]

corruptions_h5 = [
    # 'clean',
    'add_global', 'add_local', 'dropout_global', 'dropout_local', 'jitter', 'rotate', 'scale'
]


def load_tta_dataset(args, config):
    # we have 3 choices - every tta_loader returns only point and labels
    root = config.tta_dataset_path  # being lazy - 1

    if args.dataset_name == 'modelnet':
        root = os.path.join(root, f'{args.dataset_name}_c')

        if args.corruption == 'clean':
            inference_dataset = tta_datasets.ModelNet_h5(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)
        else:
            inference_dataset = tta_datasets.ModelNet40C(args, root)
            tta_loader = DataLoader(dataset=inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    elif args.dataset_name == 'partnet':
        if args.corruption != 'clean':
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}_{args.severity}')
        else:
            root = os.path.join(root, f'{args.dataset_name}_c',
                                f'{args.corruption}')

        inference_dataset = tta_datasets.PartNormalDataset(root=root, npoints=config.npoints, split='test',
                                                           normal_channel=config.normal, debug=args.debug)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    elif args.dataset_name == 'scanobject_wbg':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    elif args.dataset_name == 'scanobject_nbg':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ScanObjectNN(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    elif args.dataset_name == 'shapenetcore':

        root = os.path.join(root, f'{args.dataset_name}_c')

        inference_dataset = tta_datasets.ShapeNetCore(args=args, root=root)
        tta_loader = DataLoader(inference_dataset, batch_size=args.batch_size, shuffle=args.shuffle)

    else:
        raise NotImplementedError(f'TTA for {args.tta} is not implemented')

    print(f'\n\n Loading data from ::: {root} ::: level ::: {args.severity}\n\n')

    return tta_loader


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


def eval_source(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject_wbg':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path='source_only_results/')  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):

                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name in ['scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    logits = base_model.module.classification_only(points, only_unmasked=False)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:', os.path.join('source_only_results/', f'{logtime}_results.txt'))


def eval_source_rotnet(args, config):
    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)
    dataset_name = args.dataset_name

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject':
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):

            if corr_id == 0:
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path='source_only_results_rotnet/')  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'Source Only Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Check Point: {args.ckpts}' + '\n\n')

            base_model = load_base_model(args, config, logger)
            print('Testing Source Performance...')
            test_pred = []
            test_label = []
            base_model.eval()

            inference_loader = load_tta_dataset(args, config)

            with torch.no_grad():
                for idx_inference, (data, labels) in enumerate(inference_loader):

                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'scanobject':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                        label = labels.cuda()
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                        label = labels.cuda()

                    points = points.cuda()
                    labels = label.cuda()
                    logits = base_model.module.classification_only(points, 0, 0, 0, tta=True)
                    target = labels.view(-1)
                    pred = logits.argmax(-1).view(-1)

                    test_pred.append(pred.detach())
                    test_label.append(target.detach())

                test_pred = torch.cat(test_pred, dim=0)
                test_label = torch.cat(test_label, dim=0)

                if args.distributed:
                    test_pred = dist_utils.gather_tensor(test_pred, args)
                    test_label = dist_utils.gather_tensor(test_label, args)

                acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
                print(f'Source Peformance ::: Corruption ::: {args.corruption} ::: {acc}')

                f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
                f_write.flush()
                if corr_id == len(corruptions) - 1:
                    f_write.close()
                    print(f'Final Results Saved at:',
                          os.path.join('source_only_results_rotnet/', f'{logtime}_results.txt'))


def tta_rotnet(args, config, train_writer=None):
    dataset_name = args.dataset_name

    assert args.tta_rot
    assert dataset_name is not None
    assert args.mask_ratio == 0.9
    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject_wbg':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    args.batch_size_tta = 48
    args.batch_size = 1
    args.disable_bn_adaptation = True

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='tta_rotnet_results/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred = []
            test_label = []

            if args.online:
                base_model = load_base_model(args, config, logger)
                optimizer = builder.build_opti_sche(base_model, config)[0]

            for idx, (data, labels) in enumerate(tta_loader):
                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    base_model = load_base_model(args, config, logger)
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # making a batch
                    points = [points for _ in range(args.batch_size_tta)]
                    points = torch.squeeze(torch.vstack(points))
                    pts_rot, label_rot = rotate_batch(points)
                    pts_rot, label_rot = pts_rot.cuda(), label_rot.cuda()
                    loss = base_model(0, pts_rot, 0, label_rot, tta=True)  # get out only rotnet loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Rot Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)
                logits = base_model.module.classification_only(points, 0, 0, 0, tta=True)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 50 == 0:
                    # intermediate results
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.
                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('tta_rotnet_results/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def tta(args, config, train_writer=None):
    dataset_name = args.dataset_name
    assert args.tta
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject_wbg':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    args.disable_bn_adaptation = True

    args.batch_size_tta = 48
    args.batch_size = 1

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            all_sample_loss = list()
            all_sample_acc = list()

            acc_sliding_window = list()
            acc_avg = list()
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            total_batches = len(tta_loader)
            test_pred_real = []
            test_label_real = []
            if args.online:
                base_model = load_base_model(args, config, logger)
                optimizer = builder.build_opti_sche(base_model, config)[0]
                args.grad_steps = 1

            for idx, (data, labels) in enumerate(tta_loader):
                sample_acc = list()

                sample_loss = list()

                losses = AverageMeter(['Reconstruction Loss'])

                if not args.online:
                    base_model = load_base_model(args, config, logger)
                    optimizer = builder.build_opti_sche(base_model, config)[0]
                base_model.zero_grad()

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    test_pred = list()
                    test_label = list()

                    base_model.train()
                    if args.disable_bn_adaptation:  # disable statistical alignment
                        for m in base_model.modules():
                            if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                            nn.BatchNorm3d):
                                m.eval()
                    else:
                        pass
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:

                        points = [points for _ in range(args.batch_size_tta)]
                        points = torch.squeeze(torch.vstack(points))

                        loss_recon, loss_p_consistency, loss_regularize = base_model(points)
                        loss = loss_recon + (args.alpha * loss_regularize)  # + (0.0001 * loss_p_consistency)
                        loss = loss.mean()
                        loss.backward()
                        optimizer.step()
                        base_model.zero_grad()
                        optimizer.zero_grad()
                        sample_loss.append(loss.detach().cpu())

                        base_model.eval()
                        points = data.cuda()
                        labels = labels.cuda()
                        points = misc.fps(points, npoints)

                        logits = base_model.module.classification_only(points, only_unmasked=False)

                        target = labels.view(-1)
                        pred = logits.argmax(-1).view(-1)

                        test_pred.append(pred.detach())
                        test_label.append(target.detach())

                        test_pred_ = torch.cat(test_pred, dim=0)
                        test_label_ = torch.cat(test_label, dim=0)

                        acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                        sample_acc.append(acc.detach().cpu())

                    else:
                        continue

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample

                all_sample_loss.append(sample_loss)
                all_sample_acc.append(sample_acc)

                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                logits = base_model.module.classification_only(points, only_unmasked=False)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred_real.append(pred.detach())
                test_label_real.append(target.detach())

                if idx % 50 == 0:
                    test_pred_int = torch.cat(test_pred_real, dim=0)
                    test_label_int = torch.cat(test_label_real, dim=0)
                    acc = (test_pred_int == test_label_int).sum() / float(test_label_int.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

                    acc_avg.append(acc.cpu())

            Path(f'loss_standarad/').mkdir(exist_ok=True, parents=True)
            np.save(f'loss_standarad/{args.corruption}.npy', all_sample_loss)

            Path(f'acc_standarad/').mkdir(exist_ok=True, parents=True)
            np.save(f'acc_standarad/{args.corruption}.npy', all_sample_acc)

            test_pred_final = torch.cat(test_pred_real, dim=0)
            test_label_final = torch.cat(test_label_real, dim=0)


            acc = (test_pred_final == test_label_final).sum() / float(test_label_final.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def tta_dua(args, config, train_writer=None):
    dataset_name = args.dataset_name
    # assert args.tta
    assert dataset_name is not None
    # assert args.mask_ratio == 0.9

    if dataset_name == 'modelnet':
        config.model.cls_dim = 40
    elif dataset_name == 'scanobject_wbg':  # for with background
        config.model.cls_dim = 15
    elif dataset_name == 'scanobject_nbg':  # for no background
        config.model.cls_dim = 15
    elif dataset_name == 'partnet':
        config.model.cls_dim = 16
    elif dataset_name == 'shapenetcore':
        config.model.cls_dim = 55
    else:
        raise NotImplementedError

    args.disable_bn_adaptation = False

    args.batch_size_tta = 48
    args.batch_size = 1

    config.model.transformer_config.mask_ratio = args.mask_ratio  # overwrite the mask_ratio configuration parameter
    config.model.group_norm = args.group_norm
    npoints = config.npoints
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions):
            if args.corruption == 'clean':
                continue
                # raise NotImplementedError('Not possible to use tta with clean data, please modify the list above')

            if corr_id == 0:  # for saving results for easy copying to google sheet

                f_write, logtime = get_writer_to_all_result(args, config, custom_path='results_final_tta/')
                f_write.write(f'All Corruptions: {corruptions}' + '\n\n')
                f_write.write(f'TTA Results for Dataset: {dataset_name}' + '\n\n')
                f_write.write(f'Checkpoint Used: {args.ckpts}' + '\n\n')
            tta_loader = load_tta_dataset(args, config)
            test_pred = []
            test_label = []
            base_model = load_base_model(args, config, logger)

            for idx, (data, labels) in enumerate(tta_loader):
                base_model.train()

                # TTA Loop (for N grad steps)
                for grad_step in range(args.grad_steps):
                    if dataset_name == 'modelnet':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'shapenetcore':
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name in ['scanobject_wbg', 'scanobject_nbg']:
                        points = data.cuda()
                        points = misc.fps(points, npoints)
                    elif dataset_name == 'partnet':
                        points = data.cuda()
                    else:
                        raise NotImplementedError

                    # make a batch
                    if idx % args.stride_step == 0 or idx == len(tta_loader) - 1:
                        points = [points for _ in range(args.batch_size_tta)]
                        points = torch.squeeze(torch.vstack(points))

                        _ = base_model.module.classification_only(points,
                                                                  only_unmasked=True)  # only a forward pass through the encoder with BN in train mode
                        # loss=0
                    else:
                        continue

                    # print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},',
                    #           logger=logger)

                # now inferring on this one sample
                base_model.eval()
                points = data.cuda()
                labels = labels.cuda()
                points = misc.fps(points, npoints)

                logits = base_model.module.classification_only(points, only_unmasked=False)
                target = labels.view(-1)
                pred = logits.argmax(-1).view(-1)

                test_pred.append(pred.detach())
                test_label.append(target.detach())

                if idx % 100 == 0:
                    test_pred_ = torch.cat(test_pred, dim=0)
                    test_label_ = torch.cat(test_label, dim=0)

                    if args.distributed:
                        test_pred_ = dist_utils.gather_tensor(test_pred_, args)
                        test_label_ = dist_utils.gather_tensor(test_label_, args)

                    acc = (test_pred_ == test_label_).sum() / float(test_label_.size(0)) * 100.

                    print_log(f'\n\n\nIntermediate Accuracy - IDX {idx} - {acc:.1f}\n\n\n',
                              logger=logger)

            test_pred = torch.cat(test_pred, dim=0)
            test_label = torch.cat(test_label, dim=0)

            if args.distributed:
                test_pred = dist_utils.gather_tensor(test_pred, args)
                test_label = dist_utils.gather_tensor(test_label, args)

            acc = (test_pred == test_label).sum() / float(test_label.size(0)) * 100.
            print_log(f'\n\n######## Final Accuracy ::: {args.corruption} ::: {acc} ########\n\n',
                      logger=logger)
            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [acc]]) + '\n')
            f_write.flush()

            if corr_id == len(corruptions) - 1:
                f_write.close()

                print(f'Final Results Saved at:', os.path.join('results_final/', f'{logtime}_results.txt'))
                if train_writer is not None:
                    train_writer.close()


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def tta_partseg(args, config, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio
    seg_classes = config.seg_classes
    num_classes = config.model.num_classes

    test_metrics = {}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions_partnet):
            root = config.root_partseg

            shape_ious = {cat: [] for cat in seg_classes.keys()}

            print(f'Evaluating ::: {args.corruption} ::: Level ::: {args.severity}')

            if args.corruption != 'clean':
                root = os.path.join(root, f'{args.dataset_name}_c',
                                    f'{args.corruption}_{args.severity}')
            else:
                root = os.path.join(root, f'{args.dataset_name}_c',
                                    f'{args.corruption}')

            if corr_id == 0:
                res_dir_for_lazy_copying = 'tta_results_part_seg/'
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path=res_dir_for_lazy_copying)  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions_partnet}' + '\n\n')

            TEST_DATASET = tta_datasets.PartNormalDatasetSeg(root=root, npoints=config.npoint, split='test',
                                                             normal_channel=config.normal, debug=args.debug)
            tta_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

            total_batches = len(tta_loader)

            if args.online:
                base_model = load_base_model(args, config, logger, load_part_seg=True)
                optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for idx, (data, label, target) in enumerate(tta_loader):
                points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()
                losses = AverageMeter(['Reconstruction Loss'])
                if not args.online:
                    base_model = load_base_model(args, config, logger, load_part_seg=True)
                    optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)

                for grad_step in range(args.grad_steps):
                    # making a batch
                    input_points = [points for _ in range(48)]
                    input_points = torch.squeeze(torch.vstack(input_points))
                    loss = base_model(input_points, to_categorical(label, num_classes), tta=True)[
                        0]  # only take recon loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()
                    del input_points

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                with torch.no_grad():
                    base_model.eval()
                    points = data.float().cuda()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    seg_pred = base_model.module.classification_only(points, to_categorical(label, num_classes),
                                                                     only_unmasked=False)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                    if idx % 50 == 0:
                        all_shape_ious = []
                        for cat in shape_ious.keys():
                            for iou in shape_ious[cat]:
                                all_shape_ious.append(iou)
                        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                        instance_iou = test_metrics['inctance_avg_iou'] * 100
                        print_log(f'\n\n\nIntermediate Instance mIOU - IDX {idx} - {instance_iou:.1f}\n\n\n',
                                  logger=logger)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            instance_iou = test_metrics['inctance_avg_iou'] * 100

            print_log(f'{args.corruption} ::: Instance Avg IOU ::: {instance_iou}', logger=logger)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [instance_iou]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions_partnet) - 1:
                f_write.close()
                print(f'Final Results Saved at:',
                      os.path.join(f'{res_dir_for_lazy_copying}/', f'{logtime}_results.txt'))

            if train_writer is not None:
                train_writer.close()


def tta_shapenet(args, config, train_writer=None):
    config.model.transformer_config.mask_ratio = args.mask_ratio
    seg_classes = config.seg_classes
    num_classes = config.model.num_classes

    test_metrics = {}
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    logger = get_logger(args.log_name)

    for args.severity in level:
        for corr_id, args.corruption in enumerate(corruptions_h5):
            shape_ious = {cat: [] for cat in seg_classes.keys()}

            print(f'Evaluating ::: {args.corruption} ::: Level ::: {args.severity}')

            if corr_id == 0:
                res_dir_for_lazy_copying = 'tta_results_shape_net/'
                f_write, logtime = get_writer_to_all_result(args, config,
                                                            custom_path=res_dir_for_lazy_copying)  # for saving results for easy copying to google sheet
                f_write.write(f'All Corruptions: {corruptions_h5}' + '\n\n')

            TEST_DATASET = tta_datasets.ShapeNetC(args,
                                                  root='./data/shapenet_c')
            tta_loader = DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=10)

            total_batches = len(tta_loader)

            if args.online:
                base_model = load_base_model(args, config, logger, load_part_seg=True)
                optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            for idx, (data, label, target) in enumerate(tta_loader):
                points, label, target = data.float().cuda(), label.long().cuda(), target.long().cuda()
                losses = AverageMeter(['Reconstruction Loss'])
                if not args.online:
                    base_model = load_base_model(args, config, logger, load_part_seg=True)
                    optimizer = builder.build_opti_sche(base_model, config, tta_part_seg=True)[0]

                base_model.zero_grad()
                base_model.train()
                if args.disable_bn_adaptation:  # disable statistical alignment
                    for m in base_model.modules():
                        if isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.BatchNorm1d) or isinstance(m,
                                                                                                        nn.BatchNorm3d):
                            m.eval()
                else:
                    pass

                # TTA Loop (for N grad steps)

                for grad_step in range(args.grad_steps):
                    # making a batch
                    input_points = [points for _ in range(48)]
                    input_points = torch.squeeze(torch.vstack(input_points))
                    loss = base_model(input_points, to_categorical(label, num_classes), tta=True)[
                        0]  # only take recon loss
                    loss = loss.mean()
                    loss.backward()
                    optimizer.step()
                    base_model.zero_grad()
                    optimizer.zero_grad()
                    del input_points

                    if args.distributed:
                        loss = dist_utils.reduce_tensor(loss, args)
                        losses.update([loss.item() * 1000])
                    else:
                        losses.update([loss.item() * 1000])

                    print_log(f'[TEST - {args.corruption}], Sample - {idx} / {total_batches},'
                              f'GradStep - {grad_step} / {args.grad_steps},'
                              f'Reconstruction Loss {[l for l in losses.val()]}',
                              logger=logger)

                # now inferring on this one sample
                with torch.no_grad():
                    base_model.eval()
                    points = data.float().cuda()
                    cur_batch_size, NUM_POINT, _ = points.size()
                    seg_pred = base_model.module.classification_only(points, to_categorical(label, num_classes),
                                                                     only_unmasked=False)
                    cur_pred_val = seg_pred.cpu().data.numpy()
                    cur_pred_val_logits = cur_pred_val
                    cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                    target = target.cpu().data.numpy()

                    for i in range(cur_batch_size):
                        cat = seg_label_to_cat[target[i, 0]]
                        logits = cur_pred_val_logits[i, :, :]
                        cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                    for i in range(cur_batch_size):
                        segp = cur_pred_val[i, :]
                        segl = target[i, :]
                        cat = seg_label_to_cat[segl[0]]
                        part_ious = [0.0 for _ in range(len(seg_classes[cat]))]
                        for l in seg_classes[cat]:
                            if (np.sum(segl == l) == 0) and (
                                    np.sum(segp == l) == 0):  # part is not present, no prediction as well
                                part_ious[l - seg_classes[cat][0]] = 1.0
                            else:
                                part_ious[l - seg_classes[cat][0]] = np.sum((segl == l) & (segp == l)) / float(
                                    np.sum((segl == l) | (segp == l)))
                        shape_ious[cat].append(np.mean(part_ious))

                    if idx % 50 == 0:
                        all_shape_ious = []
                        for cat in shape_ious.keys():
                            for iou in shape_ious[cat]:
                                all_shape_ious.append(iou)
                        test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
                        instance_iou = test_metrics['inctance_avg_iou'] * 100
                        print_log(f'\n\n\nIntermediate Instance mIOU - IDX {idx} - {instance_iou:.1f}\n\n\n',
                                  logger=logger)

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)
            instance_iou = test_metrics['inctance_avg_iou'] * 100

            print_log(f'{args.corruption} ::: Instance Avg IOU ::: {instance_iou}', logger=logger)

            f_write.write(' '.join([str(round(float(xx), 3)) for xx in [instance_iou]]) + '\n')
            f_write.flush()
            if corr_id == len(corruptions_h5) - 1:
                f_write.close()
                print(f'Final Results Saved at:',
                      os.path.join(f'{res_dir_for_lazy_copying}/', f'{logtime}_results.txt'))

            if train_writer is not None:
                train_writer.close()
