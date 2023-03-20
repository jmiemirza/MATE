"""
Author: Benny
Date: Nov 2019
"""
import argparse
import os
import torch
import datetime
import logging
import sys
import importlib
import shutil
import segmentation.provider as provider
import numpy as np
import torch.optim as optim
from timm.scheduler import CosineLRScheduler
from pathlib import Path
from tqdm import tqdm
from segmentation.dataset import PartNormalDataset
from utils.config import *
from tools import builder
from torch.utils.data import DataLoader
import torch.nn as nn


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True


def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def parse_args():
    parser = argparse.ArgumentParser('Model')
    # parser.add_argument('--model', type=str, default='pt', help='model name')
    # parser.add_argument('--batch_size', type=int, default=16, help='batch Size during training')
    parser.add_argument('--epoch', default=300, type=int, help='epoch to run')
    parser.add_argument('--warmup_epoch', default=10, type=int, help='warmup epoch')
    # parser.add_argument('--learning_rate', default=0.0002, type=float, help='initial learning rate')
    parser.add_argument('--gpu', type=str, default='1', help='specify GPU devices')
    # parser.add_argument('--optimizer', type=str, default='AdamW', help='Adam or SGD')
    parser.add_argument('--experiment_path', type=str, default='./exp_part_seg', help='log path')
    # parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay')
    # parser.add_argument('--npoint', type=int, default=2048, help='point Number')
    # parser.add_argument('--normal', action='store_true', default=False, help='use normals')
    # parser.add_argument('--step_size', type=int, default=20, help='decay step for lr decay')
    # parser.add_argument('--lr_decay', type=float, default=0.5, help='decay rate for lr decay')
    parser.add_argument('--ckpts', type=str, default=None, help='ckpts')
    parser.add_argument('--config', type=str, default='./cfgs/pretrain_partseg.yaml', help='config')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--distributed', action='store_true')

    return parser.parse_args()


def main(args, config, logger):

    seg_classes = config.seg_classes
    seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
    for cat in seg_classes.keys():
        for label in seg_classes[cat]:
            seg_label_to_cat[label] = cat

    def log_string(str):
        logger.info(str)
        print(str)

    base_model = builder.model_builder(config.model)
    if args.ckpts is not None:
        base_model.load_model_from_ckpt(args.ckpts)
    else:
        log_string('Training from Scratch...')
    if args.gpu:
        base_model.to(args.local_rank)

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

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    exp_dir = Path('./log/')
    exp_dir.mkdir(exist_ok=True)
    exp_dir = exp_dir.joinpath('part_seg')
    exp_dir.mkdir(exist_ok=True)
    if args.experiment_path is None:
        exp_dir = exp_dir.joinpath(timestr)
    else:
        exp_dir = exp_dir.joinpath(args.experiment_path)
    exp_dir.mkdir(exist_ok=True)
    checkpoints_dir = exp_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = exp_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    # logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, config.model.NAME))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    log_string('CONFIGURATION ...')
    log_string(config)

    # root = args.root

    TRAIN_DATASET = PartNormalDataset(root=config.root, npoints=config.npoint, split='trainval', normal_channel=config.normal)
    trainDataLoader = DataLoader(TRAIN_DATASET, batch_size=config.batch_size, shuffle=True, num_workers=10, drop_last=True)
    TEST_DATASET = PartNormalDataset(root=config.root, npoints=config.npoint, split='test', normal_channel=config.normal)
    testDataLoader = DataLoader(TEST_DATASET, batch_size=config.batch_size, shuffle=False, num_workers=10)

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = config.num_classes
    num_part = config.num_part

    '''MODEL LOADING'''
    # MODEL = importlib.import_module(args.model)
    # shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    # classifier = MODEL.get_model(num_part).cuda()
    # criterion = MODEL.get_loss().cuda()
    # classifier.apply(inplace_relu)
    # print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))
    start_epoch = 0

    # if args.ckpts is not None:
    #     classifier.load_model_from_ckpt(args.ckpts)

## we use adamw and cosine scheduler
    # def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    #     decay = []
    #     no_decay = []
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             continue  # frozen weights
    #         if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
    #                     # print(name)
    #             no_decay.append(param)
    #         else:
    #             decay.append(param)
    #     return [
    #                 {'params': no_decay, 'weight_decay': 0.},
    #                 {'params': decay, 'weight_decay': weight_decay}]

    # param_groups = add_weight_decay(base_model, weight_decay=0.05)
    # optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )
    optimizer, scheduler = builder.build_opti_sche(base_model, config)

    # scheduler = CosineLRScheduler(optimizer,
    #                               t_initial=args.epoch,
    #                               t_mul=1,
    #                               lr_min=1e-6,
    #                               decay_rate=0.1,
    #                               warmup_lr_init=1e-6,
    #                               warmup_t=args.warmup_epoch,
    #                               cycle_limit=1,
    #                               t_in_epochs=True)

    best_acc = 0
    global_epoch = 0
    best_class_avg_iou = 0
    best_inctance_avg_iou = 0

    base_model.zero_grad()
    for epoch in range(start_epoch, args.epoch):
        mean_correct = []

        log_string('Epoch %d (%d/%s):' % (global_epoch + 1, epoch + 1, args.epoch))
        '''Adjust learning rate and BN momentum'''

        # classifier = classifier.train()
        loss_batch = []
        num_iter = 0
        '''learning one epoch'''
        base_model.train()
        for i, (points, label, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            num_iter += 1
            points = points.data.numpy()
            points[:, :, 0:3] = provider.random_scale_point_cloud(points[:, :, 0:3])
            points[:, :, 0:3] = provider.shift_point_cloud(points[:, :, 0:3])
            points = torch.Tensor(points)
            points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
            # points = points.transpose(2, 1)

            loss_recon, seg_pred = base_model(points, to_categorical(label, num_classes))
            # seg_pred = seg_pred.contiguous().view(-1, num_part)
            # target = target.view(-1, 1)[:, 0]
            # pred_choice = seg_pred.data.max(1)[1]
            #
            # correct = pred_choice.eq(target.data).cpu().sum()
            loss_seg, acc_current = base_model.modules.get_loss_acc(seg_pred, target)
            mean_correct.append(acc_current)
            loss = loss_recon + loss_seg
            # loss = criterion(seg_pred, target)
            loss.backward()
            optimizer.step()
            loss_batch.append(loss.detach().cpu())

            torch.nn.utils.clip_grad_norm_(base_model.parameters(), 10, norm_type=2)

            # num_iter = 0
            optimizer.step()
            optimizer.zero_grad()
            base_model.zero_grad()

        if isinstance(scheduler, list):
            for item in scheduler:
                item.step(epoch)
        else:
            scheduler.step(epoch)

        train_instance_acc = np.mean(mean_correct)
        loss1 = np.mean(loss_batch)
        log_string('Train accuracy is: %.5f' % train_instance_acc)
        log_string('Train loss: %.5f' % loss1)
        log_string('lr: %.6f' % optimizer.param_groups[0]['lr'])

        with torch.no_grad():
            test_metrics = {}
            total_correct = 0
            total_seen = 0
            total_seen_class = [0 for _ in range(num_part)]
            total_correct_class = [0 for _ in range(num_part)]
            shape_ious = {cat: [] for cat in seg_classes.keys()}
            seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}

            for cat in seg_classes.keys():
                for label in seg_classes[cat]:
                    seg_label_to_cat[label] = cat

            # classifier = classifier.eval()
            base_model.eval()
            for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                cur_batch_size, NUM_POINT, _ = points.size()
                points, label, target = points.float().cuda(), label.long().cuda(), target.long().cuda()
                seg_pred = base_model.modules.classification_only(points, to_categorical(label, num_classes), only_unmasked=False)
                cur_pred_val = seg_pred.cpu().data.numpy()
                cur_pred_val_logits = cur_pred_val
                cur_pred_val = np.zeros((cur_batch_size, NUM_POINT)).astype(np.int32)
                target = target.cpu().data.numpy()

                for i in range(cur_batch_size):
                    cat = seg_label_to_cat[target[i, 0]]
                    logits = cur_pred_val_logits[i, :, :]
                    cur_pred_val[i, :] = np.argmax(logits[:, seg_classes[cat]], 1) + seg_classes[cat][0]

                correct = np.sum(cur_pred_val == target)
                total_correct += correct
                total_seen += (cur_batch_size * NUM_POINT)

                for l in range(num_part):
                    total_seen_class[l] += np.sum(target == l)
                    total_correct_class[l] += (np.sum((cur_pred_val == l) & (target == l)))

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

            all_shape_ious = []
            for cat in shape_ious.keys():
                for iou in shape_ious[cat]:
                    all_shape_ious.append(iou)
                shape_ious[cat] = np.mean(shape_ious[cat])
            mean_shape_ious = np.mean(list(shape_ious.values()))
            test_metrics['accuracy'] = total_correct / float(total_seen)
            test_metrics['class_avg_accuracy'] = np.mean(
                np.array(total_correct_class) / np.array(total_seen_class, dtype=np.float))
            for cat in sorted(shape_ious.keys()):
                log_string('eval mIoU of %s %f' % (cat + ' ' * (14 - len(cat)), shape_ious[cat]))
            test_metrics['class_avg_iou'] = mean_shape_ious
            test_metrics['inctance_avg_iou'] = np.mean(all_shape_ious)

        log_string('Epoch %d test Accuracy: %f  Class avg mIOU: %f   Inctance avg mIOU: %f' % (
            epoch + 1, test_metrics['accuracy'], test_metrics['class_avg_iou'], test_metrics['inctance_avg_iou']))
        if (test_metrics['inctance_avg_iou'] >= best_inctance_avg_iou):
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/best_model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'train_acc': train_instance_acc,
                'test_acc': test_metrics['accuracy'],
                'class_avg_iou': test_metrics['class_avg_iou'],
                'inctance_avg_iou': test_metrics['inctance_avg_iou'],
                'model_state_dict': base_model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']
        if test_metrics['class_avg_iou'] > best_class_avg_iou:
            best_class_avg_iou = test_metrics['class_avg_iou']
        if test_metrics['inctance_avg_iou'] > best_inctance_avg_iou:
            best_inctance_avg_iou = test_metrics['inctance_avg_iou']
        log_string('Best accuracy is: %.5f' % best_acc)
        log_string('Best class avg mIOU is: %.5f' % best_class_avg_iou)
        log_string('Best inctance avg mIOU is: %.5f' % best_inctance_avg_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    logger = logging.getLogger("Model")

    config = get_config(args, logger=logger)

    main(args, config, logger)