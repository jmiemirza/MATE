import os
import argparse
from pathlib import Path


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--mask_ratio', type=float, default= 0.9)
    parser.add_argument('--alpha', type=float, default= 0.0)
    parser.add_argument('--batch_size_tta', type=int, default=48)
    parser.add_argument('--stride_step', type=int, default=1)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--grad_steps', type=int, default=20)
    parser.add_argument('--disable_bn_adaptation', action='store_true', help='to disable bn_for adaptation')
    parser.add_argument('--online', action='store_true', help='online-adapt')
    parser.add_argument('--visualize_data', action='store_true', help='image creation')
    parser.add_argument('--ckpts', type=str, default=None, help='test used ckpt path')
    parser.add_argument('--config', type=str, help='yaml config file')
    parser.add_argument('--group_norm', action='store_true', help='If Group Norm shall be used instead of Batch Norm')
    parser.add_argument('--test_source', action='store_true')
    parser.add_argument('--tta', action='store_true', default=False, help='test mode for test-time adaptation')
    parser.add_argument('--tta_seg', action='store_true', default=False, help='test mode for test-time adaptation for part segmentation')
    parser.add_argument('--debug', action='store_true', default=False, help='only load small number of samples')
    parser.add_argument('--partnet_cls', action='store_true', default=False, help='train partnet for obj classification task')
    parser.add_argument('--jt', action='store_true', default=False, help='train model with JT')
    parser.add_argument('--only_cls', action='store_true', default=False, help='train model only for cls task / without JT')
    parser.add_argument('--train_aug', action='store_true', default=False, help='weather to use augmentations for train/test')
    parser.add_argument('--dataset_name', type=str, default=None, help='which dataset to use for tta', choices=['modelnet', 'scanobject', 'scanobject_nbg', 'partnet', 'shapenetcore', 'shapenet'])
    parser.add_argument('--cyclic', action='store_true', default=False, help='get cls loss with 100% tokens and recon loss with 10% - used for joint pretraining!!!')
    parser.add_argument('--tta_rot', action='store_true', default=False, help='do tta for rotnet')
    parser.add_argument('--train_tttrot', action='store_true', default=False, help='train ttt rotnet')
    parser.add_argument('--only_unmasked', action='store_true', default=False, help='weather to use 100% tokens for classification or not')
    parser.add_argument('--test_source_rotnet', action='store_true', default=False, help='train-ttt-rot')
    parser.add_argument('--shuffle', action='store_true', default=True, help='shuffle data for adaptation')
    parser.add_argument('--tta_dua', action='store_true', default=False, help='for running adaptatation with DUA')
    parser.add_argument('--baseline', type=str, default='tent', help='run adaptive baselines, choose from TENT, DUA, T3A, SHOT')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--num_workers', type=int, default=8)
    # seed
    parser.add_argument('--seed', type=int, default=0, help='random seed')
    parser.add_argument(
        '--deterministic',
        action='store_true',
        help='whether to set deterministic options for CUDNN backend.')
    # bn
    parser.add_argument(
        '--sync_bn',
        action='store_true',
        default=False,
        help='whether to use sync bn')
    # some args
    parser.add_argument('--exp_name', type=str, default='default', help='experiment name')
    parser.add_argument('--loss', type=str, default='cd1', help='loss name')
    parser.add_argument('--start_ckpts', type=str, default=None, help='reload used ckpt path')
    parser.add_argument('--val_freq', type=int, default=1, help='test freq')
    parser.add_argument(
        '--vote',
        action='store_true',
        default=False,
        help='vote acc')
    parser.add_argument(
        '--resume',
        action='store_true',
        default=False,
        help='autoresume training (interrupted by accident)')
    parser.add_argument(
        '--test',
        action='store_true',
        default=False,
        help='test mode for certain ckpt')
    parser.add_argument(
        '--finetune_model',
        action='store_true',
        default=False,
        help='finetune modelnet with pretrained weight')
    parser.add_argument(
        '--scratch_model',
        action='store_true',
        default=False,
        help='training modelnet from scratch')
    parser.add_argument(
        '--mode',
        choices=['easy', 'median', 'hard', None],
        default=None,
        help='difficulty mode for shapenet')
    parser.add_argument(
        '--way', type=int, default=-1)
    parser.add_argument(
        '--shot', type=int, default=-1)
    parser.add_argument(
        '--fold', type=int, default=-1)

    args = parser.parse_args()

    if args.test and args.resume:
        raise ValueError(
            '--test and --resume cannot be both activate')

    if args.resume and args.start_ckpts is not None:
        raise ValueError(
            '--resume and --start_ckpts cannot be both activate')

    if args.test and args.ckpts is None:
        raise ValueError(
            'ckpts shouldnt be None while test mode')

    if args.finetune_model and args.ckpts is None:
        print(
            'training from scratch')

    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)

    if args.test:
        args.exp_name = 'test_' + args.exp_name
    if args.mode is not None:
        args.exp_name = args.exp_name + '_' + args.mode
    args.experiment_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem,
                                        args.exp_name)
    args.tfboard_path = os.path.join('./experiments', Path(args.config).stem, Path(args.config).parent.stem, 'TFBoard',
                                     args.exp_name)
    args.log_name = Path(args.config).stem
    create_experiment_dir(args)
    return args


def create_experiment_dir(args):
    if not os.path.exists(args.experiment_path):
        os.makedirs(args.experiment_path)
        print('Create experiment path successfully at %s' % args.experiment_path)
    if not os.path.exists(args.tfboard_path):
        os.makedirs(args.tfboard_path)
        print('Create TFBoard path successfully at %s' % args.tfboard_path)
