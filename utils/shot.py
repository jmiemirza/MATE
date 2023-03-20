import torch.optim as optim
import torch
import torch.nn as nn
from scipy.spatial.distance import cdist
import numpy as np


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def obtain_shot_label(loader, ext, task_head, args, n_clips=None, c=None):
    start_test = True
    with torch.no_grad():
        # iter_test = iter(loader)
        for batch_idx, (inputs, labels) in enumerate(loader):
            inputs = inputs.cuda()
            if args.arch == 'tanet':
                actual_bz = inputs.shape[0]
                inputs = inputs.view(-1, 3, inputs.size(2), inputs.size(3))

                inputs = inputs.view(actual_bz * args.test_crops * n_clips,
                                     args.clip_length, 3, inputs.size(2), inputs.size(3))
                feas = ext(inputs)
                outputs = task_head(feas)
                outputs = torch.squeeze(outputs)
                outputs = outputs.reshape(actual_bz, args.test_crops * n_clips, -1).mean(1)

            elif args.arch == 'videoswintransformer':
                avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
                n = inputs.shape[0]
                n_views = inputs.shape[1]  # n_views   n_spatial_crops * n_temporal clips
                inputs = inputs.reshape(
                    (-1,) + inputs.shape[2:])  # (N, n_views, C, T, H, W) ->  (N * n_views, C, T, H, W)
                feas = ext(inputs.cuda())
                outputs = task_head(feas)
                feas = avg_pool(feas)
                feas = outputs.view(feas.shape[0], -1)
                bz = outputs.shape[0]
                cls_score = outputs.view(bz // n_views, n_views, -1)  # (bz, n_views, n_class)
                outputs = cls_score.mean(dim=1)

                # outputs = torch.squeeze(vid_cls_score)


            else:
                inputs = inputs.reshape(
                    (-1,) + inputs.shape[2:])
                feas = ext(inputs)
                outputs = task_head(feas)
                outputs = torch.squeeze(outputs)

            if start_test:
                all_fea = feas.float().cpu()
                all_output = outputs.float().cpu()
                all_label = labels.float()
                start_test = False
            else:
                all_fea = torch.cat((all_fea, feas.float().cpu()), 0)
                all_output = torch.cat((all_output, outputs.float().cpu()), 0)
                all_label = torch.cat((all_label, labels.float()), 0)

    all_output = nn.Softmax(dim=1)(all_output)
    _, predict = torch.max(all_output, 1)
    accuracy = torch.sum(torch.squeeze(predict).float() == all_label).item() / float(all_label.size()[0])
    a = torch.ones(all_fea.size(0), 1)
    all_fea = torch.squeeze(all_fea)
    all_fea = torch.cat((all_fea, torch.ones(all_fea.size(0), 1)), 1)  # (bz, 1024 + 1) add one more dimension of  ones
    all_fea = (all_fea.t() / torch.norm(all_fea, p=2, dim=1)).t()  # (bz, 1025), normalize along feature dimension
    all_fea = all_fea.float().cpu().numpy()

    K = all_output.size(1)
    aff = all_output.float().cpu().numpy()
    initc = aff.transpose().dot(all_fea)
    initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
    dd = cdist(all_fea, initc, 'cosine')
    pred_label = dd.argmin(axis=1)
    acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    for round in range(1):  # todo udpate the pseudo labels once
        aff = np.eye(K)[pred_label]
        initc = aff.transpose().dot(all_fea)
        initc = initc / (1e-8 + aff.sum(axis=0)[:, None])
        dd = cdist(all_fea, initc, 'cosine')
        pred_label = dd.argmin(axis=1)
        acc = np.sum(pred_label == all_label.float().numpy()) / len(all_fea)

    log_str = 'Accuracy = {:.2f}% -> {:.2f}%'.format(accuracy * 100, acc * 100)
    print(log_str + '\n')
    return pred_label.astype('int')


def Entropy(input_):
    bs = input_.size(0)
    entropy = -input_ * torch.log(input_ + 1e-5)
    entropy = torch.sum(entropy, dim=1)
    return entropy


def lr_scheduler(optimizer, iter_num, max_iter, gamma=10, power=0.75):
    decay = (1 + gamma * iter_num / max_iter) ** (-power)
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr0'] * decay
        param_group['weight_decay'] = 1e-3
        param_group['momentum'] = 0.9
        param_group['nesterov'] = True
    return optimizer


def op_copy(optimizer):
    for param_group in optimizer.param_groups:
        param_group['lr0'] = param_group['lr']
    return optimizer


def configure_shot(net, logger, args):
    logger.debug('---- Configuring SHOT ----')
    if args.arch == 'tanet':
        classifier = net.module.new_fc
        ext = net
        ext.module.new_fc = nn.Identity()
        for k, v in classifier.named_parameters():
            v.requires_grad = False

    elif args.arch == 'videoswintransformer':
        classifier = net.module.cls_head
        ext = net.module.backbone
        # ext.module.cls_head = nn.Identity()

        for k, v in classifier.named_parameters():
            v.requires_grad = False

    else:
        for k, v in net.named_parameters():
            if 'logits' in k:
                v.requires_grad = False  # freeze the  classifier
        classifier = nn.Sequential(*list(net.module.logits.children()))
        ext = list(net.module.children())[3:] + list(net.module.children())[:2]
        ext = nn.Sequential(*ext)

    optimizer = optim.SGD(ext.parameters(), lr=args.lr, momentum=0.9)
    return optimizer, classifier, ext


def safe_log(x, ver):
    if ver == 1:
        return torch.log(x + 1e-5)
    elif ver == 2:
        return torch.log(x + 1e-7)
    elif ver == 3:
        return torch.clamp(torch.log(x), min=-100)
    else:
        raise ValueError("safe_log version is not properly defined !!!")


def softmax_diversity_regularizer(x):
    x2 = x.softmax(-1).mean(0)  # [b, c] -> [c]
    return (x2 * safe_log(x2, ver=3)).sum()
