import torch.nn as nn
import torch
from knn_cuda import KNN
from utils import misc


def softmax_entropy(x: torch.Tensor) -> torch.Tensor:
    """Entropy of softmax distribution from logits."""
    return -(x.softmax(1) * x.log_softmax(1)).sum(1)


def get_cls_ext(net):
    ext = net.module.MAE_encoder
    classifier = net.module.class_head
    return ext, classifier


class Group(nn.Module):
    def __init__(self, num_group, group_size):
        super().__init__()
        self.num_group = num_group
        self.group_size = group_size
        self.knn = KNN(k=self.group_size, transpose_mode=True)

    def forward(self, xyz):
        '''
            input: B N 3    N  number of points ,  M is number of centers (number of groups )
            ---------------------------
            output: B G M 3     G is group size 32
            center : B G 3
        '''
        batch_size, num_points, _ = xyz.shape
        # fps the centers out
        center = misc.fps(xyz, self.num_group)  # B G 3    sample 128 center points from 2048 points
        # knn to get the neighborhood
        _, idx = self.knn(xyz,
                          center)  # B G M,   kNN samples for each center  idx (B, M, G)   every center has G (group size) NN points
        assert idx.size(1) == self.num_group
        assert idx.size(2) == self.group_size
        idx_base = torch.arange(0, batch_size, device=xyz.device).view(-1, 1, 1) * num_points  # idx_base  (8, 1, 1)
        idx = idx + idx_base  # for  batch 0 offset 0,   batch 1 ~7,  offset  1*2048
        idx = idx.view(-1)
        neighborhood = xyz.view(batch_size * num_points, -1)[idx,
                       :]  # (8, 2048, 3) -> (8*2048, 3)   # todo sampling the neighborhoold points for each center in each batch
        neighborhood = neighborhood.view(batch_size, self.num_group, self.group_size,
                                         3).contiguous()  # (8, 128, 32, 3)  128 groups, each group has 32 points,
        # normalize
        neighborhood = neighborhood - center.unsqueeze(2)
        return neighborhood, center


class T3A(nn.Module):
    """
    Test Time Template Adjustments (T3A)

    """

    def __init__(self, args, ext, classifier, config):
        super().__init__()
        self.args = args
        self.model = ext
        self.classifier = classifier

        self.warmup_supports = self.classifier[8].weight.data
        warmup_prob = self.classifier[8](self.warmup_supports)

        self.warmup_ent = softmax_entropy(warmup_prob)
        self.warmup_labels = torch.nn.functional.one_hot(warmup_prob.argmax(1), num_classes=args.num_classes).float()

        self.supports = self.warmup_supports.data

        self.labels = self.warmup_labels.data
        self.ent = self.warmup_ent.data

        self.filter_K = args.t3a_filter_k
        self.num_classes = args.num_classes
        self.softmax = torch.nn.Softmax(-1)
        self.num_group = config.model.num_group
        self.group_size = config.model.group_size

        self.group_divider = Group(num_group=self.num_group, group_size=self.group_size)

    def forward(self, x):
        with torch.no_grad():
            n, c = self.group_divider(x)
            z = self.model(n, c, only_unmasked=True)[0]
            z = torch.cat([z[:, 0], z[:, 1:].max(1)[0]], dim=-1)

            p = self.classifier(z)

        yhat = torch.nn.functional.one_hot(p.argmax(1), num_classes=self.num_classes).float()
        ent = softmax_entropy(p)
        # prediction
        self.supports = self.supports.to(z.device)
        z = self.classifier[0](z)
        z = self.classifier[1](z)
        z = self.classifier[2](z)
        z = self.classifier[3](z)
        z = self.classifier[4](z)
        z = self.classifier[5](z)
        z = self.classifier[6](z)
        z = self.classifier[7](z)
        self.labels = self.labels.to(z.device)
        self.ent = self.ent.to(z.device)

        self.supports = torch.cat([self.supports, z])
        self.labels = torch.cat([self.labels, yhat])
        self.ent = torch.cat([self.ent, ent])

        supports, labels = self.select_supports()
        supports = torch.nn.functional.normalize(supports, dim=1)
        weights = (supports.T @ (labels))
        return z @ torch.nn.functional.normalize(weights, dim=0)

    def select_supports(self):
        ent_s = self.ent
        y_hat = self.labels.argmax(dim=1).long()
        filter_K = self.filter_K
        if filter_K == -1:
            indices = torch.LongTensor(list(range(len(ent_s))))

        indices = []
        indices1 = torch.LongTensor(list(range(len(ent_s))))
        for i in range(self.num_classes):
            _, indices2 = torch.sort(ent_s[y_hat == i])
            indices.append(indices1[y_hat == i][indices2][:filter_K])
        indices = torch.cat(indices)

        self.supports = self.supports[indices]
        self.labels = self.labels[indices]
        self.ent = self.ent[indices]

        return self.supports, self.labels


def average_clips(cls_score, num_segs=1):
    bz = cls_score.shape[0]
    cls_score = cls_score.view(bz // num_segs, num_segs, -1)  # (bz, n_views, n_class)
    vid_cls_score = cls_score.mean(dim=1)
    return vid_cls_score, cls_score
