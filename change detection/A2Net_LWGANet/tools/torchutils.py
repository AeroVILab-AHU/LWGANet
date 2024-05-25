import torch
from torch.optim import lr_scheduler
from torch.utils.data import Subset
import torch.nn.functional as F
import numpy as np
import math
import random
import os
from torch.nn import MaxPool1d, AvgPool1d
from torch import Tensor
from typing import Iterable, Set, Tuple

__all__ = ['cls_accuracy']


def visualize_imgs(*imgs):
    import matplotlib.pyplot as plt
    nums = len(imgs)
    if nums > 1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(imgs):
            axs[i].imshow(image, cmap='jet')
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(imgs):
            ax.imshow(image, cmap='jet')
        plt.show()
    plt.show()


def minmax(tensor):
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1] * shape[-2]])
    min_, _ = tensor.min(-1, keepdim=True)
    max_, _ = tensor.max(-1, keepdim=True)
    return min_, max_


def norm_tensor(tensor, min_=None, max_=None, mode='minmax'):
    assert tensor.ndim >= 2
    shape = tensor.shape
    tensor = tensor.view([*shape[:-2], shape[-1] * shape[-2]])
    if mode == 'minmax':
        if min_ is None:
            min_, _ = tensor.min(-1, keepdim=True)
        if max_ is None:
            max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'thres':
        N = tensor.shape[-1]
        thres_a = 0.001
        top_k = round(thres_a * N)
        max_ = tensor.topk(top_k, dim=-1, largest=True)[0][..., -1]
        max_ = max_.unsqueeze(-1)
        min_ = tensor.topk(top_k, dim=-1, largest=False)[0][..., -1]
        min_ = min_.unsqueeze(-1)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)

    elif mode == 'std':
        mean, std = torch.std_mean(tensor, [-1], keepdim=True)
        tensor = (tensor - mean) / std
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    elif mode == 'exp':
        tai = 1
        tensor = torch.nn.functional.softmax(tensor / tai, dim=-1, )
        min_, _ = tensor.min(-1, keepdim=True)
        max_, _ = tensor.max(-1, keepdim=True)
        tensor = (tensor - min_) / (max_ - min_ + 0.00000000001)
    else:
        raise NotImplementedError
    tensor = torch.clamp(tensor, 0, 1)
    return tensor.view(shape)


def visulize_features(features, normalize=False):
    from torchvision.utils import make_grid
    assert features.ndim == 4
    b, c, h, w = features.shape
    features = features.view((b * c, 1, h, w))
    if normalize:
        features = norm_tensor(features)
    grid = make_grid(features)
    visualize_tensors(grid)


def visualize_tensors(*tensors):
    import matplotlib.pyplot as plt
    # from misc.torchutils import tensor2np
    images = []
    for tensor in tensors:
        assert tensor.ndim == 3 or tensor.ndim == 2
        if tensor.ndim == 3:
            assert tensor.shape[0] == 1 or tensor.shape[0] == 3
        images.append(tensor2np(tensor))
    nums = len(images)
    if nums > 1:
        fig, axs = plt.subplots(1, nums)
        for i, image in enumerate(images):
            axs[i].imshow(image, cmap='jet')
        plt.show()
    elif nums == 1:
        fig, ax = plt.subplots(1, nums)
        for i, image in enumerate(images):
            ax.imshow(image, cmap='jet')
        plt.show()


def np_to_tensor(image):
    """
    input: nd.array: H*W*C/H*W
    """
    if isinstance(image, torch.Tensor):
        return image
    elif isinstance(image, np.ndarray):
        if image.ndim == 3:
            if image.shape[2] == 3:
                image = np.transpose(image, [2, 0, 1])
        elif image.ndim == 2:
            image = np.newaxis(image, 0)
        image = torch.from_numpy(image)
        return image.unsqueeze(0)


def seed_torch(seed=2019):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def simplex(t: Tensor, axis=1) -> bool:
    _sum = t.sum(axis).type(torch.float32)
    _ones = torch.ones_like(_sum, dtype=torch.float32)
    return torch.allclose(_sum, _ones)


# Assert utils
def uniq(a: Tensor) -> Set:
    return set(torch.unique(a.cpu()).numpy())


def sset(a: Tensor, sub: Iterable) -> bool:
    return uniq(a).issubset(sub)


def eq(a: Tensor, b) -> bool:
    return torch.eq(a, b).all()


def one_hot(t: Tensor, axis=1) -> bool:
    return simplex(t, axis) and sset(t, [0, 1])


def class2one_hot(seg: Tensor, C: int) -> Tensor:
    if len(seg.shape) == 2:  # Only w, h, used by the dataloader
        seg = seg.unsqueeze(dim=0)
    assert sset(seg, list(range(C)))

    b, w, h = seg.shape  # type: Tuple[int, int, int]

    res = torch.stack([seg == c for c in range(C)], dim=1).type(torch.int32)
    assert res.shape == (b, C, w, h)
    assert one_hot(res)

    return res


class ChannelMaxPool(MaxPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.max_pool1d(input, self.kernel_size, self.stride,
                              self.padding, self.dilation, self.ceil_mode,
                              self.return_indices)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


class ChannelAvePool(AvgPool1d):
    def forward(self, input):
        n, c, w, h = input.size()
        input = input.view(n, c, w * h).permute(0, 2, 1)
        pooled = F.avg_pool1d(input, self.kernel_size, self.stride,
                              self.padding)
        _, _, c = pooled.size()
        pooled = pooled.permute(0, 2, 1)
        return pooled.view(n, c, w, h)


def cross_entropy(input, target, weight=None, reduction='mean', ignore_index=255):
    """
    logSoftmax_with_loss
    :param input: torch.Tensor, N*C*H*W
    :param target: torch.Tensor, N*1*H*W,/ N*H*W
    :param weight: torch.Tensor, C
    :return: torch.Tensor [0]
    """
    target = target.long()
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    return F.cross_entropy(input=input, target=target, weight=weight,
                           ignore_index=ignore_index, reduction=reduction)


def balanced_cross_entropy(input, target, weight=None, ignore_index=255):
    if target.dim() == 4:
        target = torch.squeeze(target, dim=1)
    if input.shape[-1] != target.shape[-1]:
        input = F.interpolate(input, size=target.shape[1:], mode='bilinear', align_corners=True)

    pos = (target == 1).float()
    neg = (target == 0).float()
    pos_num = torch.sum(pos) + 0.0000001
    neg_num = torch.sum(neg) + 0.0000001
    target_pos = target.float()
    target_pos[target_pos != 1] = ignore_index
    target_neg = target.float()
    target_neg[target_neg != 0] = ignore_index

    loss_pos = cross_entropy(input, target_pos, weight=weight, reduction='sum', ignore_index=ignore_index)
    loss_neg = cross_entropy(input, target_neg, weight=weight, reduction='sum', ignore_index=ignore_index)
    loss = 0.5 * loss_pos / pos_num + 0.5 * loss_neg / neg_num
    return loss


def get_scheduler(optimizer, opt):
    """Return a learning rate scheduler
    """
    if opt.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + opt.epoch_count - opt.niter) / float(opt.niter_decay + 1)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'poly':
        max_step = opt.niter + opt.niter_decay
        power = 0.9

        def lambda_rule(epoch):
            current_step = epoch + opt.epoch_count
            lr_l = (1.0 - current_step / (max_step + 1)) ** float(power)
            return lr_l

        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule)
    elif opt.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=opt.lr_decay_iters, gamma=0.1)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', opt.lr_policy)
    return scheduler


def mul_cls_acc(preds, targets, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        bs, C = targets.shape
        _, pred = preds.topk(maxk, 1, True, True)
        pred += 1
        correct = torch.zeros([bs, maxk]).long()
        if preds.device != torch.device(type='cpu'):
            correct = correct.cuda()
        for i in range(C):
            label = i + 1
            target = targets[:, i] * label
            correct = correct + pred.eq(target.view(-1, 1).expand_as(pred)).long()
        n = (targets == 1).long().sum(1)
        # print(n)
        res = []
        for k in topk:
            acc_k = correct[:, :k].sum(1).float() / n.float()
            acc_k = acc_k.sum() / bs
            res.append(acc_k)
    return res


def cls_accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


class PolyOptimizer(torch.optim.SGD):

    def __init__(self, params, lr, weight_decay, max_step, init_step=0, momentum=0.9):
        super().__init__(params, lr, weight_decay)

        self.global_step = init_step
        print(self.global_step)
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.global_step += 1


class PolyAdamOptimizer(torch.optim.Adam):
    def __init__(self, params, lr, betas, max_step, momentum=0.9):
        super().__init__(params, lr, betas)

        self.global_step = 0
        self.max_step = max_step
        self.momentum = momentum

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.global_step < self.max_step:
            lr_mult = (1 - self.global_step / self.max_step) ** self.momentum

            for i in range(len(self.param_groups)):
                self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)
        self.global_step += 1


class SGDROptimizer(torch.optim.SGD):

    def __init__(self, params, steps_per_epoch, lr=0, weight_decay=0, epoch_start=1, restart_mult=2):
        super().__init__(params, lr, weight_decay)

        self.global_step = 0
        self.local_step = 0
        self.total_restart = 0

        self.max_step = steps_per_epoch * epoch_start
        self.restart_mult = restart_mult

        self.__initial_lr = [group['lr'] for group in self.param_groups]

    def step(self, closure=None):

        if self.local_step >= self.max_step:
            self.local_step = 0
            self.max_step *= self.restart_mult
            self.total_restart += 1

        lr_mult = (1 + math.cos(math.pi * self.local_step / self.max_step)) / 2 / (self.total_restart + 1)

        for i in range(len(self.param_groups)):
            self.param_groups[i]['lr'] = self.__initial_lr[i] * lr_mult

        super().step(closure)

        self.local_step += 1
        self.global_step += 1


def split_dataset(dataset, n_splits):
    return [Subset(dataset, np.arange(i, len(dataset), n_splits)) for i in range(n_splits)]


def gap2d(x, keepdims=False):
    out = torch.mean(x.view(x.size(0), x.size(1), -1), -1)
    if keepdims:
        out = out.view(out.size(0), out.size(1), 1, 1)

    return out


def decode_seg(label_mask, toTensor=False):
    """
    :param label_mask: mask (np.ndarray): (M, N)/  tensor: N*C*H*W
    :return: color label: (M, N, 3),
    """
    if not isinstance(label_mask, np.ndarray):
        if isinstance(label_mask, torch.Tensor):  # get the data from a variable
            image_tensor = label_mask.data
        else:
            return label_mask
        label_mask = image_tensor[0][0].cpu().numpy()

    rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3), dtype=np.float)
    r = label_mask % 6
    g = (label_mask % 36) // 6
    b = label_mask // 36
    rgb[:, :, 0] = r / 6
    rgb[:, :, 1] = g / 6
    rgb[:, :, 2] = b / 6
    if toTensor:
        rgb = torch.from_numpy(rgb.transpose([2, 0, 1])).unsqueeze(0)

    return rgb


def tensor2im(input_image, imtype=np.uint8, normalize=True):
    if not isinstance(input_image, np.ndarray):
        if isinstance(input_image, torch.Tensor):
            image_tensor = input_image.data
        else:
            return input_image
        image_numpy = image_tensor[0].cpu().float().numpy()
        if image_numpy.shape[0] == 3:
            image_numpy = np.transpose(image_numpy, (1, 2, 0))
            if normalize:
                image_numpy = (image_numpy + 1) / 2.0 * 255.0
    else:
        image_numpy = input_image
    return image_numpy.astype(imtype)


def tensor2np(input_image, if_normalize=True):
    """
    :param input_image: C*H*W / H*W
    :return: ndarray, H*W*C / H*W
    """
    if isinstance(input_image, torch.Tensor):  # get the data from a variable
        image_tensor = input_image.data
        image_numpy = image_tensor.cpu().float().numpy()  # convert it into a numpy array

    else:
        image_numpy = input_image
    if image_numpy.ndim == 2:
        return image_numpy
    elif image_numpy.ndim == 3:
        C, H, W = image_numpy.shape
        image_numpy = np.transpose(image_numpy, (1, 2, 0))
        if C == 1:
            image_numpy = image_numpy[:, :, 0]
        if if_normalize and C == 3:
            image_numpy = (image_numpy + 1) / 2.0 * 255.0
            image_numpy[image_numpy < 0] = 0
            image_numpy[image_numpy > 255] = 255
            image_numpy = image_numpy.astype(np.uint8)
    return image_numpy

