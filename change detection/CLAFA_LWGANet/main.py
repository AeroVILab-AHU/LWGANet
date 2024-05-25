import torch
from data.cd_dataset import DataLoader
from model.create_model import create_model
from tqdm import tqdm
import math
from util.metric_tool import ConfuseMatrixMeter
import os
import numpy as np
import random
import argparse
from thop import profile

class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser()

    def init(self):
        self.parser.add_argument('--gpu_ids', type=str, default='0', help='gpu ids: e.g. 0. use -1 for CPU')
        self.parser.add_argument('--name', type=str, default='lwganet_l0_e200')
        self.parser.add_argument('--backbone', type=str, default='lwganet_l0',
                                 help='lwganet_l0 | lwganet_l2 | mobilenetv2 | resnet18d')
        self.parser.add_argument('--dataroot', type=str, default='/dataset/CD')
        self.parser.add_argument('--dataset', type=str, default='SYSU_256',
                                 help='LEVIR_256_split | WHU_256 | CDD_256 | SYSU_256')
        self.parser.add_argument('--checkpoint_dir', type=str, default='./checkpoints', help='models are saved here')

        self.parser.add_argument('--result_dir', type=str, default='./results', help='results are saved here')
        self.parser.add_argument('--load_pretrain', type=bool, default=False)

        self.parser.add_argument('--phase', type=str, default='train')
        self.parser.add_argument('--input_size', type=int, default=256)
        self.parser.add_argument('--fpn', type=str, default='fpn')
        self.parser.add_argument('--fpn_channels', type=int, default=128)
        self.parser.add_argument('--deform_groups', type=int, default=4)
        self.parser.add_argument('--gamma_mode', type=str, default='SE')
        self.parser.add_argument('--beta_mode', type=str, default='contextgatedconv')
        self.parser.add_argument('--num_heads', type=int, default=1)
        self.parser.add_argument('--num_points', type=int, default=8)
        self.parser.add_argument('--kernel_layers', type=int, default=1)
        self.parser.add_argument('--init_type', type=str, default='kaiming_normal')
        self.parser.add_argument('--alpha', type=float, default=0.25)
        self.parser.add_argument('--gamma', type=int, default=4, help='gamma for Focal loss')
        self.parser.add_argument('--dropout_rate', type=float, default=0.1)

        self.parser.add_argument('--batch_size', type=int, default=16)
        self.parser.add_argument('--num_epochs', type=int, default=200)
        self.parser.add_argument('--warmup_epochs', type=int, default=20)
        self.parser.add_argument('--num_workers', type=int, default=4, help='#threads for loading data')
        self.parser.add_argument('--lr', type=float, default=5e-4)
        self.parser.add_argument('--weight_decay', type=float, default=5e-4)

    def parse(self):
        self.init()
        self.opt = self.parser.parse_args()

        str_ids = self.opt.gpu_ids.split(',')
        self.opt.gpu_ids = []
        for str_id in str_ids:
            id = int(str_id)
            if id >= 0:
                self.opt.gpu_ids.append(id)

        # set gpu ids
        if len(self.opt.gpu_ids) > 0:
            torch.cuda.set_device(self.opt.gpu_ids[0])

        args = vars(self.opt)

        print('------------ Options -------------')
        for k, v in sorted(args.items()):
            print('%s: %s' % (str(k), str(v)))
        print('-------------- End ----------------')

        return self.opt


def setup_seed(seed):
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False #!
    torch.backends.cudnn.benchmark = True      #!
    torch.backends.cudnn.enabled = True        #! for accelerating training 


class Trainval(object):
    def __init__(self, opt):
        self.opt = opt

        train_loader = DataLoader(opt)
        self.train_data = train_loader.load_data()
        train_size = len(train_loader)
        print("#training images = %d" % train_size)
        opt.phase = 'val'
        val_loader = DataLoader(opt)
        self.val_data = val_loader.load_data()
        val_size = len(val_loader)
        print("#validation images = %d" % val_size)
        opt.phase = 'train'

        self.model = create_model(opt)
        self.optimizer = self.model.optimizer
        self.schedular = self.model.schedular

        self.iters = 0
        self.total_iters = math.ceil(train_size / opt.batch_size) * opt.num_epochs
        self.previous_best = 0.0
        self.running_metric = ConfuseMatrixMeter(n_class=2)

    def train(self):
        tbar = tqdm(self.train_data, ncols=80)
        opt.phase = 'train'
        _loss = 0.0
        _focal_loss = 0.0
        _dice_loss = 0.0

        for i, data in enumerate(tbar):
            self.model.detector.train()
            focal, dice, p2_loss, p3_loss, p4_loss, p5_loss = self.model(data['img1'].cuda(), data['img2'].cuda(), data['cd_label'].cuda())
            loss = focal * 0.5 + dice + p3_loss + p4_loss + p5_loss
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            self.schedular.step()
            _loss += loss.item()
            _focal_loss += focal.item()
            _dice_loss += dice.item()
            del loss

            tbar.set_description("Loss: %.3f, Focal: %.3f, Dice: %.3f, LR: %.6f" %
                                 (_loss / (i + 1), _focal_loss / (i + 1), _dice_loss / (i + 1), self.optimizer.param_groups[0]['lr']))

    def val(self):
        tbar = tqdm(self.val_data, ncols=80)
        self.running_metric.clear()
        opt.phase = 'val'
        self.model.eval()

        with torch.no_grad():
            for i, _data in enumerate(tbar):
                val_pred = self.model.inference(_data['img1'].cuda(), _data['img2'].cuda())
                # update metric
                val_target = _data['cd_label'].detach()
                val_pred = torch.argmax(val_pred.detach(), dim=1)
                _ = self.running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
            val_scores = self.running_metric.get_scores()
            message = '(phase: %s) ' % (self.opt.phase)
            for k, v in val_scores.items():
                message += '%s: %.3f ' % (k, v * 100)
            print(message)

        if val_scores['mf1'] >= self.previous_best:
            name = opt.dataset + '_' + opt.name
            print("name", name)
            self.model.save(name, self.opt.backbone)
            self.previous_best = val_scores['mf1']


if __name__ == "__main__":
    opt = Options().parse()
    trainval = Trainval(opt)
    setup_seed(seed=1)

    # train and val
    for epoch in range(1, opt.num_epochs + 1):
        name = opt.dataset + '_' + opt.name
        print("name", name)
        print("\n==> Name %s, Epoch %i, previous best = %.3f" % (name, epoch, trainval.previous_best * 100))
        trainval.train()
        trainval.val()

    # test
    opt.phase = 'test'
    test_loader = DataLoader(opt)
    test_data = test_loader.load_data()
    test_size = len(test_loader)
    print("#testing images = %d" % test_size)

    opt.load_pretrain = True
    model = create_model(opt)

    tbar = tqdm(test_data, ncols=80)
    total_iters = test_size
    running_metric = ConfuseMatrixMeter(n_class=2)
    running_metric.clear()

    model.eval()
    with torch.no_grad():
        for i, _data in enumerate(tbar):
            val_pred = model.inference(_data['img1'].cuda(), _data['img2'].cuda())
            # update metric
            val_target = _data['cd_label'].detach()
            val_pred = torch.argmax(val_pred.detach(), dim=1)
            _ = running_metric.update_cm(pr=val_pred.cpu().numpy(), gt=val_target.cpu().numpy())
        val_scores = running_metric.get_scores()
        message = '(phase: %s) ' % (opt.phase)
        for k, v in val_scores.items():
            message += '%s: %.3f ' % (k, v * 100)
        print('test: \n')
        print('model_name: {},\n dataset: {},\n message: {}\n'.format(opt.name, opt.dataset, message))
