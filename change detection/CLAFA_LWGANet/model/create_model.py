from .network import Detector
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange
import os
import torch.optim as optim
from .block.schedular import get_cosine_schedule_with_warmup
from .loss.focal import FocalLoss
from .loss.dice import DICELoss
from thop import profile

def get_model(backbone_name='mobilenetv2', fpn_name='fpn', fpn_channels=128, deform_groups=4,
              gamma_mode='SE', beta_mode='contextgatedconv', num_heads=1, num_points=8, kernel_layers=1,
              dropout_rate=0.1, init_type='kaiming_normal'):
    detector = Detector(backbone_name, fpn_name, fpn_channels, deform_groups, gamma_mode, beta_mode,
                        num_heads, num_points, kernel_layers, dropout_rate, init_type)
    print(detector)

    return detector


class Model(nn.Module):
    def __init__(self, opt):
        super(Model, self).__init__()
        self.device = torch.device("cuda:%s" % opt.gpu_ids[0] if torch.cuda.is_available() else "cpu")
        self.opt = opt
        self.base_lr = opt.lr
        self.save_dir = os.path.join(opt.checkpoint_dir, opt.name)
        os.makedirs(self.save_dir, exist_ok=True)

        self.detector = get_model(backbone_name=opt.backbone, fpn_name=opt.fpn, fpn_channels=opt.fpn_channels,
                                  deform_groups=opt.deform_groups, gamma_mode=opt.gamma_mode, beta_mode=opt.beta_mode,
                                  num_heads=opt.num_heads, num_points=opt.num_points, kernel_layers=opt.kernel_layers,
                                  dropout_rate=opt.dropout_rate, init_type=opt.init_type)
        self.focal = FocalLoss(alpha=opt.alpha, gamma=opt.gamma)
        self.dice = DICELoss()

        self.optimizer = optim.AdamW(self.detector.parameters(), lr=opt.lr, weight_decay=opt.weight_decay)
        # Here, 625 = #training images // batch_size. 
        self.schedular = get_cosine_schedule_with_warmup(self.optimizer, num_warmup_steps=625 * opt.warmup_epochs,
                                                         num_training_steps=625 * opt.num_epochs)
        if opt.load_pretrain:
            self.load_ckpt(self.detector, self.optimizer, opt.dataset + '_' + opt.name, opt.backbone)
        self.detector.cuda()

        print("---------- Networks initialized -------------")

    def forward(self, x1, x2, label):
        pred, pred_p2, pred_p3, pred_p4, pred_p5 = self.detector(x1, x2)
        label = label.long()
        focal = self.focal(pred, label)
        dice = self.dice(pred, label)
        p2_loss = self.focal(pred_p2, label) * 0.5 + self.dice(pred_p2, label)
        p3_loss = self.focal(pred_p3, label) * 0.5 + self.dice(pred_p3, label)
        p4_loss = self.focal(pred_p4, label) * 0.5 + self.dice(pred_p4, label)
        p5_loss = self.focal(pred_p5, label) * 0.5 + self.dice(pred_p5, label)

        return focal, dice, p2_loss, p3_loss, p4_loss, p5_loss

    def inference(self, x1, x2):
        with torch.no_grad():
            pred, _, _, _, _ = self.detector(x1, x2)
            #
            # input_data = torch.randn(1, 3, 256, 256).cuda('cuda:0')
            # flops, params = profile(self.detector, inputs=(input_data, input_data))
            # print("flops: {}, params: {}.".format(flops, params))
        return pred

    def load_ckpt(self, network, optimizer, name, backbone):
        save_filename = '%s_%s_best.pth' % (name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if not os.path.isfile(save_path):
            print("%s not exists yet!" % save_path)
            raise ("%s must exist!" % save_filename)
        else:
            checkpoint = torch.load(save_path, map_location=self.device)
            network.load_state_dict(checkpoint['network'], False)

    def save_ckpt(self, network, optimizer, model_name, backbone):
        save_filename = '%s_%s_best.pth' % (model_name, backbone)
        save_path = os.path.join(self.save_dir, save_filename)
        if os.path.exists(save_path):
            os.remove(save_path)
        torch.save({'network': network.cpu().state_dict(),
                    'optimizer': optimizer.state_dict()},
                   save_path)
        if torch.cuda.is_available():
            network.cuda()

    def save(self, model_name, backbone):
        self.save_ckpt(self.detector, self.optimizer, model_name, backbone)

    def name(self):
        return self.opt.name


def create_model(opt):
    model = Model(opt)
    print("model [%s] was created" % model.name())

    return model.cuda()

