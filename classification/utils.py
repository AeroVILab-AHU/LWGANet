import datetime
import sys
import json
import pickle
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import Dataset
import math

from models.LWGANet import *
from models.MobileNetV2 import *
from models.efficientformerv2 import *
from models.fasternet import *
from models.edgevit import *
from models.edgenext.edgenext import *
from models.ghostnetv2 import *
from models.mobileViT import *
from models.PVT_V2 import *


class RSDataSet(Dataset):
    def __init__(self, images_path: list, images_class: list, transform=None):
        self.images_path = images_path
        self.images_class = images_class
        self.transform = transform

    def __len__(self):
        return len(self.images_path)

    def __getitem__(self, item):
        img = Image.open(self.images_path[item])
        if img.mode != 'RGB':
            img = img.convert("RGB")
        label = self.images_class[item]

        if self.transform is not None:
            img = self.transform(img)

        return img, label

    @staticmethod
    def collate_fn(batch):
        images, labels = tuple(zip(*batch))

        images = torch.stack(images, dim=0)
        labels = torch.as_tensor(labels)
        return images, labels


def dataset_path_cla(args):
    dataset = args.datasets
    if args.datasets == 'RESISC45-82':
        data_path = '/dataset/classification/NWPU-RESISC45-82/'
        num_classes = 45
    elif args.datasets == 'UCM-82':
        data_path = '/dataset/classification/UCM-82/'
        num_classes = 21
    elif args.datasets == 'AID-82':
        data_path = '/dataset/classification/AID-82/'
        num_classes = 30
    else:
        assert args.datasets, 'path is incorrect or does not exist!'

    return dataset, data_path, num_classes


def load_model(args, num_classes, device):
    weights = ''
    if args.model == 'LWGANet_L0':
        model = LWGANet_L0_1242_e32_k11_GELU(num_classes=num_classes).to(device)
    elif args.model == 'LWGANet_L1':
        model = LWGANet_L1_1242_e64_k11_GELU(num_classes=num_classes).to(device)
    elif args.model == 'LWGANet_L2':
        model = LWGANet_L2_1442_e96_k11_ReLU(num_classes=num_classes).to(device)
    elif args.model == 'mobilenet_v2_10':
        model = mobilenet_v2_10(num_classes=num_classes).to(device)
    elif args.model == 'mobilenet_v2_20':
        model = mobilenet_v2_20(num_classes=num_classes).to(device)
    elif args.model == 'mobilenet_v2_25':
        model = mobilenet_v2_25(num_classes=num_classes).to(device)
    elif args.model == 'efficientformerv2_s0':
        model = efficientformerv2_s0(num_classes=num_classes).to(device)
    elif args.model == 'efficientformerv2_s1':
        model = efficientformerv2_s1(num_classes=num_classes).to(device)
    elif args.model == 'efficientformerv2_s2':
        model = efficientformerv2_s2(num_classes=num_classes).to(device)
    elif args.model == 'fasternet_t0':
        model = fasternet_t0(num_classes=num_classes).to(device)
    elif args.model == 'fasternet_t1':
        model = fasternet_t1(num_classes=num_classes).to(device)
    elif args.model == 'fasternet_t2':
        model = fasternet_t2(num_classes=num_classes).to(device)
    elif args.model == 'edgevit_xxs':
        model = edgevit_xxs(num_classes=num_classes).to(device)
    elif args.model == 'edgevit_xs':
        model = edgevit_xs(num_classes=num_classes).to(device)
    elif args.model == 'edgevit_s':
        model = edgevit_s(num_classes=num_classes).to(device)
    elif args.model == 'edgenext_xx_small':
        model = edgenext_xx_small(num_classes=num_classes).to(device)
    elif args.model == 'edgenext_x_small':
        model = edgenext_x_small(num_classes=num_classes).to(device)
    elif args.model == 'edgenext_small':
        model = edgenext_small(num_classes=num_classes).to(device)
    elif args.model == 'ghostnetv2_06':
        model = ghostnetv2_06(num_classes=num_classes).to(device)
    elif args.model == 'ghostnetv2_10':
        model = ghostnetv2_10(num_classes=num_classes).to(device)
    elif args.model == 'ghostnetv2_20':
        model = ghostnetv2_20(num_classes=num_classes).to(device)
    elif args.model == 'mobilevit_xxs':
        model = mobilevit_xxs(num_classes=num_classes).to(device)
    elif args.model == 'mobilevit_xs':
        model = mobilevit_xs(num_classes=num_classes).to(device)
    elif args.model == 'mobilevit_s':
        model = mobilevit_s(num_classes=num_classes).to(device)
    elif args.model == 'pvt_v2_b0':
        model = pvt_v2_b0(num_classes=num_classes).to(device)
    elif args.model == 'pvt_v2_b1':
        model = pvt_v2_b1(num_classes=num_classes).to(device)
    else:
        assert args.model, ' is incorrect!'
    version = args.model + '_batchsize64_lr0.0005_wd0.05'

    return model, version, weights


def read_train_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    imagenet_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    imagenet_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(imagenet_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_train_indices.json', 'w') as json_file:
        json_file.write(json_str)

    train_images_path = []
    train_images_label = []
    every_class_num = []
    supported = [".jpeg", ".jpg", ".JPG", ".png", ".PNG", ".JPEG", ".tif"]

    for cla in imagenet_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        for img_path in images:
            train_images_path.append(img_path)
            train_images_label.append(image_class)

    print("{} images for training.".format(len(train_images_path)))
    assert len(train_images_path) > 0, "not find data for train."

    return train_images_path, train_images_label


def read_val_data(root: str):
    random.seed(0)
    assert os.path.exists(root), "dataset root: {} does not exist.".format(root)

    imagenet_class = [cla for cla in os.listdir(root) if os.path.isdir(os.path.join(root, cla))]

    imagenet_class.sort()

    class_indices = dict((k, v) for v, k in enumerate(imagenet_class))
    json_str = json.dumps(dict((val, key) for key, val in class_indices.items()), indent=4)
    with open('class_val_indices.json', 'w') as json_file:
        json_file.write(json_str)

    val_images_path = []
    val_images_label = []
    every_class_num = []
    supported = [".jpeg", ".jpg", ".JPG", ".png", ".PNG", ".JPEG", ".tif"]

    for cla in imagenet_class:
        cla_path = os.path.join(root, cla)

        images = [os.path.join(root, cla, i) for i in os.listdir(cla_path)
                  if os.path.splitext(i)[-1] in supported]

        image_class = class_indices[cla]

        every_class_num.append(len(images))

        for img_path in images:
            val_images_path.append(img_path)
            val_images_label.append(image_class)

    print("{} images for validation.".format(len(val_images_path)))
    assert len(val_images_path) > 0, "not find data for train."

    return val_images_path, val_images_label


def write_pickle(list_info: list, file_name: str):
    with open(file_name, 'wb') as f:
        pickle.dump(list_info, f)


def read_pickle(file_name: str) -> list:
    with open(file_name, 'rb') as f:
        info_list = pickle.load(f)
        return info_list


def train_one_epoch(model, optimizer, scheduler, data_loader, device, epoch):
    model.train()
    loss_function = torch.nn.CrossEntropyLoss()
    accu_loss = torch.zeros(1).to(device)
    accu_num = torch.zeros(1).to(device)
    optimizer.zero_grad()

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        loss.backward()
        accu_loss += loss.detach()

        data_loader.desc = "[train epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

        if not torch.isfinite(loss):
            print('WARNING: non-finite loss, ending training ', loss)
            sys.exit(1)

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


@torch.no_grad()
def evaluate(model, data_loader, device, epoch):
    loss_function = torch.nn.CrossEntropyLoss()

    model.eval()

    accu_num = torch.zeros(1).to(device)
    accu_loss = torch.zeros(1).to(device)

    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

        pred = model(images.to(device))
        pred_classes = torch.max(pred, dim=1)[1]
        accu_num += torch.eq(pred_classes, labels.to(device)).sum()

        loss = loss_function(pred, labels.to(device))
        accu_loss += loss

        data_loader.desc = "[valid epoch {}] loss: {:.3f}, acc: {:.3f}".format(epoch,
                                                                               accu_loss.item() / (step + 1),
                                                                               accu_num.item() / sample_num)

    return accu_loss.item() / (step + 1), accu_num.item() / sample_num


def sample(data_loader):
    sample_num = 0
    data_loader = tqdm(data_loader, file=sys.stdout)
    for step, data in enumerate(data_loader):
        images, labels = data
        sample_num += images.shape[0]

    return sample_num


def create_lr_scheduler(optimizer,
                        num_step: int,
                        epochs: int,
                        warmup=True,
                        warmup_epochs=1,
                        warmup_factor=1e-3,
                        end_factor=1e-6):
    assert num_step > 0 and epochs > 0
    if warmup is False:
        warmup_epochs = 0

    def f(x):
        if warmup is True and x <= (warmup_epochs * num_step):
            alpha = float(x) / (warmup_epochs * num_step)
            return warmup_factor * (1 - alpha) + alpha
        else:
            current_step = (x - warmup_epochs * num_step)
            cosine_steps = (epochs - warmup_epochs) * num_step
            return ((1 + math.cos(current_step * math.pi / cosine_steps)) / 2) * (1 - end_factor) + end_factor

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=f)

