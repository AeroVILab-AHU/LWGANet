from .transform import Transforms
import numpy as np
import os
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms


def make_dataset(dir):
    img_paths = []
    names = []
    assert os.path.isdir(dir), '%s is not a valid directory' % dir

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            path = os.path.join(root, fname)
            img_paths.append(path)
            names.append(fname)

    return img_paths, names


class Load_Dataset(Dataset):
    def __init__(self, opt):
        super(Load_Dataset, self).__init__()
        self.opt = opt
        if (opt.dataset == 'LEVIR_256_split') or (opt.dataset == 'SYSU_256'):
            self.dir1 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'A')
            self.t1_paths, self.fnames = sorted(make_dataset(self.dir1))

            self.dir2 = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'B')
            self.t2_paths, _ = sorted(make_dataset(self.dir2))

            self.dir_label = os.path.join(opt.dataroot, opt.dataset, opt.phase, 'label')
            self.label_paths, _ = sorted(make_dataset(self.dir_label))
        else:
            self.fnames = open(os.path.join(opt.dataroot, opt.dataset, "list", opt.phase + '.txt')).read().splitlines()
            self.dir1 = os.path.join(opt.dataroot, opt.dataset, "A")
            self.t1_paths = [os.path.join(self.dir1, name) for name in self.fnames]

            self.dir2 = os.path.join(opt.dataroot, opt.dataset, "B")
            self.t2_paths = [os.path.join(self.dir2, name) for name in self.fnames]

            self.dir_label = os.path.join(opt.dataroot, opt.dataset, 'label')
            self.label_paths = [os.path.join(self.dir_label, name) for name in self.fnames]

        self.dataset_size = len(self.t1_paths)

        self.normalize = transforms.Compose([transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        self.transform = transforms.Compose([Transforms()])
        self.to_tensor = transforms.Compose([transforms.ToTensor()])

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, index):
        t1_path = self.t1_paths[index]
        fname = self.fnames[index]
        img1 = Image.open(t1_path)

        t2_path = self.t2_paths[index]
        img2 = Image.open(t2_path)

        label_path = self.label_paths[index]
        label = np.array(Image.open(label_path)) // 255
        cd_label = Image.fromarray(label)

        if self.opt.phase == 'train':
            _data = self.transform({'img1': img1, 'img2': img2, 'cd_label': cd_label})
            img1, img2, cd_label = _data['img1'], _data['img2'], _data['cd_label']

        img1 = self.to_tensor(img1)
        img2 = self.to_tensor(img2)
        img1 = self.normalize(img1)
        img2 = self.normalize(img2)
        cd_label = torch.from_numpy(np.array(cd_label))
        input_dict = {'img1': img1, 'img2': img2, 'cd_label': cd_label, 'fname': fname}

        return input_dict


class DataLoader(torch.utils.data.Dataset):

    def __init__(self, opt):
        self.dataset = Load_Dataset(opt)
        self.dataloader = torch.utils.data.DataLoader(self.dataset,
                                                      batch_size=opt.batch_size,
                                                      shuffle=opt.phase=='train',
                                                      pin_memory=True,
                                                      drop_last=opt.phase=='train',
                                                      num_workers=int(opt.num_workers)
                                                      )

    def load_data(self):
        return self.dataloader

    def __len__(self):
        return len(self.dataset)
