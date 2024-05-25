import datetime
import sys
import torch.backends.cudnn as cudnn
import torch.optim.lr_scheduler
import dataset as myDataLoader
import Transforms as myTransforms
from utils import *
import os
import numpy as np
from argparse import ArgumentParser
from torch.utils.tensorboard import SummaryWriter
from thop import profile

sys.path.insert(0, 'tools')


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--inWidth', type=int, default=256, help='Width of RGB image')
    parser.add_argument('--inHeight', type=int, default=256, help='Height of RGB image')
    parser.add_argument('--max_steps', type=int, default=40000, help='Max. number of iterations')
    parser.add_argument('--num_workers', type=int, default=4, help='No. of parallel threads')
    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--step_loss', type=int, default=100, help='Decrease learning rate after how many epochs')
    parser.add_argument('--lr', type=float, default=5e-4, help='Initial learning rate')
    parser.add_argument('--lr_mode', default='poly', help='Learning rate policy, step or poly')
    parser.add_argument('--resume', default=False, help='Use this checkpoint to continue training | '
                                                        './results_ep100/checkpoint.pth.tar')
    parser.add_argument('--logFile', default='trainValLog.txt',
                        help='File that stores the training and validation logs')
    parser.add_argument('--onGPU', default=True, type=lambda x: (str(x).lower() == 'true'),
                        help='Run on CPU or GPU. If TRUE, then GPU.')
    parser.add_argument('--weight', default='', type=str, help='pretrained weight, can be a non-strict copy')
    parser.add_argument('--ms', type=int, default=0, help='apply multi-scale training, default False')
    parser.add_argument('--pretrained', default=True, help='Use the pre-train checkpoint to training')
    parser.add_argument('--file_root', default="LEVIR",
                        help='Data directory | LEVIR | SYSU | WHUCD256 | CDD')
    args = parser.parse_args()
    print('Called with args:')
    print(args)
    return args

def trainValidateSegmentation(args):
    torch.backends.cudnn.benchmark = True
    SEED = 2333
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    # model = BaseNet_LWGANet_L0(pretrained=args.pretrained)
    model = BaseNet_LWGANet_L2(pretrained=args.pretrained)
    # model = BaseNet_MobileNetv2(3, 1, pretrained=args.pretrained)

    # model_name = 'A2Net/L0'
    # model_name = 'A2Net_LWGANet/L0'
    model_name = 'A2Net_LWGANet/L2'


    input_data = torch.randn(1, 3, 256, 256)
    flops, params = profile(model, inputs=(input_data, input_data))
    print("flops: {}, params: {}.".format(flops, params))

    args.savedir = ('results/' + model_name + '/' + args.file_root +
                    '/pretrain=' + '{}'.format(args.pretrained) +
                    '/{}'.format(datetime.datetime.now().strftime("%y.%m.%d-%H:%M")) +
                    '_' + args.file_root + '_iter_' + str(args.max_steps) + '_lr_' + str(args.lr) + '/')
    args.vis_dir = args.savedir + '/Vis/'

    tensorboard = os.path.join(args.savedir, 'runs')
    tb_writer = SummaryWriter(tensorboard)

    dataset_name = args.file_root
    if args.file_root == 'LEVIR':
        args.file_root = '/dataset/CD/LEVIR_256_split'
    elif args.file_root == 'SYSU':
        args.file_root = '/dataset/CD/SYSU_256'
    elif args.file_root == 'WHUCD256':
        args.file_root = '/dataset/CD/WHU_256'
    elif args.file_root == 'CDD':
        args.file_root = '/dataset/CD/CDD_256'
    elif args.file_root == 'quick_start':
        args.file_root = './samples'
    else:
        raise TypeError('%s has not defined' % args.file_root)

    if not os.path.exists(args.savedir):
        os.makedirs(args.savedir)

    if not os.path.exists(args.vis_dir):
        os.makedirs(args.vis_dir)

    if args.onGPU:
        model = model.cuda()

    total_params = sum([np.prod(p.size()) for p in model.parameters()])
    print('Total network parameters (excluding idr): ' + str(total_params))

    mean = [0.406, 0.456, 0.485, 0.406, 0.456, 0.485]
    std = [0.225, 0.224, 0.229, 0.225, 0.224, 0.229]
    # mean = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
    # std = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]

    # compose the data with transforms
    trainDataset_main = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.RandomCropResize(int(7. / 224. * args.inWidth)),
        myTransforms.RandomFlip(),
        myTransforms.RandomExchange(),
        # myTransforms.GaussianNoise(),
        myTransforms.ToTensor()
    ])

    valDataset = myTransforms.Compose([
        myTransforms.Normalize(mean=mean, std=std),
        myTransforms.Scale(args.inWidth, args.inHeight),
        myTransforms.ToTensor()
    ])

    train_data = myDataLoader.Dataset("train", file_root=args.file_root,
                                      transform=trainDataset_main, dataset_name=dataset_name)

    trainLoader = torch.utils.data.DataLoader(
        train_data,
        batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False, drop_last=True
    )

    val_data = myDataLoader.Dataset("val", file_root=args.file_root,
                                    transform=valDataset, dataset_name=dataset_name)
    valLoader = torch.utils.data.DataLoader(
        val_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    test_data = myDataLoader.Dataset("test", file_root=args.file_root,
                                     transform=valDataset, dataset_name=dataset_name)
    testLoader = torch.utils.data.DataLoader(
        test_data, shuffle=False,
        batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=False)

    # whether use multi-scale training

    max_batches = len(trainLoader)

    print('For each epoch, we have {} batches'.format(max_batches))

    if args.onGPU:
        cudnn.benchmark = True

    args.max_epochs = int(np.ceil(args.max_steps / max_batches))
    start_epoch = 0
    cur_iter = 0
    max_F1_val = 0

    if args.resume is not None:
        args.resume = args.savedir + '/checkpoint.pth.tar'
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            start_epoch = checkpoint['epoch']
            cur_iter = start_epoch * len(trainLoader)
            # args.lr = checkpoint['lr']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    logFileLoc = args.savedir + args.logFile
    if os.path.isfile(logFileLoc):
        logger = open(logFileLoc, 'a')
    else:
        logger = open(logFileLoc, 'w')
        # logger.write("args: ", args)
        logger.write("Parameters: %s" % (str(total_params)))
        logger.write("\nflops(G): %s\t\tparams(M): %s\n" % (flops, params))

    optimizer = torch.optim.Adam(model.parameters(), args.lr, (0.9, 0.99), eps=1e-08, weight_decay=1e-4)

    start_time = datetime.datetime.now()
    print("start_time:", start_time)
    for epoch in range(start_epoch, args.max_epochs):

        lossTr, score_tr, lr = \
            train(args, trainLoader, model, optimizer, epoch, max_batches, cur_iter)
        cur_iter += len(trainLoader)

        torch.cuda.empty_cache()

        # evaluate on validation set
        if epoch == 0:
            continue

        lossVal, score_val = val(args, valLoader, model, epoch)
        torch.cuda.empty_cache()

        logger.write("\nEpoch(val): %d\t\tKappa: %.4f\t\tIoU: %.4f\t\tF1: %.4f\t\tR: %.4f\t\tP: %.4f"
                     % (epoch, score_val['Kappa'], score_val['IoU'],
                        score_val['F1'], score_val['recall'], score_val['precision']))
        logger.flush()

        torch.save({
            'epoch': epoch + 1,
            'arch': str(model),
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lossTr': lossTr,
            'lossVal': lossVal,
            'F_Tr': score_tr['F1'],
            'F_val': score_val['F1'],
            'lr': lr
        }, args.savedir + 'checkpoint.pth.tar')

        # save the model also
        best_model_file_name = args.savedir + 'best_model.pth'
        if epoch % 1 == 0 and max_F1_val <= score_val['F1']:
            max_F1_val = score_val['F1']
            torch.save(model.state_dict(), best_model_file_name)

        print("Epoch " + str(epoch) + ': Details')
        print("\nEpoch No. %d:\tTrain Loss = %.4f\tVal Loss = %.4f\t F1(tr) = %.4f\t F1(val) = %.4f" \
              % (epoch, lossTr, lossVal, score_tr['F1'], score_val['F1']))
        torch.cuda.empty_cache()

        tags = ["Kappa(val)", "mIoU(val)", "F1(val)", "Recall(val)", "Precision(val)"]
        tb_writer.add_scalar(tags[0], score_val['Kappa'], epoch)
        tb_writer.add_scalar(tags[1], score_val['IoU'], epoch)
        tb_writer.add_scalar(tags[2], score_val['F1'], epoch)
        tb_writer.add_scalar(tags[3], score_val['recall'], epoch)
        tb_writer.add_scalar(tags[4], score_val['precision'], epoch)

    # train time
    end_time = datetime.datetime.now()
    print("start_time:", start_time)
    print("end_time:", end_time)
    all_time = end_time - start_time
    print("all_time:", all_time)

    best_model_file_name = args.savedir + 'best_model.pth'

    best_state_dict = torch.load(best_model_file_name)
    model.load_state_dict(best_state_dict)

    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest(best_epoch): \t Kappa = %.4f\t IoU = %.4f\t F1 = %.4f\t R = %.4f\t P = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\tKappa: %.4f\t\tIoU: %.4f\t\tF1: %.4f\t\tR: %.4f\t\tP: %.4f\nParameters: %s\t\ttrain_all_time: %s"
                 % ('Test(best_epoch): ', score_test['Kappa'], score_test['IoU'], score_test['F1'],
                    score_test['recall'], score_test['precision'], str(total_params), all_time))

    epo = 200
    tags = ["Kappa(val)", "mIoU(val)", "F1(val)", "Recall(val)", "Precision(val)"]
    tb_writer.add_scalar(tags[0], score_test['Kappa'], epo)
    tb_writer.add_scalar(tags[1], score_test['IoU'], epo)
    tb_writer.add_scalar(tags[2], score_test['F1'], epo)
    tb_writer.add_scalar(tags[3], score_test['recall'], epo)
    tb_writer.add_scalar(tags[4], score_test['precision'], epo)

    last_model_file_name = args.savedir + 'checkpoint.pth.tar'

    last_state_dict = torch.load(last_model_file_name)
    model.load_state_dict(last_state_dict['state_dict'])

    loss_test, score_test = val(args, testLoader, model, 0)
    print("\nTest(last_epoch): \t Kappa = %.4f\t IoU = %.4f\t F1 = %.4f\t R = %.4f\t P = %.4f" \
          % (score_test['Kappa'], score_test['IoU'], score_test['F1'], score_test['recall'], score_test['precision']))
    logger.write("\n%s\t\tKappa: %.4f\t\tIoU: %.4f\t\tF1: %.4f\t\tR: %.4f\t\tP: %.4f\nParameters: %s\t\ttrain_all_time: %s"
                 % ('Test(last_epoch): ', score_test['Kappa'], score_test['IoU'], score_test['F1'],
                    score_test['recall'], score_test['precision'], str(total_params), all_time))

    epo = 220
    tags = ["Kappa(val)", "mIoU(val)", "F1(val)", "Recall(val)", "Precision(val)"]
    tb_writer.add_scalar(tags[0], score_test['Kappa'], epo)
    tb_writer.add_scalar(tags[1], score_test['IoU'], epo)
    tb_writer.add_scalar(tags[2], score_test['F1'], epo)
    tb_writer.add_scalar(tags[3], score_test['recall'], epo)
    tb_writer.add_scalar(tags[4], score_test['precision'], epo)

    logger.flush()
    logger.close()

if __name__ == '__main__':
    args = parse_args()
    trainValidateSegmentation(args)

