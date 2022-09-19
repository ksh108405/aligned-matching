from data import *
from utils.augmentations import SSDAugmentation
from layers.modules import MultiBoxLoss
from ssd import build_ssd
import os
import sys
import time
import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.init as init
import torch.utils.data as data
import numpy as np
import argparse
import random
from telegram_alerter import send_train_finished, send_train_error
from torchinfo import summary


def str2bool(v):
    return v.lower() in ("yes", "true", "t", "1", "True")


parser = argparse.ArgumentParser(
    description='Single Shot MultiBox Detector Training With Pytorch')
train_set = parser.add_mutually_exclusive_group()
parser.add_argument('--dataset', default='VOC', choices=['VOC', 'COCO', 'TT100K'],
                    type=str, help='VOC or COCO or TT100K')
parser.add_argument('--basenet', default='vgg16_reducedfc.pth',
                    help='Pretrained base model')
parser.add_argument('--batch_size', default=32, type=int,
                    help='Batch size for training')
parser.add_argument('--resume', default=None, type=str,
                    help='Checkpoint state_dict file to resume training from')
parser.add_argument('--start_iter', default=1, type=int,
                    help='Resume training at this iter')
parser.add_argument('--num_workers', default=4, type=int,
                    help='Number of workers used in dataloading')
parser.add_argument('--cuda', default=True, type=str2bool,
                    help='Use CUDA to train model')
parser.add_argument('--gpu_id', default='0', type=str,
                    help='ID of GPU to use during training')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,
                    help='initial learning rate')
parser.add_argument('--momentum', default=0.9, type=float,
                    help='Momentum value for optim')
parser.add_argument('--weight_decay', default=5e-4, type=float,
                    help='Weight decay for SGD')
parser.add_argument('--gamma', default=0.1, type=float,
                    help='Gamma update for SGD')
parser.add_argument('--save_folder', default='weights/',
                    help='Directory for saving checkpoint models')
parser.add_argument('--weight_name', default='None_',
                    help='Saved weight name')
parser.add_argument('--matching_strategy', default='legacy',
                    choices=['legacy', 'aligned_cpu', 'aligned_2_cpu', 'aligned_1a_cpu', 'aligned_3_cpu', 'resized', 'aligned'],
                    help='Select matching strategy (legacy or aligned or aligned_2)')
parser.add_argument('--train_set', default='trainval',
                    help='used for divide train or test')
parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam'],
                    help='Whether to use SGD or Adam optimizer')
parser.add_argument('--time_verbose', default=False, type=float,
                    help='Print more specific training time')
parser.add_argument('--augmentation', default=True, type=str2bool,
                    help='Whether to take augmentation process')
parser.add_argument('--shuffle', default=False, type=str2bool,
                    help='set to True to have the data reshuffled at every epoch')
parser.add_argument('--one_epoch', default=False, type=str2bool,
                    help='Only iterate for one epoch')
parser.add_argument('--fix_loss', default=False, type=str2bool,
                    help='Fix localization loss bugs')
parser.add_argument('--ensure_size', default=None, type=float,
                    help='Ensure conv4_3 default box size')
parser.add_argument('--ensure_archi', default=None, type=int,
                    help='Ensure SSD architecture')
parser.add_argument('--multi_matching', default=True, type=str2bool,
                    help='Enable multi matching after bipartite matching')
parser.add_argument('--multi_thresh', default=0.5, type=float,
                    help='Threshold IoU value of multi-matching algorithm')
parser.add_argument('--saved_matching', default=False, type=str2bool,
                    help='Use saved multi-matching')
parser.add_argument('--saved_conf_loc', default=False, type=str2bool,
                    help='Use saved whole matching')
parser.add_argument('--relative_multi', default=False, type=float,
                    help='Change multi matching to relative')
parser.add_argument('--seed', default=None, type=int,
                    help='Seed value for initialization')
args = parser.parse_args()

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id

# For reproducible output (Added after 22.07.08 01:33:04 KST)
if args.seed is None:
    if args.dataset == 'TT100K':
        SEED = 3407
    elif args.dataset == 'VOC':
        SEED = 3763722138
    elif args.dataset == 'COCO':
        SEED = 3763722138
    else:
        raise Exception(f'Seed not provided for {args.dataset} dataset.')
else:
    SEED = args.seed

random.seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
g = torch.Generator(device='cuda')
g.manual_seed(SEED)
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True
# torch.backends.cudnn.enabled = False
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

print('Number of available GPUs:', torch.cuda.device_count())
print('Current cuda device:', torch.cuda.current_device())

if torch.cuda.is_available():
    if args.cuda:
        torch.set_default_tensor_type('torch.cuda.FloatTensor')
    if not args.cuda:
        print("WARNING: It looks like you have a CUDA device, but aren't " +
              "using CUDA.\nRun with --cuda for optimal training speed.")
        torch.set_default_tensor_type('torch.FloatTensor')
else:
    torch.set_default_tensor_type('torch.FloatTensor')

if not os.path.exists(args.save_folder):
    os.mkdir(args.save_folder)


def train():
    if args.dataset == 'VOC':
        cfg = voc
        image_sets = [('2007', args.train_set), ('2012', args.train_set)]
        dataset = VOCDetection(root=VOC_ROOT, image_sets=image_sets,
                               transform=SSDAugmentation(cfg['min_dim'], MEANS, args.augmentation))
    elif args.dataset == 'TT100K':
        cfg = tt100k
        image_sets = args.train_set
        if args.train_set == 'trainval':
            image_sets = 'train'
        dataset = TT100KDetection(root=TT100K_ROOT, image_sets=image_sets,
                                  transform=SSDAugmentation(cfg['min_dim'], MEANS, args.augmentation))
    elif args.dataset == 'COCO':
        cfg = coco
        dataset = COCODetection(root=COCO_ROOT,
                                transform=SSDAugmentation(cfg['min_dim'], MEANS, args.augmentation))
    else:
        raise Exception('Select on VOC or TT100K or COCO.')

    if args.ensure_size is not None:
        if args.dataset == 'VOC':
            assert VOC_CONV4_3_SIZE == args.ensure_size
        if args.dataset == 'TT100K':
            assert TT100K_CONV4_3_SIZE == args.ensure_size
        if args.dataset == 'COCO':
            assert COCO_CONV4_3_SIZE == args.ensure_size

    if args.ensure_archi is not None:
        if args.dataset == 'VOC':
            assert VOC_NETWORK_SIZE == args.ensure_archi
        if args.dataset == 'TT100K':
            assert TT100K_NETWORK_SIZE == args.ensure_archi
        if args.dataset == 'COCO':
            assert COCO_NETWORK_SIZE == args.ensure_archi

    ssd_net = build_ssd('train', cfg['min_dim'], cfg['num_classes'], args.dataset)
    # summary(ssd_net, input_size=(args.batch_size, 3, cfg['min_dim'], cfg['min_dim']), device='cuda')
    net = ssd_net

    if args.cuda:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        net = net.to(device)

    if args.resume:
        print('Resuming training, loading {}...'.format(args.resume))
        ssd_net.load_weights(args.resume)
    else:
        vgg_weights = torch.load(args.basenet)
        print('Loading base network...')
        ssd_net.vgg.load_state_dict(vgg_weights)

    if not args.resume:
        print('Initializing weights...')
        # initialize newly added layers' weights with xavier method
        ssd_net.extras.apply(weights_init)
        ssd_net.loc.apply(weights_init)
        ssd_net.conf.apply(weights_init)

    if args.optimizer == 'SGD':
        optimizer = optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum,
                              weight_decay=args.weight_decay)
    elif args.optimizer == 'Adam':
        optimizer = optim.Adam(net.parameters(), lr=args.lr, betas=(0.9, 0.999), eps=1e-08,
                               weight_decay=args.weight_decay,
                               amsgrad=True)  # use adam for tt100k training
    criterion = MultiBoxLoss(cfg['num_classes'], args.multi_thresh, True, 0, True, 3, 0.5,
                             False, cfg, args.cuda, matching=args.matching_strategy, fix_loss=args.fix_loss,
                             multi_matching=args.multi_matching, saved_matching=args.saved_matching,
                             saved_conf_loc=args.saved_conf_loc, relative_multi=args.relative_multi)

    net.train()

    # loss counters
    print('Loading the dataset...')

    epoch_size = float(len(dataset)) / args.batch_size
    warm_up_target = np.ceil(5 * epoch_size)
    print('Training SSD on:', dataset.name)
    print('Using the specified args:')
    print(args)

    step_index = 0

    # lr warming up
    lr_warm_up_timer = 0
    if args.dataset == 'VOC':
        lr_warm_up_timer = warm_up_target - args.start_iter + 1
        print(f'Learning rate warming up until iteration {warm_up_target}')

    # adjust lr on resuming training
    lr_adjusted = True
    if args.start_iter != 1:
        lr_adjusted = False

    data_loader = data.DataLoader(dataset, args.batch_size,
                                  num_workers=args.num_workers,
                                  shuffle=args.shuffle, collate_fn=detection_collate,
                                  pin_memory=True, worker_init_fn=seed_worker, generator=g)
    # create batch iterator
    batch_iterator = iter(data_loader)
    for iteration in range(args.start_iter, cfg['max_iter'] + 1):
        # adjust lr on resuming training
        if not lr_adjusted:
            for i, lr_step in enumerate(reversed(cfg['lr_steps'])):
                if iteration > lr_step:
                    step_index = len(cfg['lr_steps']) - i
                    set_learning_rate_decay(optimizer, args.gamma, step_index)
                    lr_adjusted = True
                    break
        # lr warming up
        if lr_warm_up_timer != 0:
            set_learning_rate(optimizer, args.lr * (iteration / warm_up_target))
            lr_warm_up_timer -= 1

        if iteration - 1 in cfg['lr_steps']:
            step_index += 1
            set_learning_rate_decay(optimizer, args.gamma, step_index)

        # load train data
        try:
            images, targets = next(batch_iterator)
        except StopIteration:
            batch_iterator = iter(data_loader)
            images, targets = next(batch_iterator)

        if args.cuda:
            images = Variable(images.cuda())
            with torch.no_grad():
                targets = [Variable(ann.cuda()) for ann in targets]
        else:
            images = Variable(images)
            with torch.no_grad():
                targets = [Variable(ann) for ann in targets]

        # forward
        t0 = time.time()
        out = net(images)
        if args.time_verbose:
            t_1 = time.time()

        # backprop
        optimizer.zero_grad()
        loss_l, loss_c = criterion(out, targets)
        loss = loss_l + loss_c
        if args.time_verbose:
            t_2 = time.time()
        loss.backward()
        if args.time_verbose:
            t_3 = time.time()
        optimizer.step()
        t1 = time.time()

        if iteration % 1 == 0:
            if args.time_verbose:
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.data + ' forward: %.4f sec.' % (t_1 - t0) +
                      ' loss calc: %.4f sec.' % (t_2 - t_1) + ' backward: %.4f sec.' % (t_3 - t_2) +
                      ' weight update: %.4f sec.' % (t1 - t_3))
            else:
                print('iter ' + repr(iteration) + ' || Loss: %.4f ||' % loss.data + ' time: %.4f sec.' % (t1 - t0))
            if loss.data > 100000:
                send_train_error()
                raise Exception(f'seed {SEED} is not promising. try another.')

        if iteration != 0 and iteration % 10000 == 0:
            print('Saving state, iter:', iteration)
            weight_path = args.save_folder + args.weight_name + '_' + repr(iteration)
            while os.path.isfile(weight_path + '.pth'):
                weight_path += '_add'
            torch.save(ssd_net.state_dict(), weight_path + '.pth')

        if args.one_epoch and iteration > epoch_size + 1:
            print('One epoch reached: exiting training...')
            return 0
    torch.save(ssd_net.state_dict(),
               args.save_folder + args.weight_name + '_full.pth')

    send_train_finished(args.ensure_archi, args.matching_strategy, args.ensure_size, args.augmentation, args.dataset)


def set_learning_rate_decay(optimizer, gamma, step):
    """Sets the learning rate to the initial LR decayed by 10 at every
        specified step
    # Adapted from PyTorch Imagenet example:
    # https://github.com/pytorch/examples/blob/master/imagenet/main.py
    """
    lr = args.lr * (gamma ** (step))
    print(f'adjusting learning rate to {lr}')
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def set_learning_rate(optimizer, lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def xavier(param):
    init.xavier_uniform_(param)


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        xavier(m.weight.data)
        m.bias.data.zero_()


if __name__ == '__main__':
    train()
