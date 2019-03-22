import argparse
import os
import logging
import sys
import itertools

import torch
from torch.utils.data import DataLoader, ConcatDataset
from torch.optim.lr_scheduler import CosineAnnealingLR, MultiStepLR

from vision.utils.misc import str2bool, Timer, freeze_net_layers, store_labels
from vision.ssd.ssd import MatchPrior
from vision.ssd.vgg_ssd import create_vgg_ssd
from vision.ssd.mobilenetv1_ssd import create_mobilenetv1_ssd
from vision.ssd.mobilenetv1_ssd_lite import create_mobilenetv1_ssd_lite
from vision.ssd.mobilenet_v2_ssd_lite import create_mobilenetv2_ssd_lite
from vision.ssd.squeezenet_ssd_lite import create_squeezenet_ssd_lite
from vision.datasets.voc_dataset import VOCDataset
from vision.datasets.open_images import OpenImagesDataset
from vision.nn.multibox_loss import MultiboxLoss
from vision.ssd.config import vgg_ssd_config
from vision.ssd.config import mobilenetv1_ssd_config
from vision.ssd.config import squeezenet_ssd_config
from vision.ssd.data_preprocessing import TrainAugmentation, TestTransform

from tensorboardX import SummaryWriter

parser = argparse.ArgumentParser(description='Single Shot MultiBox Detector Training With Pytorch')

parser.add_argument("--dataset_type",       default="voc", type=str,        help='Specify dataset type. Currently support voc and open_images.')

parser.add_argument('--datasets',           nargs='+',                      help='Dataset directory path')
parser.add_argument('--validation_dataset',                                 help='Dataset directory path')
parser.add_argument('--balance_data',       action='store_true',            help="Balance training data by down-sampling more frequent labels.")

parser.add_argument('--net',                default="vgg16-ssd",            help="The network architecture, it can be mb1-ssd, mb1-lite-ssd, mb2-ssd-lite or vgg16-ssd.")
parser.add_argument('--freeze_base_net',    action='store_true',            help="Freeze base net layers.")
parser.add_argument('--freeze_net',         action='store_true',            help="Freeze all the layers except the prediction head.")

parser.add_argument('--mb2_width_mult',     default=1.0, type=float,        help='Width Multiplifier for MobilenetV2')

# Params for SGD
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float,    help='initial learning rate')
parser.add_argument('--momentum',           default=0.9, type=float,        help='Momentum value for optim')
parser.add_argument('--weight_decay',       default=5e-4, type=float,       help='Weight decay for SGD')
parser.add_argument('--gamma',              default=0.1, type=float,        help='Gamma update for SGD')
parser.add_argument('--base_net_lr',        default=None, type=float,       help='initial learning rate for base net.')
parser.add_argument('--extra_layers_lr',    default=None, type=float,       help='initial learning rate for the layers not in base net and prediction heads.')

# Params for loading pretrained basenet or checkpoints.
parser.add_argument('--base_net',                                           help='Pretrained base model')
parser.add_argument('--pretrained_ssd',                                     help='Pre-trained base model')

# Scheduler
parser.add_argument('--scheduler',          default="multi-step", type=str, help="Scheduler for SGD. It can one of multi-step and cosine")

# Params for Multi-step Scheduler
parser.add_argument('--milestones',         default="80,100", type=str,     help="milestones for MultiStepLR")

# Params for Cosine Annealing
parser.add_argument('--t_max',              default=120, type=float,        help='T_max value for Cosine Annealing Scheduler.')

# Train params
parser.add_argument('--batch_size',         default=32, type=int,           help='Batch size for training')
parser.add_argument('--num_epochs',         default=120, type=int,          help='the number epochs')
parser.add_argument('--num_workers',        default=16, type=int,           help='Number of workers used in dataloading')
parser.add_argument('--validation_epochs',  default=5, type=int,            help='the number epochs')
parser.add_argument('--debug_steps',        default=50, type=int,           help='Set the debug log output frequency.')
parser.add_argument('--use_cuda',           default=True, type=str2bool,    help='Use CUDA to train model')


parser.add_argument('--exp_time',           default=None, type=str,  help='set if you want to use exp time')
parser.add_argument('--exp_name',           default=None, type=str,  help='set if you want to use exp name')
parser.add_argument('--seed',               default=None, type=int,            help='Manual seed')
parser.add_argument('--resume', '-r',       default=None, type=str,         help='resume from checkpoint')
parser.add_argument('--port',               default=8804,                   help='port binding for tensorboardX')

args = parser.parse_args()

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() and args.use_cuda else "cpu")

from config import initialize_logger, ROOT_LOGGER_NAME
logger, args = initialize_logger(args)
logger_train = logging.getLogger( ROOT_LOGGER_NAME + '.train' )
logger_test = logging.getLogger( ROOT_LOGGER_NAME + '.test' )

if args.seed is not None:
    # https://github.com/pytorch/tutorials/blob/master/beginner_source/blitz/data_parallel_tutorial.py
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

if not torch.backends.cudnn.deterministic:
    logger.debug('You have chosen to seed training. '
                'This will turn on the CUDNN deterministic setting, '
                'which can slow down your training considerably! '
                'You may see unexpected behavior when restarting '
                'from checkpoints.')

if args.use_cuda and torch.cuda.is_available():
    # https://discuss.pytorch.org/t/what-does-torch-backends-cudnn-benchmark-do/5936/3
    torch.backends.cudnn.benchmark = True

    logger.debug('benchmark mode is good whenever your input sizes for your network do not vary. '
                'This way, cudnn will look for the optimal set of algorithms for that particular configuration (which takes some time). '
                'This usually leads to faster runtime. '
                'But if your input sizes changes at each iteration, then cudnn will benchmark every time a new size appears, '
                'possibly leading to worse runtime performances.')
    

### tensorboard
writer = SummaryWriter(os.path.join(args.jobs_dir, 'tensorboardX'))

def train(loader, net, criterion, optimizer, device, debug_steps=100, epoch=-1):
    logger_train.info('')
    net.train(True)
    total_loss = 0.0
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0
    N = len(loader)
    for batch_idx, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)
        
        optimizer.zero_grad()
        confidence, locations = net(images)
        regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)  # TODO CHANGE BOXES
        loss = regression_loss + classification_loss
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()
        if (batch_idx+1) % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logger_train.info(
                f"Epoch: {epoch:3d}, Step: [{batch_idx:4d}/{N:4d}], " +
                f"Avg Loss: {avg_loss:.4f}, " +
                f"Avg Reg. Loss {avg_reg_loss:.4f}, " +
                f"Avg Cls. Loss: {avg_clf_loss:.4f}"
            )

            writer.add_scalars('train/avg_loss', {'loss': avg_loss/debug_steps}, epoch*len(loader)+batch_idx )
            writer.add_scalars('train/avg_reg_loss', {'loss': avg_reg_loss/debug_steps}, epoch*len(loader)+batch_idx )
            writer.add_scalars('train/avg_cls_loss', {'loss': avg_clf_loss/debug_steps}, epoch*len(loader)+batch_idx )
            
            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

    return total_loss / N

def test(loader, net, criterion, device, debug_steps, epoch):
    logger_test.info('')
    net.eval()

    total_loss = 0.0
    running_loss = 0.0
    running_regression_loss = 0.0
    running_classification_loss = 0.0    
    N = len(loader)

    for batch_idx, data in enumerate(loader):
        images, boxes, labels = data
        images = images.to(device)
        boxes = boxes.to(device)
        labels = labels.to(device)          

        with torch.no_grad():
            confidence, locations = net(images)
            regression_loss, classification_loss = criterion(confidence, locations, labels, boxes)
            loss = regression_loss + classification_loss

        total_loss += loss.item()
        running_loss += loss.item()
        running_regression_loss += regression_loss.item()
        running_classification_loss += classification_loss.item()

        if (batch_idx+1) % debug_steps == 0:
            avg_loss = running_loss / debug_steps
            avg_reg_loss = running_regression_loss / debug_steps
            avg_clf_loss = running_classification_loss / debug_steps
            logger_test.info(
                f"Epoch: {epoch:3d}, Step: [{batch_idx:4d}/{N:4d}], " +
                f"Avg Loss: {avg_loss:.4f}, " +
                f"Avg Reg. Loss {avg_reg_loss:.4f}, " +
                f"Avg Cls. Loss: {avg_clf_loss:.4f}"
            )

            running_loss = 0.0
            running_regression_loss = 0.0
            running_classification_loss = 0.0

    return total_loss / N


if __name__ == '__main__':
    timer = Timer()
    
    if args.net == 'vgg16-ssd':
        create_net = create_vgg_ssd
        config = vgg_ssd_config
    elif args.net == 'mb1-ssd':
        create_net = create_mobilenetv1_ssd
        config = mobilenetv1_ssd_config
    elif args.net == 'mb1-ssd-lite':
        create_net = create_mobilenetv1_ssd_lite
        config = mobilenetv1_ssd_config
    elif args.net == 'sq-ssd-lite':
        create_net = create_squeezenet_ssd_lite
        config = squeezenet_ssd_config
    elif args.net == 'mb2-ssd-lite':
        create_net = lambda num: create_mobilenetv2_ssd_lite(num, width_mult=args.mb2_width_mult)
        config = mobilenetv1_ssd_config
    else:
        logger.fatal("The net type is wrong.")
        parser.print_help(sys.stderr)
        sys.exit(1)
    train_transform = TrainAugmentation(config.image_size, config.image_mean, config.image_std, config.image_mean_tensor, config.image_stds_tensor)
    target_transform = MatchPrior(config.priors, config.center_variance,
                                  config.size_variance, 0.5)

    test_transform = TestTransform(config.image_size, config.image_mean, config.image_std, config.image_mean_tensor, config.image_stds_tensor)

    logger.info("Prepare training datasets.")
    datasets = []
    for dataset_path in args.datasets:
        if args.dataset_type == 'voc':
            dataset = VOCDataset(dataset_path, transform=train_transform,
                                 target_transform=target_transform)
            label_file = os.path.join(args.jobs_dir, "voc-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            num_classes = len(dataset.class_names)
        elif args.dataset_type == 'open_images':
            dataset = OpenImagesDataset(dataset_path,
                 transform=train_transform, target_transform=target_transform,
                 dataset_type="train", balance_data=args.balance_data)
            label_file = os.path.join(args.jobs_dir, "open-images-model-labels.txt")
            store_labels(label_file, dataset.class_names)
            logging.info(dataset)
            num_classes = len(dataset.class_names)

        else:
            raise ValueError(f"Dataset tpye {args.dataset_type} is not supported.")
        datasets.append(dataset)
    logger.info(f"Stored labels into file {label_file}.")
    train_dataset = ConcatDataset(datasets)    
    logger.info("Train dataset size: {}".format(len(train_dataset)))
    train_loader = DataLoader(train_dataset, args.batch_size,
                              num_workers=args.num_workers,
                              shuffle=True)
    logger.info("Prepare Validation datasets.")
    if args.dataset_type == "voc":
        val_dataset = VOCDataset(args.validation_dataset, transform=test_transform,
                                 target_transform=target_transform, is_test=True)
    elif args.dataset_type == 'open_images':
        val_dataset = OpenImagesDataset(dataset_path,
                                        transform=test_transform, target_transform=target_transform,
                                        dataset_type="test")
        logger.info(val_dataset)
    logger.info("validation dataset size: {}".format(len(val_dataset)))

    val_loader = DataLoader(val_dataset, args.batch_size,
                            num_workers=args.num_workers,
                            shuffle=False)
    logger.info("Build network.")
    net = create_net(num_classes)
    min_loss = -10000.0
    last_epoch = -1

    base_net_lr = args.base_net_lr if args.base_net_lr is not None else args.lr
    extra_layers_lr = args.extra_layers_lr if args.extra_layers_lr is not None else args.lr
    if args.freeze_base_net:
        logger.info("Freeze base net.")
        freeze_net_layers(net.base_net)
        params = itertools.chain(net.source_layer_add_ons.parameters(), net.extras.parameters(),
                                 net.regression_headers.parameters(), net.classification_headers.parameters())        
        params = [
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]
    elif args.freeze_net:
        freeze_net_layers(net.base_net)
        freeze_net_layers(net.source_layer_add_ons)
        freeze_net_layers(net.extras)
        params = itertools.chain(net.regression_headers.parameters(), net.classification_headers.parameters())
        logger.info("Freeze all the layers except prediction heads.")
    else:
        params = [
            {'params': net.base_net.parameters(), 'lr': base_net_lr},
            {'params': itertools.chain(
                net.source_layer_add_ons.parameters(),
                net.extras.parameters()
            ), 'lr': extra_layers_lr},
            {'params': itertools.chain(
                net.regression_headers.parameters(),
                net.classification_headers.parameters()
            )}
        ]

    timer.start("Load Model")
    if args.resume:
        logger.info(f"Resume from the model {args.resume}")
        net.load(args.resume)
    elif args.base_net:
        logger.info(f"Init from base net {args.base_net}")
        net.init_from_base_net(args.base_net)
    elif args.pretrained_ssd:
        logger.info(f"Init from pretrained ssd {args.pretrained_ssd}")
        net.init_from_pretrained_ssd(args.pretrained_ssd)
    logger.info(f'Took {timer.end("Load Model"):.2f} seconds to load the model.')
    
    net = torch.nn.DataParallel(net)
    net = net.to(DEVICE)
    

    criterion = MultiboxLoss(iou_threshold=0.5, neg_pos_ratio=3, center_variance=0.1, size_variance=0.2).to(DEVICE)
    optimizer = torch.optim.SGD(params, lr=args.lr, momentum=args.momentum,
                                weight_decay=args.weight_decay)
    logger.info(f"Learning rate: {args.lr}, Base net learning rate: {base_net_lr}, "
                 + f"Extra Layers learning rate: {extra_layers_lr}.")

    if args.scheduler == 'multi-step':
        logger.info("Uses MultiStepLR scheduler.")
        milestones = [int(v.strip()) for v in args.milestones.split(",")]
        scheduler = MultiStepLR(optimizer, milestones=milestones,
                                                     gamma=0.1, last_epoch=last_epoch)
    elif args.scheduler == 'cosine':
        logger.info("Uses CosineAnnealingLR scheduler.")
        scheduler = CosineAnnealingLR(optimizer, args.t_max, last_epoch=last_epoch)
    else:
        logger.fatal(f"Unsupported Scheduler: {args.scheduler}.")
        parser.print_help(sys.stderr)
        sys.exit(1)

    logger.info(f"Start training from epoch {last_epoch + 1}.")
    for epoch in range(last_epoch + 1, args.num_epochs):
        scheduler.step()
        train_loss = train(train_loader, net, criterion, optimizer,
              device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)        
        writer.add_scalars('loss_epoch', {'train': train_loss}, epoch)

        if epoch % args.validation_epochs == 0 or epoch == args.num_epochs - 1:
            val_loss = test(val_loader, net, criterion, device=DEVICE, debug_steps=args.debug_steps, epoch=epoch)
            writer.add_scalars('loss_epoch', {'val': val_loss}, epoch)
            # val_loss, val_regression_loss, val_classification_loss = test(val_loader, net, criterion, DEVICE)
            # logger.info(
            #     f"Epoch: {epoch}, " +
            #     f"Validation Loss: {val_loss:.4f}, " +
            #     f"Validation Regression Loss {val_regression_loss:.4f}, " +
            #     f"Validation Classification Loss: {val_classification_loss:.4f}"
            # )
            # model_path = os.path.join(args.checkpoint_folder, f"{args.net}-Epoch-{epoch}-Loss-{val_loss}.pth")
            model_path = os.path.join(args.jobs_dir, "snapshots", f"{args.net}-Epoch-{epoch}-Loss-{val_loss:.4f}.pth")
            # net.save(model_path)
            torch.save(net.module.state_dict(), model_path)
            logger.info(f"Saved model {model_path}")
