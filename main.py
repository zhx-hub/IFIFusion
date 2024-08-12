import argparse
import datetime
import numpy as np
import time
import torch
import torch.backends.cudnn as cudnn
import json
from torchvision import transforms
from pathlib import Path

from timm.data import Mixup
from timm.models import create_model
from timm.loss import LabelSmoothingCrossEntropy, SoftTargetCrossEntropy
from timm.scheduler import create_scheduler
from timm.optim import create_optimizer
from timm.utils import NativeScaler, get_state_dict, ModelEma
from data.mt_voc import MTVOCSegmentation
from models.Net_deit_s import Net
import utils
import yaml
from engine import train_one_epoch, evaluate
import tools.optimizer as module_optimizer
from tools.scheduler import _setup_param_groups, get_instance

def get_args_parser():
    parser = argparse.ArgumentParser('DeiT training and evaluation script', add_help=False)
    parser.add_argument('--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--bce-loss', action='store_true')
    parser.add_argument('--unscale-lr', action='store_true')
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument('--cfg', type=str, default='../expri1.yml', help='hyperparameters path')
    # Model parameters

    parser.add_argument("--ck_path", type=str, default="../deit_small_patch16.pth")
    parser.add_argument('--lr', type=float, default=5e-4, metavar='LR',
                        help='learning rate (default: 5e-4)')    
    #Dataset
    parser.add_argument('--base-size', type=int, default=None,#520
                        help='base image short width size')
    parser.add_argument('--crop-size', type=int, default=None,# 450
                        help='crop image size')
    parser.add_argument('--pad-size', type=int, default=None,# data padding
                        help='padding image size')
    parser.add_argument('--scale-ratio', type=float, default=0.8,# image scale
                        help='scale image size')

    parser.add_argument("--clip_grad",  type=float, default=None, metavar='NORM',
                        help='Clip gradient norm (default: None, no clipping)')

    parser.add_argument("--data_path", type=str, default="../MT-Defect/")
    parser.add_argument('--device', default='cuda',
                        help='device to use for training / testing')
    parser.add_argument('--start_epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument('--mean', default=(0.485, 0.456, 0.406))
    parser.add_argument('--std', default=(0.229, 0.224, 0.225))
    parser.add_argument('--output_dir', default="../test/",
                    help='path where to save, empty for no saving')
    parser.add_argument('--resume', default="", help='resume from checkpoint')
    parser.add_argument('--num_workers', default=6, type=int)
    parser.add_argument('--pin-mem', action='store_true',
                        help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
    parser.add_argument('--no-pin-mem', action='store_false', dest='pin_mem',
                        help='')
    parser.set_defaults(pin_mem=True) 

    parser.add_argument('--world_size', default=8, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    
    return parser

def main(args):

    utils.init_distributed_mode(args)
    device = torch.device(args.device)

    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)


    cudnn.benchmark = True

    # image transform 
    input_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([.485, .456, .406], [.229, .224, .225]),
    ])
    # dataset and dataloader
    data_kwargs = {'transform': input_transform, 'base_size': args.base_size, 
                    'crop_size': args.crop_size, 'pad_size': (280,280), 'scale_ratio':args.scale_ratio}

    dataset_train = MTVOCSegmentation(args.data_path, split='train', mode='train', **data_kwargs)
    dataset_val = MTVOCSegmentation(args.data_path, split='val', mode='val', **data_kwargs)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, 
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,

    )

    data_loader_val = torch.utils.data.DataLoader(
        dataset_val, 
        batch_size=int(1.5 * args.batch_size),
        num_workers=args.num_workers,
        pin_memory=True,
        shuffle=True,

    )

    model = Net(ckpt=args.ck_path ,img_size=(384,384),num_classes=5)
    model.init_weights()
    
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print('number of params:', n_parameters)
    
    
    f = open(args.cfg)
    cfg = yaml.load(f,Loader=yaml.FullLoader)
    params = _setup_param_groups(model, cfg)
    optimizer = get_instance(module_optimizer,  'optimizer', cfg, params)
    lr_scheduler =  torch.optim.lr_scheduler.CosineAnnealingLR(optimizer = optimizer,T_max=100) 
    criterion=loss = torch.nn.BCEWithLogitsLoss()
    loss_scaler = NativeScaler()

    model = model.to(device)
    output_dir = Path(args.output_dir)


    if args.resume:

        checkpoint = torch.load(args.resume, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
        if not args.eval and 'optimizer' in checkpoint and 'lr_scheduler' in checkpoint and 'epoch' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            args.start_epoch = checkpoint['epoch'] + 1
            if 'scaler' in checkpoint:
                loss_scaler.load_state_dict(checkpoint['scaler'])
        lr_scheduler.step(args.start_epoch)


    print(f"Start training for {args.epochs} epochs")
    start_time = time.time()
    best_loss = 10.0
    epoch1=0.0


    multi_list = []
    for epoch in range(args.start_epoch, args.epochs):

        print('LR base: {}, LR head: {},LR1 head: {},LR2 head: {}'.format(optimizer.param_groups[0]['lr'],
                                                optimizer.param_groups[1]['lr'],
                                               optimizer.param_groups[2]['lr'],
                                               optimizer.param_groups[3]['lr']))

        if args.distributed:
            data_loader_train.sampler.set_epoch(epoch)

        train_stats = train_one_epoch(
            model, criterion, data_loader_train,
            optimizer, device, epoch, loss_scaler,
            args.clip_grad,
            args = args,
        )
        lr_scheduler.step(epoch)
        multi_list.append(lr_scheduler.get_last_lr()[0])

        if args.output_dir:
            if epoch%199 ==0:
                checkpoint_paths = [output_dir / 'checkpoint{}.pth'.format(epoch)]
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)


        test_stats = evaluate(data_loader_val, model,criterion,device)

        if test_stats["loss"] < best_loss:
            best_loss = test_stats["loss"]
            if args.output_dir:
                checkpoint_paths = [output_dir / 'best_checkpoint1.pth']
                for checkpoint_path in checkpoint_paths:
                    utils.save_on_master({
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'lr_scheduler': lr_scheduler.state_dict(),
                        'epoch': epoch,
                        'scaler': loss_scaler.state_dict(),
                        'args': args,
                    }, checkpoint_path)
                epoch1 = epoch
        print(f'Best loss: {best_loss:.3f}%  epoch: {epoch1}')

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                     **{f'test_{k}': v for k, v in test_stats.items()},
                     'epoch': epoch,
                     'n_parameters': n_parameters}




        if args.output_dir and utils.is_main_process():
            with (output_dir / "log.txt").open("a") as f:
                f.write(json.dumps(log_stats) + "\n")

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str)) 
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser('DeiT training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    main(args)