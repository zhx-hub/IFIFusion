import math
import sys
from typing import Iterable, Optional
from tools.score import SegmentationMetric
import torch
import numpy as np
from timm.data import Mixup
import utils
from tools.loss import bce_loss
import torch.nn.functional as F


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)
    _, pred = output.topk(maxk, 1, True, True)
    tgt_val, tgt_idx = target.topk(maxk, 1, True, True)
    non_zero_idx=np.nonzero(tgt_val.flatten()).flatten()
    if len(non_zero_idx)==0:
        pred = pred.t()
        correct = pred.eq(tgt_idx.reshape(1, -1).expand_as(pred))
        result = [correct[:k].reshape(-1).float().sum(0) * 100. / batch_size for k in topk]
    else:
        pred = pred[non_zero_idx].t()
        correct = pred.eq(tgt_idx[non_zero_idx].reshape(1, -1).expand_as(pred))
        result = [correct[:k].reshape(-1).float().sum(0) * 100. / len(non_zero_idx) for k in topk]

    return result



def train_one_epoch(model: torch.nn.Module, criterion: bce_loss,
                    data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0,
                     mixup_fn: Optional[Mixup] = None,
                    set_training_mode=True, args = None):    
    
    
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    
    for samples, targets,se_label,_  in metric_logger.log_every(data_loader, print_freq, header):
        samples = samples.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)  
        se_label = se_label.to(device, non_blocking=True).float()


        if True:
            targets = targets.gt(0.0).type(targets.dtype)         
        with torch.cuda.amp.autocast():
       
            pred,cls = model(samples)
            scale = int(targets.size(-1) / pred[-2].size(-1))
            targets = targets.unsqueeze(1).gt(0.5).float()
   
            pred = torch.max(F.pixel_shuffle(pred, scale), dim=1).values.unsqueeze(1)     
            loss = criterion(pred, targets)+0.5*criterion(cls,se_label)

        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()


        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        loss_scaler(loss, optimizer, clip_grad=max_norm,
                    parameters=model.parameters(), create_graph=is_second_order)

        torch.cuda.synchronize()

        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}



def calculate_mae(img1, img2):
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)

    mae = np.mean(np.abs(img1 - img2))

    return mae
@torch.no_grad()
def evaluate(data_loader, model,loss, device):
    criterion = loss

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # switch to evaluation mode
    model.eval()
    mae =0
    count=0
    for images, target,se_label,_ in metric_logger.log_every(data_loader, 10, header):
        images = images.to(device, non_blocking=True)
        target = target.to(device, non_blocking=True)
        se_label = se_label.to(device, non_blocking=True).float() 

        if True:
            target = target.gt(0.0).type(target.dtype)
                    
        with torch.cuda.amp.autocast():
       
            pred,cls = model(images)
            scale = int(target.size(-1) / pred[-2].size(-1))             
            target = target.unsqueeze(1).gt(0.5).float()
            pred = torch.max(F.pixel_shuffle(pred, scale), dim=1).values.unsqueeze(1) 
            loss= criterion(pred, target)+0.5*criterion(cls,se_label)
           
        pred_sal = pred
        pred_sal = torch.sigmoid(pred_sal).squeeze().detach().cpu().numpy()

        acc1 = accuracy(cls, se_label)[0]
        count+=1


        batch_size = images.shape[0]
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)

    metric_logger.synchronize_between_processes()
    print('* Acc@1 {top1.global_avg:.3f}  loss {losses.global_avg:.3f}'
          .format(top1=metric_logger.acc1, losses=metric_logger.loss))

    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}
