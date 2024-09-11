from models import *
import time
import torch
from config import Config
from generate_data import generate_data

def str2model(name):
    d = {
        'deit_tiny': deit_tiny_patch16_224,
        'deit_small': deit_small_patch16_224,
        'deit_base': deit_base_patch16_224,
        'vit_base': vit_base_patch16_224,
        'vit_large': vit_large_patch16_224,
        'swin_tiny': swin_tiny_patch4_window7_224,
        'swin_small': swin_small_patch4_window7_224,
        'swin_base': swin_base_patch4_window7_224,
    }
    print('Model: %s' % d[name].__name__)
    return d[name]


def seed(seed=0):
    import os
    import random
    import sys

    import numpy as np
    import torch
    sys.setrecursionlimit(100000)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    np.random.seed(seed)
    random.seed(seed)
class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
def accuracy(output, target, topk=(1, )):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res



def validate(args, val_loader, model, criterion, device, bit_config=None):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    val_start_time = end = time.time()
    for i, (data, target) in enumerate(val_loader):
        data = data.to(device)
        target = target.to(device)
        if i == 0:
            plot_flag = False
        else:
            plot_flag = False
        with torch.no_grad():
            output, FLOPs, distance = model(data, bit_config, plot_flag)
        loss = criterion(output, target)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data.item(), data.size(0))
        top1.update(prec1.data.item(), data.size(0))
        top5.update(prec5.data.item(), data.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i,
                      len(val_loader),
                      batch_time=batch_time,
                      loss=losses,
                      top1=top1,
                      top5=top5,
                  ))
    val_end_time = time.time()
    print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.
          format(top1=top1, top5=top5, time=val_end_time - val_start_time))

    return losses.avg, top1.avg, top5.avg
def model_make(model_name, ptf, lis, quant_method, device):
    device = torch.device(device)
    cfg = Config(ptf, lis, quant_method)
    model = str2model(model_name)(pretrained=True, cfg=cfg)
    model = model.to(device)
    return model
    
def calibrate_model(mode = 0, args = None, model = None, train_loader = None, device = None):
    if mode == 2:
        print("Generating data...")
        calibrate_data = generate_data(args)
        print("Calibrating with generated data...")
        model.model_open_calibrate()
        with torch.no_grad():
            model.model_open_last_calibrate()
            output = model(calibrate_data)
        return model
    # Case 1: Gaussian noise
    elif args.mode == 1:
        calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)
        print("Calibrating with Gaussian noise...")
        model.model_open_calibrate()
        with torch.no_grad():
            model.model_open_last_calibrate()
            output = model(calibrate_data)
        return model
    # Case 2: Real data (Standard)
    elif args.mode == 0:
        # Get calibration set.
        image_list = []
        # output_list = []
        for i, (data, target) in enumerate(train_loader):
            if i == args.calib_iter:
                break
            data = data.to(device)
            # target = target.to(device)
            image_list.append(data)
            # output_list.append(target)

        print("Calibrating with real data...")
        model.model_open_calibrate()
        with torch.no_grad():
            # TODO:
            # for i, image in enumerate(image_list):
            #     if i == len(image_list) - 1:
            #         # This is used for OMSE method to
            #         # calculate minimum quantization error
            #         model.model_open_last_calibrate()
            #     output, FLOPs, global_distance = model(image, plot=False)
            # model.model_quant(flag='off')
            model.model_open_last_calibrate()
            output, FLOPs, global_distance = model(image_list[0], plot=False)

    model.model_close_calibrate()
    model.model_quant()
    return model