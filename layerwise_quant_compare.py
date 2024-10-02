import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '3'
import time
import random
import torch.nn.functional as F
from model_utility import *
from dataset_utility import *
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from config import Config
from models import *
import numpy as np
import itertools

parser = argparse.ArgumentParser(description='FQ-ViT')

parser.add_argument('--model', choices=['deit_tiny', 'deit_small', 'deit_base', 'vit_base', 'vit_large', 'swin_tiny', 'swin_small', 'swin_base'], default='deit_tiny', help='model')
parser.add_argument('--data', metavar='DIR', default='/data/deepops/temp/easy-lora-and-gptq/imagenet', help='path to dataset')
parser.add_argument('--quant', default=True, action='store_true')
parser.add_argument('--ptf', default=True)
parser.add_argument('--lis', default=True)
parser.add_argument('--quant-method', default='minmax', choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--mixed', default=True, action='store_true')
parser.add_argument('--calib-batchsize', default=10, type=int, help='batchsize of calibration set')
parser.add_argument("--mode", default=0, type=int, help="mode of calibration data, 0: PSAQ-ViT, 1: Gaussian noise, 2: Real data")
parser.add_argument('--calib-iter', default=10, type=int)
parser.add_argument('--val-batchsize', default=20, type=int, help='batchsize of validation set')
parser.add_argument('--num-workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq', default=100, type=int, help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')

args = parser.parse_args(args=[])
seed(args.seed)

device = torch.device(args.device)
cfg = Config(args.ptf, args.lis, args.quant_method)

# Note: Different models have different strategies of data preprocessing.
model_type = args.model.split('_')[0]
if model_type == 'deit':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_pct = 0.875
elif model_type == 'vit':
    mean = (0.5, 0.5, 0.5)
    std = (0.5, 0.5, 0.5)
    crop_pct = 0.9
elif model_type == 'swin':
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)
    crop_pct = 0.9
else:
    raise NotImplementedError

train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
val_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)

# Data
traindir = os.path.join(args.data, 'train')
valdir = os.path.join(args.data, 'val')

val_dataset = datasets.ImageFolder(valdir, val_transform)
val_loader = torch.utils.data.DataLoader(
    val_dataset,
    batch_size=args.val_batchsize,
    shuffle=False,
    num_workers=args.num_workers,
    pin_memory=True,
)

# define loss function (criterion)
criterion = nn.CrossEntropyLoss().to(device)

train_dataset = datasets.ImageFolder(traindir, train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=10,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
)

int4_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
int4_model = calibrate_model(args.mode, args, int4_model, train_loader, device)
int4_model.eval()

def generate_restore_indices_combinations():
    # 직접 지정한 10개의 조합
    combinations = {
    5: [
        (15, 16, 31, 35, 5), 
        (13, 18, 40, 24, 14), 
        (0, 40, 36, 38, 10), 
        (3, 14, 16, 27, 34), 
        (5, 11, 19, 28, 40), 
        (6, 8, 22, 28, 39),
        (7, 15, 24, 31, 35), 
        (2, 10, 19, 25, 32), 
        (0, 40, 1, 39, 2), 
        (27, 8, 18, 29, 37), 
        (24, 40, 36, 38, 10)
    ],
    6: [
        (1, 2, 9, 13, 25, 36),
        (3, 6, 12, 17, 28, 40),
        (4, 10, 15, 21, 29, 33),
        (5, 11, 19, 23, 31, 37),
        (0, 7, 14, 18, 26, 34),
        (2, 8, 16, 24, 32, 39),
        (1, 5, 13, 22, 27, 35),
        (3, 9, 17, 20, 30, 38),
        (4, 6, 15, 25, 33, 40),
        (0, 10, 18, 23, 28, 36)
    ],
    
    7: [
        (1, 3, 8, 12, 19, 28, 37),
        (0, 5, 11, 17, 23, 31, 34),
        (2, 6, 10, 16, 22, 27, 35),
        (4, 9, 14, 18, 25, 32, 40),
        (3, 7, 13, 20, 26, 30, 38),
        (1, 5, 12, 15, 24, 29, 36),
        (2, 8, 16, 21, 25, 31, 39),
        (0, 4, 11, 18, 23, 27, 33),
        (6, 9, 14, 19, 24, 32, 37),
        (1, 7, 13, 22, 28, 34, 40)
    ],

    8: [
        (0, 3, 6, 12, 18, 23, 29, 35),
        (2, 4, 10, 14, 19, 25, 32, 40),
        (1, 5, 11, 16, 21, 28, 33, 37),
        (3, 7, 13, 20, 26, 31, 36, 39),
        (2, 9, 15, 17, 22, 27, 30, 38),
        (1, 4, 8, 14, 19, 24, 30, 40),
        (5, 9, 13, 22, 25, 29, 34, 37),
        (0, 6, 11, 17, 23, 27, 32, 38),
        (2, 8, 12, 18, 24, 29, 33, 40),
        (3, 7, 10, 15, 20, 26, 35, 39)
    ],

    9: [
        (1, 3, 5, 12, 16, 21, 27, 32, 38),
        (2, 4, 7, 10, 18, 23, 29, 34, 39),
        (0, 5, 9, 13, 19, 25, 30, 33, 37),
        (3, 6, 11, 15, 20, 24, 31, 35, 40),
        (2, 8, 14, 17, 22, 28, 30, 36, 39),
        (1, 4, 10, 16, 19, 26, 29, 32, 40),
        (0, 7, 12, 18, 22, 27, 31, 35, 38),
        (5, 9, 13, 20, 23, 30, 34, 37, 40),
        (6, 11, 14, 18, 21, 26, 32, 36, 39),
        (3, 8, 12, 17, 22, 25, 29, 31, 38)
    ],
    10: [
        (36, 34, 3, 27, 35, 30, 29, 33, 12, 23),
        (39, 17, 22, 24, 16, 18, 21, 26, 38, 13), 
        (12, 23, 25, 11, 28, 37, 32, 6, 10, 20), 
        (12, 23, 25, 11, 28, 37, 31, 6, 10, 20), 
        (39, 17, 22, 24, 16, 18, 21, 26, 38, 6), 
        (3, 17, 18, 21, 22, 14, 25, 30, 34, 38),
        (31, 5, 6, 10, 20, 7, 8, 9, 19, 15), 
        (1, 2, 3, 4, 6, 5, 7, 8, 9, 10), 
        (11, 12, 13, 14, 15, 16, 17, 18, 19, 20), 
        (21, 22, 23, 24, 25, 26, 27, 28, 29, 30), 
        (31, 32, 33, 34, 35, 36, 37, 38, 39, 40)
    ],
    
    20: [
        (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40),
        (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39), 
        (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39), 
        (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 37, 38, 39), 
        (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39), 
        (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 39)
    ]
}

    return combinations

def run_int4_baseline(int4_model, val_loader, device, criterion, result_file):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_start_time = end = time.time()

    for i, (inputs, labels) in enumerate(val_loader):
        inputs = inputs.to(device)
        labels = labels.to(device)

        four_bit_config = [4] * 50

        with torch.no_grad():
            output, FLOPs, distance = int4_model(inputs, four_bit_config, False)
        loss = criterion(output, labels)

        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.data.item(), inputs.size(0))
        top5.update(prec5.data.item(), inputs.size(0))

        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
            print('Test: [{0}/{1}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                      i, len(val_loader), batch_time=batch_time,
                      loss=losses, top1=top1, top5=top5))

    val_end_time = time.time()
    result_string = ' * Restore Index: nothing, Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.format(
        top1=top1, top5=top5, time=val_end_time - val_start_time)
    print(result_string)
    
    with open(result_file, 'w') as f:
        f.write(result_string + '\n')

    torch.cuda.empty_cache()
    
def run_experiments(combinations, int4_model, val_loader, device, criterion):
    for restore_count, restore_indices_list in combinations.items():
        result_file = f"restore_{restore_count}_layers.txt"
        
        run_int4_baseline(int4_model, val_loader, device, criterion, result_file)
        
       
        for combination_index, restore_indices in enumerate(restore_indices_list):
            batch_time = AverageMeter()
            losses = AverageMeter()
            top1 = AverageMeter()
            top5 = AverageMeter()
            val_start_time = end = time.time()

            for i, (inputs, labels) in enumerate(val_loader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                four_bit_config = [4] * 50
                for idx in restore_indices:
                    four_bit_config[idx] = 8

                with torch.no_grad():
                    output, FLOPs, distance = int4_model(inputs, four_bit_config, False)
                loss = criterion(output, labels)

                prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
                losses.update(loss.data.item(), inputs.size(0))
                top1.update(prec1.data.item(), inputs.size(0))
                top5.update(prec5.data.item(), inputs.size(0))

                batch_time.update(time.time() - end)
                end = time.time()

                if i % 10 == 0:
                    print(f'Test: [{i}/{len(val_loader)}]\t'
                            f'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                            f'Loss {losses.val:.4f} ({losses.avg:.4f})\t'
                            f'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                            f'Prec@5 {top5.val:.3f} ({top5.avg:.3f})')

            val_end_time = time.time()
            result_string = f' * Combination {combination_index + 1}, Restore Indices: {restore_indices}, ' \
                            f'Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {val_end_time - val_start_time:.3f}'
            print(result_string)
            with open(result_file, 'a') as f:
                f.write(result_string + '\n')
            
            torch.cuda.empty_cache()

        print(f"Results for {restore_count} restored layers have been saved to {result_file}")

if __name__ == "__main__":
    combinations = generate_restore_indices_combinations()
    run_experiments(combinations, int4_model, val_loader, device, criterion)