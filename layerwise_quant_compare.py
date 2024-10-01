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
        (5, 10, 15, 20, 25), 
        (21, 24, 25, 38, 39), 
        (18, 22, 24, 34, 38), 
        (13, 21, 27, 30, 38), 
        (9, 18, 24, 34, 39), 
        (13, 18, 22, 30, 38),
        (5, 15, 23, 31, 37), 
        (8, 14, 26, 31, 37), 
        (11, 15, 20, 29, 35), 
        (6, 12, 19, 28, 33), 
        (7, 16, 25, 32, 36)
    ],
    10: [
        (5, 10, 15, 20, 25, 30, 35, 40),
        (9, 13, 18, 21, 22, 24, 25, 30, 34, 38), 
        (13, 17, 18, 21, 22, 24, 27, 30, 34, 39), 
        (9, 13, 18, 21, 24, 25, 27, 30, 38, 39), 
        (9, 13, 17, 18, 22, 24, 27, 34, 38, 39), 
        (13, 17, 18, 21, 22, 24, 25, 30, 34, 38),
        (5, 8, 11, 14, 15, 20, 23, 26, 31, 37), 
        (6, 7, 12, 14, 16, 19, 28, 32, 33, 36), 
        (5, 7, 11, 15, 19, 23, 29, 31, 35, 37), 
        (6, 8, 12, 14, 16, 20, 25, 28, 32, 36), 
        (7, 11, 14, 15, 20, 23, 26, 29, 33, 35)
    ]
    # 15: [
    #     (3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36, 39, 42),
    #     (9, 13, 17, 18, 21, 22, 24, 25, 27, 30, 34, 35, 36, 38, 39), 
    #     (9, 13, 17, 18, 21, 22, 24, 25, 27, 30, 32, 34, 36, 38, 39), 
    #     (9, 13, 17, 18, 21, 22, 24, 25, 27, 30, 33, 34, 35, 38, 39), 
    #     (9, 13, 17, 18, 21, 22, 24, 25, 27, 30, 32, 33, 34, 38, 39), 
    #     (9, 13, 17, 18, 21, 22, 24, 25, 27, 30, 31, 34, 36, 38, 39),
    #     (5, 6, 7, 8, 11, 12, 14, 15, 16, 19, 20, 23, 26, 28, 31), 
    #     (5, 6, 7, 8, 11, 12, 14, 15, 16, 19, 20, 23, 26, 29, 32), 
    #     (5, 6, 7, 8, 11, 12, 14, 15, 16, 19, 20, 23, 28, 31, 33), 
    #     (5, 6, 7, 8, 11, 12, 14, 15, 16, 19, 20, 25, 26, 28, 31), 
    #     (5, 6, 7, 8, 11, 12, 14, 15, 16, 19, 20, 23, 26, 29, 31)
    # ],
    # 20: [
    #     (2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30, 32, 34, 36, 38, 40),
    #     (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 36, 38, 39), 
    #     (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 36, 38, 39), 
    #     (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 32, 33, 34, 35, 37, 38, 39), 
    #     (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 32, 34, 35, 36, 38, 39), 
    #     (9, 10, 13, 17, 18, 21, 22, 24, 25, 27, 28, 29, 30, 31, 33, 34, 35, 37, 38, 39)
    # ]
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