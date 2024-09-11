import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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

parser = argparse.ArgumentParser(description='FQ-ViT')

parser.add_argument('--model',
                    choices=[
                        'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                        'vit_large', 'swin_tiny', 'swin_small', 'swin_base'
                    ],
                    default='deit_tiny',
                    help='model')
parser.add_argument('--data', metavar='DIR',
                    default='/home/jieungkim/quantctr/imagenet',
                    help='path to dataset')
parser.add_argument('--quant', default=True, action='store_true')
parser.add_argument('--ptf', default=True)
parser.add_argument('--lis', default=True)
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--mixed', default=True, action='store_true')
# TODO: 100 --> 32
parser.add_argument('--calib-batchsize',
                    default=100,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument("--mode", default=0,
                        type=int, 
                        help="mode of calibration data, 0: PSAQ-ViT, 1: Gaussian noise, 2: Real data")
# TODO: 10 --> 1
parser.add_argument('--calib-iter', default=10, type=int)
# TODO: 100 --> 200
parser.add_argument('--val-batchsize',
                    default=200,
                    type=int,
                    help='batchsize of validation set')
parser.add_argument('--num-workers',
                    default=16,
                    type=int,
                    help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')
parser.add_argument('--print-freq',
                    default=100,
                    type=int,
                    help='print frequency')
parser.add_argument('--seed', default=0, type=int, help='seed')



args = parser.parse_args(args=[])
seed(args.seed)

device = torch.device(args.device)
cfg = Config(args.ptf, args.lis, args.quant_method)
# model = str2model(args.model)(pretrained=True, cfg=cfg)
# model = model.to(device)



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
# switch to evaluate mode
# model.eval()

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


int8_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
int4_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
not_quantized_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)

eight_bit_config = [8]*50
#basic_net, epsilon, step_size, num_steps, bit_config, args
not_quantized_attack_net = AttackPGD(
    basic_net=not_quantized_model, 
    epsilon=0.06,
    step_size=0.01,
    num_steps=50,
    bit_config=None,
    args=args)
four_bit_config = [4]*50
seed_images, seed_labels = not_quantized_attack_net.get_seed_inputs(50, rand=False)
adv_inputs = not_quantized_attack_net.gen_adv_inputs(seed_images, seed_labels)
# mutation_inputs = gen_profiling_inputs_in_blackbox(not_quantized_model, None,  int4_model, four_bit_config, seed_images, epsilon=0.02)




# int8_model = calibrate_model(args.mode, args, int8_model, train_loader, device)
int4_model = calibrate_model(args.mode, args, int4_model, train_loader, device)


# int8_model.eval()
int4_model.eval()
not_quantized_model.eval()

print()

result_file = "not_quantized_int4_restore_results.txt"


for restore_index in range(0, 50):
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    val_start_time = end = time.time()

    for i, (inputs, labels) in enumerate(val_loader):
        
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        four_bit_config = [4] * 50
        # four_bit_config = [-1] * 25
        # four_bit_config = four_bit_config + [4] * 25

        four_bit_config[restore_index] = -1
        labels = labels.to(device)
        
        with torch.no_grad():
            output, FLOPs, distance = int4_model(inputs, four_bit_config, False)
        loss = criterion(output, labels)

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output.data, labels, topk=(1, 5))
        losses.update(loss.data.item(), inputs.size(0))
        top1.update(prec1.data.item(), inputs.size(0))
        top5.update(prec5.data.item(), inputs.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % 10 == 0:
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
    result_string = ' * Restore Index: {idx}, Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f} Time {time:.3f}'.format(
        idx=restore_index, top1=top1, top5=top5, time=val_end_time - val_start_time)
        # 결과를 파일에 추가
    with open(result_file, 'a') as f:
        f.write(result_string + '\n')
print(f"Results have been saved to {result_file}")

    

