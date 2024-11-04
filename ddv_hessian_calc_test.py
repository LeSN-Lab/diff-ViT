import argparse
import math
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import time
import random
import torch.nn.functional as F
from model_utility import *
from dataset_utility import *
import torch
import torch.nn as nn
import torchvision.datasets as datasets
from cka_utility import *
# from seaborn import heatmap

from config import Config
from models import *

import numpy as np
from plot import *
parser = argparse.ArgumentParser(description='FQ-ViT')

parser.add_argument('--model',
                    choices=[
                        'deit_tiny', 'deit_small', 'deit_base', 'vit_base',
                        'vit_large', 'swin_tiny', 'swin_small', 'swin_base'
                    ],
                    default='deit_tiny',
                    help='model')
parser.add_argument('--data', metavar='DIR',
                    default='/home/ubuntu/imagenet',
                    help='path to dataset')
parser.add_argument('--quant', default=True, action='store_true')
parser.add_argument('--ptf', default=False)
parser.add_argument('--lis', default=False)
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--mixed', default=True, action='store_true')
# TODO: 100 --> 32
parser.add_argument('--calib-batchsize',
                    default=50,
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
                    default=4,
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
criterion = nn.MSELoss().to(device)

train_dataset = datasets.ImageFolder(traindir, train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=args.calib_batchsize,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
)
# int8_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
int4_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
not_quantized_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)

# restore_indices = [8, 19]

# eight_bit_config = [8]*50
#basic_net, epsilon, step_size, num_steps, bit_config, args
not_quantized_attack_net = AttackPGD(
    basic_net=not_quantized_model,
    epsilon=0.06,
    step_size=0.01,
    num_steps=50,
    bit_config=None,
    args=args)

four_bit_config = [4]*50


# seed_images, seed_labels = not_quantized_attack_net.get_seed_inputs(5, rand=False)
# adv_inputs = not_quantized_attack_net.gen_adv_inputs(seed_images, seed_labels)

# int8_model = calibrate_model(args.mode, args, int8_model, train_loader, device)
# int4_model = calibrate_model(args.mode, args, int4_model, train_loader, device)


# int8_model.eval()
# int4_model.eval()
# not_quantized_model.eval()

print()

from pyhessian import DDVHessian
# Example: Select specific layers (e.g., layers 10, 20, 30)
selectedIndex = 30

print("Calculating the sensitivities via the averaged Hessian trace...")
batch_num = 10
trace_list = []
not_quantized_model.eval()
for i, (inputs, labels) in enumerate(train_loader):
    try:
        adv_inputs = not_quantized_attack_net.gen_adv_inputs(inputs, labels)
        inputs, targets = inputs.cuda(), labels.cuda()

        # Initialize the DDVHessian class with selected layers
        hessian_comp = DDVHessian(
            model=not_quantized_model,
            q_model=int4_model,
            criterion=torch.nn.MSELoss(),
            data=(inputs, labels),
            adv_data=(adv_inputs, labels),
            attack_net=not_quantized_attack_net,
            layer_indices=selectedIndex,
            cuda=args.device
        )
        print(f"Processing batch {i + 1}/{batch_num}")
        name, trace = hessian_comp.trace()
        trace_list.append(trace)
        # 명시적인 메모리 정리
        del hessian_comp
        del inputs
        del labels
        del adv_inputs
        del name
        del trace

        # 모델의 그래디언트 정리
        not_quantized_model.zero_grad()
        int4_model.zero_grad()

        # CUDA 캐시 비우기
        torch.cuda.empty_cache()

        if i == batch_num - 1:
            break

    except RuntimeError as e:
        print(f"Error occurred: {e}")
        # 에러 발생 시에도 메모리 정리
        torch.cuda.empty_cache()
        raise e
# Process the trace_list as needed


new_global_hessian_track = []
for i in range(int(len(trace_list))):
    hessian_track = trace_list[i]
    hessian_track = [abs(x) for x in hessian_track]
    min_h = min(hessian_track)
    max_h = max(hessian_track)
    averaged_hessian_track = [(elem-min_h)/(max_h-min_h) for elem in hessian_track]
    new_global_hessian_track.append(averaged_hessian_track)


# min_hessian = []
# max_hessian = []
layer_num = len(trace_list[0])
mean_hessian = []

for i in range(layer_num):
    new_hessian = [sample[i] for sample in new_global_hessian_track]
    mean_hessian.append(sum(new_hessian)/len(new_hessian))
    # min_hessian.append(min(new_hessian))
    # max_hessian.append(max(new_hessian))

# print(name)
print('\n***Trace: ', mean_hessian)