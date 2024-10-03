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
                    default='/home/lesn/quantctr/imagenet',
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
                    default=10,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument("--mode", default=0,
                        type=int, 
                        help="mode of calibration data, 0: PSAQ-ViT, 1: Gaussian noise, 2: Real data")
# TODO: 10 --> 1
parser.add_argument('--calib-iter', default=10, type=int)
# TODO: 100 --> 200
parser.add_argument('--val-batchsize',
                    default=10,
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
criterion = nn.MSELoss().to(device)

train_dataset = datasets.ImageFolder(traindir, train_transform)
train_loader = torch.utils.data.DataLoader(
    train_dataset,
    batch_size=2,
    shuffle=True,
    num_workers=args.num_workers,
    pin_memory=True,
    drop_last=True,
)

# int8_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
int4_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
not_quantized_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)

restore_indices = [8, 19]

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


seed_images, seed_labels = not_quantized_attack_net.get_seed_inputs(2, rand=False)


# int8_model = calibrate_model(args.mode, args, int8_model, train_loader, device)
int4_model = calibrate_model(args.mode, args, int4_model, train_loader, device)


# int8_model.eval()
int4_model.eval()
not_quantized_model.eval()

print()

# # TODO: Compute the hessian metrics
    
from pyhessian import DDV_hessian
from pyhessian import hessian
# # TODO:
# #####################################################
print("Calculating the sensitiveties via the averaged Hessian trace.......")
batch_num = 10
trace_list = []
for i, (inputs, labels) in enumerate(train_loader):
    adv_inputs = not_quantized_attack_net.gen_adv_inputs(inputs, labels)
    hessian_comp = DDV_hessian(
                    quantized_model=int4_model, 
                    original_model=not_quantized_model,
                    criterion=nn.CrossEntropyLoss().to(device),
                    bit_config=four_bit_config,
                    data=(inputs, labels),
                    adv_data = (adv_inputs, labels),
                    cuda=args.device)
    print("현재 몇번쨰?", i)
    # name, trace = hessian_comp.trace()
    # trace_list.append(trace)
    for grad in hessian_comp.gradsH:
        print(grad)
    
    if i == batch_num - 1:
        break

# top_eigenvalues, _ = hessian_comp.eigenvalues()
# trace = hessian_comp.trace()
# density_eigen, density_weight = hessian_comp.density()
# print('\n***Top Eigenvalues: ', top_eigenvalues)

# new_global_hessian_track = []
# for i in range(int(len(trace_list))):
#     hessian_track = trace_list[i]
#     hessian_track = [abs(x) for x in hessian_track]
#     min_h = min(hessian_track)
#     max_h = max(hessian_track)
#     averaged_hessian_track = [(elem-min_h)/(max_h-min_h) for elem in hessian_track]
#     new_global_hessian_track.append(averaged_hessian_track)


# # min_hessian = []
# # max_hessian = []
# layer_num = len(trace_list[0])
# for i in range(layer_num):
#     new_hessian = [sample[i] for sample in new_global_hessian_track]
#     mean_hessian.append(sum(new_hessian)/len(new_hessian))
#     # min_hessian.append(min(new_hessian))
#     # max_hessian.append(max(new_hessian))

# print(name)
# print('\n***Trace: ', mean_hessian)
    # # exit()
    # ################ deit-base ################
    # mean_hessian = [0.1728846995274323, 0.5223890107224295, 0.8191925959786669, 0.7076886016952384, 0.024708840222082775, 0.06145297177505395, 0.13322631271040494, 0.06554926888319061, 0.06175339225459908, 0.030678026107910893, 0.24494822213016829, 0.06636346426025085, 0.15758525560166742, 0.04395577998269693, 0.14552961945368617, 0.060864547749392026, 0.08752683209414383, 0.05799105819299426, 0.22538750132546922, 0.06785646981946868, 0.07478358821405745, 0.036487501147269154, 0.07572471890381866, 0.04584776940321937, 0.0906965395135412, 0.052852272764886334, 0.07057863784461312, 0.054111013841287636, 0.10702172109786383, 0.06730713583013927, 0.15666245711129553, 0.062172999291384645, 0.14509012240011504, 0.091604835756826, 0.2623722516111311, 0.06393236780883862, 0.11330756525833534, 0.0961950553973105, 0.18536753690007585, 0.09250514367800573, 0.11291326692010435, 0.09088161815323087, 0.08509066277645735, 0.19602731888893016, 0.05031627704809997, 0.06092669320490903, 0.23648108326696252, 0.07698688576427923, 0.37813159586619466]
    # ################ deit-tiny ################
    # mean_hessian = [0.12777249535991195, 0.3047042506776798, 0.6836076810672933, 0.9160977695613777, 0.051443724472863196, 0.1917038465654385, 0.40636168841774706, 0.31831214126540874, 0.17167878599488856, 0.17040465195968652, 0.5848568924580573, 0.34105575377627256, 0.2250203702397191, 0.24419067521700116, 0.5773478063329939, 0.33414308463155074, 0.25956759388373196, 0.1395379949578424, 0.4314355169808728, 0.22188267697321334, 0.1817366766340382, 0.11851699436886039, 0.4161464737579431, 0.19327061829322395, 0.17012293934278208, 0.12277515606872576, 0.4558816353483174, 0.15589752294249398, 0.17898296918815426, 0.086547094124963, 0.3467772011352197, 0.08775692025611888, 0.15284702235308084, 0.10833365447369167, 0.25759808027283065, 0.08692103455348514, 0.10185882004871938, 0.06342371816526218, 0.0780091910106661, 0.03666006418635352, 0.11141181591383327, 0.035333162826754756, 0.09242800375426533, 0.06258579742709644, 0.16515551045287732, 0.017525156872452197, 0.13652986573803982, 0.12360630901916989, 0.5199713391368654]
    #################  vit-base     #####################
    # mean_hessian = [0.2548212292719357, 0.652774443906641, 0.4679151921750381, 0.701685889252979, 0.285828470166026, 0.23157499632195172, 0.3476872482482762, 0.1357167839246311, 0.15553461818570039, 0.08420512187074286, 0.13815553335403274, 0.05567239346066725, 0.05587586852723446, 0.026548078787158015, 0.040773535370026856, 0.04080585779417317, 0.042558716664201815, 0.01815925147754802, 0.0479197088365278, 0.0471326762460345, 0.04083466214898966, 0.028311625792593897, 0.05059781160702729, 0.05021307087351986, 0.053499192355708956, 0.03629001097533719, 0.05553639666005887, 0.05542365527998931, 0.07634114724354114, 0.04736352053579504, 0.048323007545345804, 0.050717087287928765, 0.04673213199666633, 0.0502429101251724, 0.06749587123992873, 0.06645178277549102, 0.06218872019962326, 0.05860797496787497, 0.08825207961909944, 0.059215633889038034, 0.05765285649664825, 0.050049860737162055, 0.11113519269279008, 0.04891473081609033, 0.06074138350325581, 0.048028355020529635, 0.03297529568771655, 0.039936908641505384, 0.4446260183369337]
