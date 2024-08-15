import argparse
import math
import os
import time
import random
import torch.nn.functional as F

import torch
import torch.nn as nn
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from PIL import Image

from config import Config
from models import *
from generate_data import generate_data
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
parser.add_argument('--mixed', default=False, action='store_true')
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
                    default=50,
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


def build_transform(input_size=224,
                    interpolation='bicubic',
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                    crop_pct=0.875):

    def _pil_interp(method):
        if method == 'bicubic':
            return Image.BICUBIC
        elif method == 'lanczos':
            return Image.LANCZOS
        elif method == 'hamming':
            return Image.HAMMING
        else:
            return Image.BILINEAR

    resize_im = input_size > 32
    t = []
    if resize_im:
        size = int(math.floor(input_size / crop_pct))
        ip = _pil_interp(interpolation)
        t.append(
            transforms.Resize(
                size,
                interpolation=ip),  # to maintain same ratio w.r.t. 224 images
        )
        t.append(transforms.CenterCrop(input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)
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

args = parser.parse_args(args=[])
seed(args.seed)

device = torch.device(args.device)
cfg = Config(args.ptf, args.lis, args.quant_method)
model = str2model(args.model)(pretrained=True, cfg=cfg)
model = model.to(device)

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
model.eval()

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

if args.quant:
        # TODO:
        # Get calibration set
        # Case 0: PASQ-ViT
        if args.mode == 2:
            print("Generating data...")
            calibrate_data = generate_data(args)
            print("Calibrating with generated data...")
            model.model_open_calibrate()
            with torch.no_grad():
                model.model_open_last_calibrate()
                output = model(calibrate_data)
        # Case 1: Gaussian noise
        elif args.mode == 1:
            calibrate_data = torch.randn((args.calib_batchsize, 3, 224, 224)).to(device)
            print("Calibrating with Gaussian noise...")
            model.model_open_calibrate()
            with torch.no_grad():
                model.model_open_last_calibrate()
                output = model(calibrate_data)
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


#사용자 정의 손실 함수.
def myloss(yhat, y):
    # 첫 번째 클래스와 나머지 클래스 사이의 차이를 최대화하는 손실 함수
    # 이는 adversarial 예제의 다양성을 증가시키는 데 도움이 됩니다.
    return -((yhat[:,0]-y[:,0])**2 + 0.1*((yhat[:,1:]-y[:,1:])**2).mean(1)).mean()

class AttackPGD(nn.Module):
    def __init__(self, basic_net, epsilon, step_size, num_steps, bit_config):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.bit_config = bit_config

    def forward(self, inputs, targets):
        
        self.basic_net.zero_grad()
        outputs, _, _ = self.basic_net(inputs.to(device), bit_config = self.bit_config, plot = False, hessian_statistic=True)
        criterion = nn.CrossEntropyLoss().to(device)
        loss = criterion(outputs, targets.to(device))
        loss.backward(create_graph=True)
        grad = inputs.grad.clone()
        print(grad)
        # x = inputs.clone().detach()
        # x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        # for i in range(self.num_steps):
        #     x = x.clone().detach().requires_grad_(True)
        #     # with torch.enable_grad():
        #     outputs, Flops, distance = self.basic_net(x, self.bit_config, False)
        #     loss = F.cross_entropy(outputs, targets, reduction='sum')
        #         # loss = myloss(outputs, targets)
        #     loss.backward()
        #     # grad = torch.autograd.grad(loss, [x], create_graph=False)[0]
        #     grad = x.grad.clone()
        #     x = x + self.step_size*torch.sign(grad)
        #     x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
        #     # x = torch.clamp(x, inputs.min().item(), inputs.max().item())
        #     x = torch.clamp(x, 0, 1)
            
        #     with torch.no_grad():
        #         self.basic_net.eval()
        #         adv_output, Flops, distance= self.basic_net(x, self.bit_config, False)
            
        # return adv_output, x
    
bit_config = [4]*50
attack_net = AttackPGD(model, epsilon=0.1, step_size=0.01, num_steps=50, bit_config=bit_config)

def get_seed_inputs(n, rand=False, input_shape = (3, 224, 224)):
    if rand:
        batch_input_size = (n, input_shape[0], input_shape[1], input_shape[2])
        images = np.random.normal(size = batch_input_size).astype(np.float32)
    else:
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

        # Data
        traindir = os.path.join(args.data, 'train')

        train_dataset = datasets.ImageFolder(traindir, train_transform)
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=n,
            shuffle=True,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=True,
        )
        
        images, labels = next(iter(train_loader))
    return images, labels
        
        
        

def gen_adv_inputs(model, inputs, labels):
    model.eval()
    inputs = inputs.to(device)
    labels = labels.to(device)
    bit_config = [8]*50
    # with torch.no_grad():
    #     clean_output, FLOPs, distance = model(inputs, bit_config, plot=False)
    # output_shape = clean_output.shape
    # batch_size = output_shape[0]
    # num_classes = output_shape[1]
    
    
    # output_mean = clean_output.mean(axis = 0)
    # target_outputs = output_mean - clean_output
    
    # y = target_outputs * 1000 
    
    # adv_outputs, adv_inputs = attack_net(inputs, y)
    # torch.cuda.empty_cache()
    
    
    model.zero_grad()
    outputs = model(inputs.to(device), hessian_statistic=True)
    criterion = nn.CrossEntropyLoss().to(device)
    loss = criterion(outputs[0], labels)
    loss.backward(create_graph=True)
    grad = inputs.grad.clone()
    print(grad)
    
    return adv_inputs.to('cpu').numpy()
    

def compute_ddv(model, normal_inputs, adv_inputs):
    model.eval()
    
    with torch.no_grad():
        
        normal_outputs, _, _ = model(normal_inputs.to(device))
        adv_outputs, _, _ = model(adv_inputs.to(device))
        
    normal_outputs = normal_outputs.cpu().numpy()
    adv_outputs = adv_outputs.cpu().numpy()
            
    output_pairs = zip(normal_outputs, adv_outputs)
    
    ddv = []
    for i, (ya, yb) in enumerate(output_pairs):
        #ya와 yb의 cosiene similarity를 계산
        # dist = spatial.distance.cosine(ya, yb) -> 대체
        ya = ya / np.linalg.norm(ya)
        yb = yb / np.linalg.norm(yb)
        cos_similarity = np.dot(ya, yb)
        ddv.append(cos_similarity)
    ddv = np.array(ddv)
    norm = np.linalg.norm(ddv)
    if norm != 0:
        ddv = ddv/ norm
    return ddv


if __name__ == '__main__':
    torch.autograd.set_detect_anomaly(True)
    normal_inputs, labels = get_seed_inputs(10)
    adv_inputs = gen_adv_inputs(model, normal_inputs, labels)
    ddv = compute_ddv(model, normal_inputs, adv_inputs)
    print(ddv)