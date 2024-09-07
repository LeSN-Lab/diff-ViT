import argparse
import math
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
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
from scipy import spatial
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
parser.add_argument('--ptf', default=False)
parser.add_argument('--lis', default=False)
parser.add_argument('--quant-method',
                    default='minmax',
                    choices=['minmax', 'ema', 'omse', 'percentile'])
parser.add_argument('--mixed', default=False, action='store_true')
# TODO: 100 --> 32
parser.add_argument('--calib-batchsize',
                    default=5,
                    type=int,
                    help='batchsize of calibration set')
parser.add_argument("--mode", default=0,
                        type=int, 
                        help="mode of calibration data, 0: PSAQ-ViT, 1: Gaussian noise, 2: Real data")
# TODO: 10 --> 1
parser.add_argument('--calib-iter', default=10, type=int)
# TODO: 100 --> 200
parser.add_argument('--val-batchsize',
                    default=5,
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

def myloss(yhat, y):
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
        x = inputs.clone().detach()
        x = x + torch.zeros_like(x).uniform_(-self.epsilon, self.epsilon)
        for i in range(self.num_steps):
            x = x.clone().detach().requires_grad_(True)
            # with torch.enable_grad():
            outputs, Flops, distance = self.basic_net(x, self.bit_config, False)
            loss = F.cross_entropy(outputs, targets, reduction='sum')
            # loss = myloss(outputs, targets)
            loss.backward()
            # grad = torch.autograd.grad(loss, [x], create_graph=False)[0]
            grad = x.grad.clone()
            x = x + self.step_size*torch.sign(grad)
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, inputs.min().item(), inputs.max().item())
            # x = torch.clamp(x, 0, 1)
            
            with torch.no_grad():
                self.basic_net.eval()
                adv_output, Flops, distance= self.basic_net(x, self.bit_config, False)
            
        return adv_output, x
    
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
    return images.cuda(), labels.cuda()
        
def get_dataset(n, input_shape = (3, 224, 224)):
    
    
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
    
    
    return train_loader
        

def gen_adv_inputs(model, inputs, labels, attack_net):
    """
    model에 대해 입력을 받아서 adversarial example을 생성하는 함수입니다.
    """
    model.eval()
    inputs = inputs.to(device)
    labels = labels.to(device)
    # bit_config = [8]*50
    # with torch.no_grad():
    #     clean_output, FLOPs, distance = model(inputs, bit_config, plot=False)
    # output_shape = clean_output.shape
    # batch_size = output_shape[0]
    # num_classes = output_shape[1]
    
    """
    다양성 최대화:
    ModelDiff 논문의 4.2절에서는 생성된 입력의 다양성(diversity)을 강조합니다.
    target_outputs = output_mean - clean_output를 사용함으로써, 각 입력이 평균 출력과 다르게 되도록 유도하고 있습니다.
    이는 생성된 입력들이 서로 다른 특성을 가지도록 하는 데 도움이 됩니다.
    """
    # output_mean = clean_output.mean(axis = 0)
    # target_outputs = output_mean - clean_output
    """
    결정 경계 탐색:
    y = target_outputs * 1000에서 큰 스케일 팩터(1000)를 사용하는 것은,
    모델의 결정 경계를 더 잘 탐색하기 위한 것으로 보입니다.
    이는 논문의 Figure 3에서 설명하는 "decision boundary" 개념과 연관됩니다.
    """
    # y = target_outputs * 1000 
    
    adv_outputs, adv_inputs = attack_net(inputs, labels)
    # adv_outputs, adv_inputs = attack_net(inputs, y)
    torch.cuda.empty_cache()
    return adv_inputs.detach()

def metrics_output_diversity(model, bit_config, inputs, use_torch=False):
    # 논문의 4.2절에서 설명한 출력 다양성 메트릭 계산
    outputs = model(inputs, bit_config, False)[0].detach().to('cpu').numpy()
#         output_dists = []
#         for i in range(0, len(outputs) - 1):
#             for j in range(i + 1, len(outputs)):
#                 output_dist = spatial.distance.euclidean(outputs[i], outputs[j])
#                 output_dists.append(output_dist)
#         diversity = sum(output_dists) / len(output_dists)
    # cdist 함수는 두 집합 모든 쌍 사이의 거리를 유클리드 거리를 이용해서 계산함.
    #outputs_dists는 모든 출력 쌍 사이의 거리를 담은 행렬.
    output_dists = spatial.distance.cdist(list(outputs), list(outputs), metric='euclidean')
    #계산된 모든 거리의 평균을 구함.
    diversity = np.mean(output_dists)
    return diversity

def gen_profiling_inputs_in_blackbox(model1, model1_bit_config, model2, model2_bit_config, seed_inputs, use_torch=False, epsilon=0.2):
    #논문의 4.2절에서 설명한 테스트 입력 생성 알고리즘 구현
    input_shape = seed_inputs[0].shape
    n_inputs = seed_inputs.shape[0]
    max_iterations = 1000 #최대 반복 횟수 설정
    # max_steps = 10 
    
    
    ndims = np.prod(input_shape) #입력의 총 차원 수 계산
#         mutate_positions = torch.randperm(ndims)

        # Move seed_inputs to GPU if not already
    if not seed_inputs.is_cuda:
        seed_inputs = seed_inputs.cuda()
        
    # 초기 모델의 출력 계산
    with torch.no_grad():
        initial_outputs1 = model1(seed_inputs, bit_config = model1_bit_config, plot=False)[0].detach().to('cpu').numpy()
        initial_outputs2 = model2(seed_inputs, bit_config = model2_bit_config, plot=False)[0].detach().to('cpu').numpy()
    
    def evaluate_inputs(inputs):
        #논문의 Equantion 1에서 설명한 score 함수 구현
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).cuda()
        elif inputs.device.type != 'cuda':
            inputs = inputs.cuda()
        with torch.no_grad():
            outputs1 = model1(inputs, bit_config = model1_bit_config, plot=False)[0].detach().to('cpu').numpy()
            outputs2 = model2(inputs, bit_config = model2_bit_config, plot=False)[0].detach().to('cpu').numpy()
        
        metrics1 = metrics_output_diversity(model1, model1_bit_config, inputs) #diversity 계산
        metrics2 = metrics_output_diversity(model2, model2_bit_config, inputs) # diversity 계산


        # divergence 계산 (초기 출력과의 거리)
        output_dist1 = np.mean(spatial.distance.cdist(
            list(outputs1),
            list(initial_outputs1),
            metric='euclidean').diagonal())
        output_dist2 = np.mean(spatial.distance.cdist(
            list(outputs2),
            list(initial_outputs2),
            metric='euclidean').diagonal())
        print(f'  output distance: {output_dist1},{output_dist2}')
        print(f'  metrics: {metrics1},{metrics2}')
        # if mutated_metrics <= metrics:
        #     break
        
        #score 계산 : divergence와 diversity의 곱
        return output_dist1 * output_dist2 * metrics1 * metrics2
    
    inputs = seed_inputs
    score = evaluate_inputs(inputs)
    print(f'score={score}')
    
    for i in range(max_iterations):
        #Alogrithm 1: Search-based input generation 
        # comparator._compute_distance(inputs)
        print(f'mutation {i}-th iteration')
        # mutation_idx = random.randint(0, len(inputs))
        # mutation = np.random.random_sample(size=input_shape).astype(np.float32)
        
        #무작위 위치 선택하여 mutation 생성
        mutation_pos = np.random.randint(0, ndims)
        mutation = np.zeros(ndims).astype(np.float32)
        mutation[mutation_pos] = epsilon
        mutation = np.reshape(mutation, input_shape) #이 코드는 1차원 mutation 벡터를 원래 입력 데이터의 shape으로 재구성합니다.
        
        
        
        mutation_batch = torch.zeros_like(inputs)
        mutation_idx = np.random.randint(0, n_inputs)
        mutation_batch[mutation_idx] = torch.from_numpy(mutation).cuda()
        
        # print(f'{inputs.shape} {mutation_perturbation.shape}')
        # for j in range(max_steps):
            # mutated_inputs = np.clip(inputs + mutation, 0, 1)
            # print(f'{list(inputs)[0].shape}')
        mutate_right_inputs = inputs + mutation_batch
        mutate_right_score = evaluate_inputs(mutate_right_inputs)
        mutate_left_inputs = inputs - mutation_batch
        mutate_left_score = evaluate_inputs(mutate_left_inputs)
        
        if mutate_right_score <= score and mutate_left_score <= score:
            continue
        if mutate_right_score > mutate_left_score:
            print(f'mutate right: {score}->{mutate_right_score}')
            inputs = mutate_right_inputs
            score = mutate_right_score
        else:
            print(f'mutate left: {score}->{mutate_left_score}')
            inputs = mutate_left_inputs
            score = mutate_left_score
    return inputs


def gen_profiling_inputs_in_whitebox(model1, model1_bit_config, model2, model2_bit_config, seed_inputs, seed_labels, use_torch=False, epsilon=0.2, whitebox_attack_net=None):
    #논문의 4.2절에서 설명한 테스트 입력 생성 알고리즘 구현
    input_shape = seed_inputs[0].shape
    n_inputs = seed_inputs.shape[0]
    max_iterations = 20 #최대 반복 횟수 설정
    # max_steps = 10 
    
    
    ndims = np.prod(input_shape) #입력의 총 차원 수 계산
#         mutate_positions = torch.randperm(ndims)

        # Move seed_inputs to GPU if not already
    if not seed_inputs.is_cuda:
        seed_inputs = seed_inputs.cuda()
        
    # 초기 모델의 출력 계산
    with torch.no_grad():
        initial_outputs1 = model1(seed_inputs, bit_config = model1_bit_config, plot=False)[0].detach().to('cpu').numpy()
        initial_outputs2 = model2(seed_inputs, bit_config = model2_bit_config, plot=False)[0].detach().to('cpu').numpy()
    
    def evaluate_inputs(inputs):
        #논문의 Equantion 1에서 설명한 score 함수 구현
        if isinstance(inputs, np.ndarray):
            inputs = torch.from_numpy(inputs).cuda()     
        elif inputs.device.type != 'cuda':
            inputs = inputs.cuda()
        with torch.no_grad():
            outputs1 = model1(inputs, bit_config = model1_bit_config, plot=False)[0].detach().to('cpu').numpy()
            outputs2 = model2(inputs, bit_config = model2_bit_config, plot=False)[0].detach().to('cpu').numpy()
        
        metrics1 = metrics_output_diversity(model1, model1_bit_config, inputs) #diversity 계산
        metrics2 = metrics_output_diversity(model2, model2_bit_config, inputs) # diversity 계산


        # divergence 계산 (초기 출력과의 거리)
        output_dist1 = np.mean(spatial.distance.cdist(
            list(outputs1),
            list(initial_outputs1),
            metric='euclidean').diagonal())
        output_dist2 = np.mean(spatial.distance.cdist(
            list(outputs2),
            list(initial_outputs2),
            metric='euclidean').diagonal())
        print(f'  output distance: {output_dist1},{output_dist2}')
        print(f'  metrics: {metrics1},{metrics2}')
        # if mutated_metrics <= metrics:
        #     break
        
        #score 계산 : divergence와 diversity의 곱
        return output_dist1 * output_dist2 * metrics1 * metrics2
    
    inputs = seed_inputs
    max_inputs = None
    labels = seed_labels
    score = evaluate_inputs(inputs)
    print(f'score={score}')
    
    for i in range(max_iterations):
        #Alogrithm 1: Search-based input generation 
        # comparator._compute_distance(inputs)
        print(f'mutation {i}-th iteration')
        # mutation_idx = random.randint(0, len(inputs))
        # mutation = np.random.random_sample(size=input_shape).astype(np.float32)
        
        #무작위 위치 선택하여 mutation 생성
        current_adv_inputs = gen_adv_inputs(model1, inputs, labels, model1_bit_config, attack_net=whitebox_attack_net)
        
        current_score = evaluate_inputs(current_adv_inputs)
        
        
        if current_score > score:
            print(f'current score update: {score}->{current_score}')
            max_inputs = current_adv_inputs
            score = current_score
            
    
    return max_inputs


int8_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
int4_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)
not_quantized_model = model_make(args.model, args.ptf, args.lis, args.quant_method, args.device)

eight_bit_config = [8]*50
not_quantized_attack_net = AttackPGD(not_quantized_model, epsilon=0.06, step_size=0.01, num_steps=50, bit_config=None)
four_bit_config = [4]*50
seed_images, seed_labels = get_seed_inputs(50, rand=False)
adv_inputs = gen_adv_inputs(not_quantized_model, seed_images, seed_labels, attack_net=not_quantized_attack_net)
# mutation_inputs = gen_profiling_inputs_in_blackbox(not_quantized_model, None,  int4_model, four_bit_config, seed_images, epsilon=0.02)


int8_model = calibrate_model(args.mode, args, int8_model, train_loader, device)
int4_model = calibrate_model(args.mode, args, int4_model, train_loader, device)


int8_model.eval()
int4_model.eval()
not_quantized_model.eval()

print()

def normalize_activations(act):
    # 입력 텐서를 2D로 재구성합니다. 첫 번째 차원은 유지하고 나머지는 평탄화합니다.
    act = act.view(act.size(0), -1)

    # 각 샘플(행)에 대해 L2 norm을 계산합니다.
    act_norm = torch.norm(act, p=2, dim=1, keepdim=True)

    # 0으로 나누는 것을 방지하기 위해 작은 값을 더합니다.
    act_norm = act_norm + 1e-8

    # 각 샘플을 해당 norm으로 나누어 정규화합니다.
    act = act / act_norm

    return act
#torch model의 layers의 수를 확인한다.
from efficient_CKA import *

def get_activations(images, model, bit_config, normalize_act=False):
    model = model.to(device)

    



    def get_module_path(module):
        return f"{module.__class__.__module__}.{module.__class__.__name__}"

    activations = []
    layer_info = []
    from models.vit_fquant import Attention, Mlp
    def hook_return(index):
        def hook(module, input, output):
            if isinstance(module, Attention):
                activations.append(module.qkv_output)
                layer_info.append({
                'relative_index': len(activations) - 1,
                'absolute_index': index,
                'name': module.__class__.__name__,
                'layer_type': type(module),
                'path': get_module_path(module)

                })
            elif isinstance(module, Mlp):
                activations.append(module.fc1_output)
                layer_info.append({
                'relative_index': len(activations) - 1,
                'absolute_index': index,
                'name': module.__class__.__name__,
                'layer_type': type(module),
                'path': get_module_path(module)

                })
            else:    
                activations.append(output)
                layer_info.append({
                    'relative_index': len(activations) - 1,
                    'absolute_index': index,
                    'name': module.__class__.__name__,
                    'layer_type': type(module),
                    'path': get_module_path(module)

                })
            

        return hook

    hooks = []


    for index, layer in enumerate(model.modules()):
        if type(layer) in [QConv2d, QLinear, Attention, Mlp]:
            hooks.append(layer.register_forward_hook(hook_return(index)))

    # 모델을 통해 이미지를 전달합니다.
    images = images.cuda()
    _ = model(images, bit_config = bit_config, plot=False)

    # 등록된 후크를 제거합니다.
    for h in hooks:
        h.remove()





    # layer_info와 activations를 절대 인덱스를 기준으로 정렬
    sorted_indices = sorted(range(len(layer_info)), key=lambda k: layer_info[k]['absolute_index'])
    layer_info = [layer_info[i] for i in sorted_indices]
    activations = [activations[i] for i in sorted_indices]

    # 상대 인덱스 재할당
    for i, info in enumerate(layer_info):
        info['relative_index'] = i


    if normalize_act:
        activations = [normalize_activations(act) for act in activations]
    return activations
    # 정렬된 레이어 정보 출력
    # for info in layer_info:
    #     print(f"Layer {info['relative_index']}(absolute: {info['absolute_index']}): {info['name']} (Type: {info['layer_type']}, Path: {info['path']})")

    # print(f"\nTotal number of activations: {len(activations)}")
    
    # 필요한 라이브러리 임포트
import torch
import argparse
import os
import pickle
import numpy as np
from plot import *

def plot_cka_map(cka_file_name, plot_name):
    base_dir = '/home/jieungkim/quantctr/diff-ViT'



    # GPU 설정


    # CKA 결과 파일 경로 설정
    cka_dir = os.path.join(base_dir, cka_file_name)


    # CKA 결과 불러오기
    with open(cka_dir, 'rb') as f:
        cka = pickle.load(f)
        
        


    # 전체 레이어에 대한 CKA 결과 플롯 생성
    plot_dir = os.path.join(base_dir, plot_name)
    plot_ckalist_resume([cka], plot_dir)
# plot_cka_map('cka_not_quantized_result.pkl', 'cka_not_quantized_result.png')
from DDV_CKA import *
def compute_cka_with_adversarial(model1, model2, use_batch = True,
                         normalize_act = False,
                         cka_batch = 50,
                         cka_batch_iter = 10,
                         cka_iter = 10,
                         result_name = 'cka_result.pkl',
                         model1_bit_config = None,
                         model2_bit_config = None,
                         ):
    model1.eval()
    model2.eval()

    sample_cka_dataset = get_dataset(cka_batch)

    sample_cka_dataset = next(iter(sample_cka_dataset))

    sample_images, sample_labels = sample_cka_dataset
    cka_attack_net1 = AttackPGD(model1, epsilon=0.06, step_size=0.01, num_steps=50, bit_config = model1_bit_config)
    
    
    
    # cka_attack_net2 = AttackPGD(model2, epsilon=0.06, step_size=0.01, num_steps=50, bit_config = bit_config)
    #@To Do: cka_attack_net2를 직접 사용해보기
    cka_attack_net2 = cka_attack_net1 #모델1과 같은 공격 네트워크를 사용한다. 
    
    
    
    

    sample_activations = get_activations(sample_images, model1, model1_bit_config, normalize_act)
    n_layers = len(sample_activations)

    cka = MinibatchAdvCKA(n_layers)
    

    if use_batch:
        for index in range(cka_iter):
            #cka_batch만큼, shuffle해서, 데이터셋을 가져온다.
            cka_dataset = get_dataset(cka_batch)
            current_iter = 0
            for images, labels in cka_dataset:
                adv_images = gen_adv_inputs(model1, images, labels, cka_attack_net1)
                
                model1_get_activation = get_activations(images, model1, model1_bit_config, normalize_act) #각 모델의 레이어별 활성화를 가져온다.
                model1_get_adv_activation = get_activations(adv_images, model1, model1_bit_config, normalize_act)
                
                model2_get_activation = get_activations(images, model2, model2_bit_config, normalize_act)
                model2_get_adv_activation = get_activations(adv_images, model2, model2_bit_config, normalize_act)
                
                cka.update_state(model1_activations=model1_get_activation,
                                 model1_adv_activations=model1_get_adv_activation,
                                 model2_activations=model2_get_activation,
                                 model2_adv_activations=model2_get_adv_activation) #레이어 마다의 activation을 다 가져옴. 예를 들어 24 * 50 * feature^2. 
                
                if current_iter > cka_batch_iter:
                    break
                current_iter += 1
            print("현재 반복:", index)
    else:
        cka_dataset = get_dataset(cka_batch)
        all_images = []
        all_labels = []
        for images, labels in cka_dataset:
            all_images.append(images)
            all_labels.append(labels)
            all_adv_images = gen_adv_inputs(model1, all_images, all_labels, cka_attack_net1)
        cka.update_state(
            model1_activations=get_activations(all_images, model1,  model1_bit_config, normalize_act),
            model1_adv_activations=get_activations(all_adv_images, model1, model1_bit_config, normalize_act),
            model2_activations=get_activations(all_images, model2, model2_bit_config, normalize_act),
            model2_adv_activations=get_activations(all_adv_images, model2, model2_bit_config, normalize_act),
            )
    heatmap = cka.result().cpu().numpy()
    with open(result_name, 'wb') as f:
        #pickle로 heatmap을 저장한다.
        pickle.dump(heatmap, f)
        
        
compute_cka_with_adversarial(not_quantized_model,
                             int4_model, 
                             use_batch = True, 
                             normalize_act = False, 
                             cka_batch = 10, 
                             cka_iter = 20, 
                             result_name='cka_with_adversarial_int4_not_quantized.pkl', 
                             model1_bit_config = None,
                             model2_bit_config = four_bit_config)