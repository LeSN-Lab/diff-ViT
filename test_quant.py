import argparse
import math
import os
import time
import random

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

parser.add_argument('--model',default = 'deit_tiny',)
parser.add_argument('--data',  default='/home/ubuntu/imagenet')
parser.add_argument('--quant', default=False, action='store_true')
parser.add_argument('--ptf', default=True)
parser.add_argument('--lis', default=True)
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
parser.add_argument('--calib-iter', default=6, type=int)
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


def main():
    args = parser.parse_args()
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
        batch_size=args.calib_batchsize,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    # # TODO: Compute the hessian metrics
    if args.mixed:
        from pyhessian import hessian
        
        # # TODO:
        # #####################################################
        print("Calculating the sensitiveties via the averaged Hessian trace.......")
        batch_num = 10
        trace_list = []
        for i, (inputs, labels) in enumerate(train_loader):
            hessian_comp = hessian(model,
                            criterion,
                            data=(inputs, labels),
                            cuda=args.device)
            print("현재 몇번쨰?", i)
            name, trace = hessian_comp.trace()
            trace_list.append(trace)
            if i == batch_num - 1:
                break
    
        # top_eigenvalues, _ = hessian_comp.eigenvalues()
        # trace = hessian_comp.trace()
        # density_eigen, density_weight = hessian_comp.density()
        # print('\n***Top Eigenvalues: ', top_eigenvalues)

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
        for i in range(layer_num):
            new_hessian = [sample[i] for sample in new_global_hessian_track]
            mean_hessian.append(sum(new_hessian)/len(new_hessian))
            # min_hessian.append(min(new_hessian))
            # max_hessian.append(max(new_hessian))

        print(name)
        print('\n***Trace: ', mean_hessian)
    # # exit()
    # ################ deit-base ################
    # mean_hessian = [0.1728846995274323, 0.5223890107224295, 0.8191925959786669, 0.7076886016952384, 0.024708840222082775, 0.06145297177505395, 0.13322631271040494, 0.06554926888319061, 0.06175339225459908, 0.030678026107910893, 0.24494822213016829, 0.06636346426025085, 0.15758525560166742, 0.04395577998269693, 0.14552961945368617, 0.060864547749392026, 0.08752683209414383, 0.05799105819299426, 0.22538750132546922, 0.06785646981946868, 0.07478358821405745, 0.036487501147269154, 0.07572471890381866, 0.04584776940321937, 0.0906965395135412, 0.052852272764886334, 0.07057863784461312, 0.054111013841287636, 0.10702172109786383, 0.06730713583013927, 0.15666245711129553, 0.062172999291384645, 0.14509012240011504, 0.091604835756826, 0.2623722516111311, 0.06393236780883862, 0.11330756525833534, 0.0961950553973105, 0.18536753690007585, 0.09250514367800573, 0.11291326692010435, 0.09088161815323087, 0.08509066277645735, 0.19602731888893016, 0.05031627704809997, 0.06092669320490903, 0.23648108326696252, 0.07698688576427923, 0.37813159586619466]
    # ################ deit-tiny ################
    # mean_hessian = [0.12777249535991195, 0.3047042506776798, 0.6836076810672933, 0.9160977695613777, 0.051443724472863196, 0.1917038465654385, 0.40636168841774706, 0.31831214126540874, 0.17167878599488856, 0.17040465195968652, 0.5848568924580573, 0.34105575377627256, 0.2250203702397191, 0.24419067521700116, 0.5773478063329939, 0.33414308463155074, 0.25956759388373196, 0.1395379949578424, 0.4314355169808728, 0.22188267697321334, 0.1817366766340382, 0.11851699436886039, 0.4161464737579431, 0.19327061829322395, 0.17012293934278208, 0.12277515606872576, 0.4558816353483174, 0.15589752294249398, 0.17898296918815426, 0.086547094124963, 0.3467772011352197, 0.08775692025611888, 0.15284702235308084, 0.10833365447369167, 0.25759808027283065, 0.08692103455348514, 0.10185882004871938, 0.06342371816526218, 0.0780091910106661, 0.03666006418635352, 0.11141181591383327, 0.035333162826754756, 0.09242800375426533, 0.06258579742709644, 0.16515551045287732, 0.017525156872452197, 0.13652986573803982, 0.12360630901916989, 0.5199713391368654]
    #################  vit-base     #####################
    # mean_hessian = [0.2548212292719357, 0.652774443906641, 0.4679151921750381, 0.701685889252979, 0.285828470166026, 0.23157499632195172, 0.3476872482482762, 0.1357167839246311, 0.15553461818570039, 0.08420512187074286, 0.13815553335403274, 0.05567239346066725, 0.05587586852723446, 0.026548078787158015, 0.040773535370026856, 0.04080585779417317, 0.042558716664201815, 0.01815925147754802, 0.0479197088365278, 0.0471326762460345, 0.04083466214898966, 0.028311625792593897, 0.05059781160702729, 0.05021307087351986, 0.053499192355708956, 0.03629001097533719, 0.05553639666005887, 0.05542365527998931, 0.07634114724354114, 0.04736352053579504, 0.048323007545345804, 0.050717087287928765, 0.04673213199666633, 0.0502429101251724, 0.06749587123992873, 0.06645178277549102, 0.06218872019962326, 0.05860797496787497, 0.08825207961909944, 0.059215633889038034, 0.05765285649664825, 0.050049860737162055, 0.11113519269279008, 0.04891473081609033, 0.06074138350325581, 0.048028355020529635, 0.03297529568771655, 0.039936908641505384, 0.4446260183369337]


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
        # exit()

    # FIXME:
    if args.mixed:
        # #####################################################
        print("Pareto Frontier.......")
        assert len(FLOPs)-1 == len(global_distance) == len(mean_hessian) 
        bit_list = []
        # model size constraint
        # TODO:
        # 모델 크기 제약 설정(논문의 Section III-D 에서 언급된 모델 크기 제약)
        #모든 레이어의 부동 소수점 연산에 대해 4bit의 연산을 한다고 했을때 1.1배까지를 제약조건으로 한다.
        model_constraint = 1.1*sum([FLOPs[i]*4 for i in range(len(FLOPs))])
        
        for i in range(2**len(global_distance)):
            # bit_config = [random.choice([torch.Tensor([4]).cuda(),torch.Tensor([8]).cuda()]) for i in range(len(FLOPs))]
            # TODO:
            bit_choice = [4,8] #4bit와 8bit 중 선택(논문의 Section III-D에서 언급된 비트 선택
            # bit_config = [random.choice(bit_choice) for i in range(len(FLOPs))]
            
            #전체 레이어 수의 절반에서 1을 뺸 만큼의 비트 설정을 랜덤하게 설정한다. 
            bit_config = [random.choice(bit_choice) for i in range(len(FLOPs)//2-1)]
            
            #첫번째 레이어에는 최대 비트수(8)을 할당한다. 위의 랜덤 비트 설정을 두배로 확장한 후 추가한다.
            new_bit_config = [max(bit_choice)] + [bit for bit in bit_config for i in range(2)] + [random.choice(bit_choice)]
            # new_bit_config = [7] + [bit for bit in bit_config for i in range(2)] + [6]
            #각 레이어의 FLOPs와 비트 설정을 곱한 후 모델 크기를 계산한다.
            model_size = sum([FLOPs[i]*new_bit_config[i] for i in range(len(FLOPs))])
            # FIXME:
            #model size가 제약조건을 넘지 않고, bit_list에 없는 경우 bit_list에 추가한다. -> 이러한 랜덤 bit 설정을 50개까지 생성한다.
            if not model_size > model_constraint and new_bit_config not in bit_list:
                bit_list.append(new_bit_config)
            if len(bit_list) > 50:
                break
        
        # compute the omega 
        #Hessian 기반 비용 계산(논문의 section III-D에서 언급된 Hessian 기반 민감도 측정.
        omega_list = []
        for bit_config in bit_list:
            select_diastance = []
            for i, bit in enumerate(bit_config):
                if i == 0:
                    continue
                for k, choice in enumerate(bit_choice):
                    if choice == bit:
                        #global distance에서 각 레이어의 거리를 구하는데 4bit와 8bit의 거리중 선택된 거리를 불러온다.
                        select_diastance.append(global_distance[i-1][k]) 
                        break
                # if bit == 4:
                #     select_diastance.append(global_distance[i][0])
                # elif bit == 6:
                #     select_diastance.append(global_distance[i][1])
                # elif bit == 8:
                #     select_diastance.append(global_distance[i][2])
                # elif bit == 10:
                #     select_diastance.append(global_distance[i][3])
                # else:
                #     assert bit == 4 or bit == 6 or bit == 8
            # TODO:
            # omega = [(mean_hessian[i]+sita_hessian[i])*select_diastance[i] for i in range(len(FLOPs))]
            #총 비용 omega를 계산한다. 
            omega = [mean_hessian[i]*select_diastance[i] for i in range(len(FLOPs)-1)]
            omega_list.append([bit_config, sum(omega)])
        
        # sort and selection
        omega_list.sort(key = lambda x : x[-1])
        #####################################################
        print('Hessien-Based Validating...')
        #상위 5개의 구성에 대해 검증을 수행한다.
        for i in range(5):
            # FIXME:
            bit_config = omega_list[i][0]
            # bit_config = [random.choice([5,6,7]) for i in range(len(FLOPs))]
            # bit_config = [6]*50
            # bit_config = [6, 7, 7, 7, 7, 6, 6, 7, 7, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 6, 7, 7, 7, 7, 5, 5, 6, 6, 5, 5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 6, 6, 5, 5, 5, 5, 5, 5, 6]
            # model_size = sum([FLOPs[i]*bit_config[i] for i in range(len(FLOPs))])
            # model_size = 0
            # FIXME:
            # if not model_size > model_constraint:
            print(bit_config)
            val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                criterion, device, bit_config)
            print('')
        
        # exit()

        # ####################### Evolutionary search ###################
        # 진화 알고리즘 초기화
        print('Start Evolutionary.......')
        parent_popu = []#부모 집단을 저장할 리스트.
        pop_size = 25   #집단 크기
        evo_iter = 8    #evolution iteration
        mutate_size = 10#돌연변이 연산으로 생성할 자식의 수
        mutate_prob = 0.5#각 비트가 돌연변이 될 확률
        crossover_size = 10 #교차 연산으로 생성할 자식 수
        crossover_prob = 0.5#교차 시 각 비트가 첫 번째 부모에서 올 확률
        
        #초기 부모 집단 생성
        for i in range(pop_size):
            bit_config = omega_list[i][0] #Hessian 기반 검색 결과에서 최상위 비트 구성을 가져온다.
            val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                criterion, device, bit_config) #해당 결과에 대해 검증을 수행한다.
            parent_popu.append([bit_config, val_prec1]) #해당 비트 구성과 정확도를 저장한다.
        parent_popu.sort(key = lambda x : x[-1], reverse=True) #정확도 기준으로 내림차순으로 정렬한다.
        
        
        #진화를 반복한다.
        for evo in range(evo_iter):                                      
            # 돌연변이(논문의 Section III-D에서 언급된 mutaitin 연산)
            children_list =[]
            mutate_bit_list =[]
            while True:
                old_bit = random.choice(parent_popu)[0]#현재 저장된 부모의 리스트에서 랜덤하게 선택한다.
                
                #old bi list에서 각 bit당 mutation 확률을 넘지 못하면 그대로 두고, 넘으면 랜덤하게 선택해서 변이를 한다.
                new_bit = [bit if random.random() < mutate_prob else random.choice(bit_choice) for bit in old_bit]
                
                #그렇게 비트구성을 변이시켰을 때 model constraint를 넘는가. 그리고 중복된 비트구성인가.
                model_size = sum([FLOPs[i]*new_bit[i] for i in range(len(FLOPs))]) 
                if not model_size > model_constraint and new_bit not in mutate_bit_list:
                    val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                criterion, device, new_bit)
                mutate_bit_list.append(new_bit)
                children_list.append([new_bit, val_prec1])   
                if len(mutate_bit_list) > mutate_size:
                    break
            
            # crossover
            crossover_bit_list =[]
            while True:
                old_bit_1 = random.choice(parent_popu)[0] #부모 집단에서 랜덤하게  첫번째 부모를 선택한다.
                old_bit_2 = random.choice(parent_popu)[0] #부모 집단에서 랜덤하게 두번째 부모를 선택한다.
                if old_bit_1 == old_bit_2: #만약 우연찮게 같은 부모를 선택했다면 다시 선택한다.
                    continue
                
                # 각 비트에 대해 crossover 확률을 넘으면 첫번째부모, 넘지 못하면 두번째 부모에서 비트를 선택한다.
                new_bit = [bit1 if random.random() < crossover_prob else bit2 for (bit1, bit2) in zip(old_bit_1, old_bit_2)]
                
                #그렇게 선택해서 비트를 구성했을 때 모델 크기를 넘지 않고, 중복된 비트 구성이 아닌 경우
                model_size = sum([FLOPs[i]*new_bit[i] for i in range(len(FLOPs))])
                if not model_size > model_constraint and new_bit not in crossover_bit_list:
                    val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                criterion, device, new_bit)
                crossover_bit_list.append(new_bit)
                children_list.append([new_bit, val_prec1])   
                if len(crossover_bit_list) > crossover_size:
                    break
            
            # updation
            for child in children_list:
                #자식들이 저장된 리스트에서 부모 리스트와 비교하여 부모의 최저 정확도보다 높은 경우 부모 리스트에 추가한다.
                if child[1] > parent_popu[-1][1]:
                    parent_popu.append(child)

            parent_popu.sort(key = lambda x : x[-1], reverse=True)
            parent_popu = parent_popu[:pop_size]
            print("Evolotionary iteration: ", evo)
            print(parent_popu)
            print('')
    else:
        # ###############################################################
        # TODO:
        bit_config = [4]*50
        print(bit_config)
        val_loss, val_prec1, val_prec5 = validate(args, val_loader, model,
                                                criterion, device, bit_config)


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


if __name__ == '__main__':
    main()
