import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import math
import torchvision.transforms as transforms
import os
import torchvision.datasets as datasets
from scipy import spatial

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


def get_dataset(n, args, input_shape = (3, 224, 224)):
    
    
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

class AttackPGD(nn.Module):
    def __init__(self, basic_net, epsilon, step_size, num_steps, bit_config, args):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.bit_config = bit_config
        self.args = args

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

    def get_seed_inputs(self, n, rand=False, input_shape = (3, 224, 224)):
        if rand:
            batch_input_size = (n, input_shape[0], input_shape[1], input_shape[2])
            images = np.random.normal(size = batch_input_size).astype(np.float32)
        else:
            model_type = self.args.model.split('_')[0]
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
            traindir = os.path.join(self.args.data, 'train')

            train_dataset = datasets.ImageFolder(traindir, train_transform)
            train_loader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=n,
                shuffle=True,
                num_workers=self.args.num_workers,
                pin_memory=True,
                drop_last=True,
            )
            
            images, labels = next(iter(train_loader))
        return images.cuda(), labels.cuda()
        

        

    def gen_adv_inputs(self, inputs, labels, device='cuda'):
        """
        model에 대해 입력을 받아서 adversarial example을 생성하는 함수입니다.
        """
        self.basic_net.eval()
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
        
        adv_outputs, adv_inputs = self.forward(inputs, labels)
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
        current_adv_inputs = whitebox_attack_net.gen_adv_inputs(model1, inputs, labels, model1_bit_config)
        
        current_score = evaluate_inputs(current_adv_inputs)
        
        
        if current_score > score:
            print(f'current score update: {score}->{current_score}')
            max_inputs = current_adv_inputs
            score = current_score
            
    
    return max_inputs