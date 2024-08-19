import argparse
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.quantization import quantize_dynamic, quantize_jit, get_default_qconfig, prepare
from torch.utils.data import DataLoader
import numpy as np

from config import Config
from models import *  # 원본 코드에서 사용된 모델 import

parser = argparse.ArgumentParser(description='PyTorch Quantization and Model Difference Analysis')
parser.add_argument('model', choices=['deit_tiny', 'deit_small', 'deit_base', 'vit_base', 'vit_large', 'swin_tiny', 'swin_small', 'swin_base'], help='model')
parser.add_argument('data', metavar='DIR', help='path to dataset')
parser.add_argument('--calib-iter', default=10, type=int, help='number of calibration iterations')
parser.add_argument('--val-batchsize', default=50, type=int, help='batchsize of validation set')
parser.add_argument('--num-workers', default=16, type=int, help='number of data loading workers (default: 16)')
parser.add_argument('--device', default='cuda', type=str, help='device')

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

def build_transform(input_size=224, interpolation='bicubic', mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), crop_pct=0.875):
    t = []
    t.append(transforms.Resize(int(input_size / crop_pct), interpolation=transforms.InterpolationMode.BICUBIC))
    t.append(transforms.CenterCrop(input_size))
    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)

def hook_fn(name, model_outputs):
    def hook(module, input, output):
        model_outputs[name] = output
    return hook

def add_hooks(model, model_outputs):
    # Input quantization
    model.qact_input.register_forward_hook(hook_fn("qact_input", model_outputs))
    
    # Patch Embedding
    model.patch_embed.register_forward_hook(hook_fn("patch_embed", model_outputs))
    model.patch_embed.qact.register_forward_hook(hook_fn("patch_embed_qact", model_outputs))
    
    # Position Embedding
    model.pos_drop.register_forward_hook(hook_fn("pos_drop", model_outputs))
    model.qact_embed.register_forward_hook(hook_fn("qact_embed", model_outputs))
    model.qact_pos.register_forward_hook(hook_fn("qact_pos", model_outputs))
    
    # Transformer Blocks
    for i, block in enumerate(model.blocks):
        block.norm1.register_forward_hook(hook_fn(f"block_{i}_norm1", model_outputs))
        block.attn.qkv.register_forward_hook(hook_fn(f"block_{i}_attn_qkv", model_outputs))
        block.attn.proj.register_forward_hook(hook_fn(f"block_{i}_attn_proj", model_outputs))
        block.attn.qact3.register_forward_hook(hook_fn(f"block_{i}_attn_qact3", model_outputs))
        block.qact2.register_forward_hook(hook_fn(f"block_{i}_qact2", model_outputs))
        block.norm2.register_forward_hook(hook_fn(f"block_{i}_norm2", model_outputs))
        block.mlp.fc1.register_forward_hook(hook_fn(f"block_{i}_mlp_fc1", model_outputs))
        block.mlp.fc2.register_forward_hook(hook_fn(f"block_{i}_mlp_fc2", model_outputs))
        block.mlp.qact2.register_forward_hook(hook_fn(f"block_{i}_mlp_qact2", model_outputs))
        block.qact4.register_forward_hook(hook_fn(f"block_{i}_qact4", model_outputs))
    
    # Final Norm Layer
    model.norm.register_forward_hook(hook_fn("final_norm", model_outputs))
    model.qact2.register_forward_hook(hook_fn("final_qact2", model_outputs))
    
    # Classifier Head
    model.head.register_forward_hook(hook_fn("head", model_outputs))
    model.act_out.register_forward_hook(hook_fn("act_out", model_outputs))

def compute_ddv(model, normal_inputs, adv_inputs, outputs):
    def forward_and_get_outputs(inputs):
        model_output = model(inputs)
        if isinstance(model_output, tuple):
            model_output = model_output[0]
        return {k: v.clone() for k, v in outputs.items()}
    
    normal_outputs = forward_and_get_outputs(normal_inputs)
    adv_outputs = forward_and_get_outputs(adv_inputs)
    
    # print(normal_outputs.keys())
    # print(adv_outputs.keys())
    model_ddv_dict = {}
    for key in normal_outputs.keys():
        normal_layer_output = normal_outputs[key]
        adv_layer_output = adv_outputs[key]
        
        ddv = []
        for ya, yb in zip(normal_layer_output, adv_layer_output):
            ya = ya.detach().cpu().numpy().flatten()
            yb = yb.detach().cpu().numpy().flatten()
            ya = ya / np.linalg.norm(ya)
            yb = yb / np.linalg.norm(yb)
            cos_similarity = np.dot(ya, yb)
            ddv.append(cos_similarity)
        
        ddv = np.array(ddv)
        norm = np.linalg.norm(ddv)
        if norm != 0:
            ddv = ddv / norm
        model_ddv_dict[key] = ddv
    
    return model_ddv_dict

def calculate_and_print_similarities(source_ddv, target_ddv):
    for key in source_ddv.keys():
        source_layer = source_ddv[key]
        target_layer = target_ddv[key]
        
        similarities = []
        for ya, yb in zip(source_layer, target_layer):
            ya = ya / np.linalg.norm(ya)
            yb = yb / np.linalg.norm(yb)
            cos_similarity = np.dot(ya, yb) * 100
            similarities.append(cos_similarity)
        
        avg_similarity = np.mean(similarities)
        print(f"{key} layer similarity: {avg_similarity:.2f}%")

def get_seed_inputs(args, n=50):
    model_type = args.model.split('_')[0]
    if model_type == 'deit':
        mean, std, crop_pct = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.875
    elif model_type == 'vit':
        mean, std, crop_pct = (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), 0.9
    elif model_type == 'swin':
        mean, std, crop_pct = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 0.9
    else:
        raise NotImplementedError

    train_transform = build_transform(mean=mean, std=std, crop_pct=crop_pct)
    traindir = os.path.join(args.data, 'train')
    train_dataset = datasets.ImageFolder(traindir, train_transform)
    train_loader = DataLoader(train_dataset, batch_size=n, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    
    images, labels = next(iter(train_loader))
    return images.to(args.device), labels.to(args.device)

class AttackPGD(nn.Module):
    def __init__(self, basic_net, epsilon, step_size, num_steps):
        super(AttackPGD, self).__init__()
        self.basic_net = basic_net
        self.step_size = step_size
        self.epsilon = epsilon
        self.num_steps = num_steps

    def forward(self, inputs, targets):
        x = inputs.clone().detach() + torch.zeros_like(inputs).uniform_(-self.epsilon, self.epsilon)
        
        def myloss(yhat, y):
            return -((yhat[:,0]-y[:,0])**2 + 0.1*((yhat[:,1:]-y[:,1:])**2).mean(1)).mean()
        for i in range(self.num_steps):
            x.requires_grad_()
            with torch.enable_grad():
                output = self.basic_net(x)
                if isinstance(output, tuple):
                    output = output[0] 
                # loss = F.cross_entropy(output, targets)
                loss = myloss(output, targets)
            grad = torch.autograd.grad(loss, [x])[0]
            x = x.detach() + self.step_size * torch.sign(grad.detach())
            x = torch.min(torch.max(x, inputs - self.epsilon), inputs + self.epsilon)
            x = torch.clamp(x, 0, 1)
        
        return x

def gen_adv_inputs(model, inputs, labels):
    model.eval()
    clean_output = model(inputs)
    
    if isinstance(clean_output, tuple):
        clean_output = clean_output[0]
    
    attack_net = AttackPGD(model, epsilon=0.3, step_size=0.01, num_steps=50)
    
    output_mean = clean_output.mean(dim=0)
    target_outputs = output_mean - clean_output
    
    y = target_outputs * 1000 
    adv_inputs = attack_net(inputs, y)
    return adv_inputs.detach()

def calculate_accuracy(model, data_loader, device, bit_config=None):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs, _, _ = model(inputs)
            if isinstance(outputs, tuple):
                outputs = outputs[0]
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = 100 * correct / total
    return accuracy

def calibrate_model(mode, args, model, train_loader, device):
    if mode == 0:  # Real data mode
        # Get calibration set.
        image_list = []
        for i, (data, target) in enumerate(train_loader):
            if i == args.calib_iter:
                break
            data = data.to(device)
            image_list.append(data)

        print("Calibrating with real data...")
        model.model_open_calibrate()
        with torch.no_grad():
            model.model_open_last_calibrate()
            output, FLOPs, global_distance = model(image_list[0], plot=False)

    model.model_close_calibrate()
    model.model_quant()
    return model

def main():
    args = parser.parse_args()
    device = torch.device(args.device)

    # 모델 로드
    cfg = Config(False, False, 'minmax')  # ptf와 lis를 False로 설정
    
    original_model = str2model(args.model)(pretrained=True, cfg=cfg)
    original_model = original_model.to(device)

    quantization_model = str2model(args.model)(pretrained=True, cfg=cfg)
    quantization_model = quantization_model.to(device)
    
    # 데이터 로더 생성
    traindir = os.path.join(args.data, 'train')
    valdir = os.path.join(args.data, 'val')
    
    train_transform = build_transform()
    val_transform = build_transform()

    train_dataset = datasets.ImageFolder(traindir, train_transform)
    val_dataset = datasets.ImageFolder(valdir, val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=10,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=10,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # Calibrate and quantize the model
    quantized_model = calibrate_model(0, args, quantization_model, train_loader, device)

    # 정확도 계산
    original_accuracy = calculate_accuracy(original_model, val_loader, device)
    quantized_accuracy = calculate_accuracy(quantized_model, val_loader, device)

    print(f"Original model accuracy: {original_accuracy:.2f}%")
    print(f"Quantized model accuracy: {quantized_accuracy:.2f}%")
    
    # 훅 추가
    original_outputs = {}
    quantized_outputs = {}
    add_hooks(original_model, original_outputs)
    add_hooks(quantized_model, quantized_outputs)

    # Seed 입력 및 적대적 예제 생성
    seed_images, seed_labels = get_seed_inputs(args)
    adv_inputs = gen_adv_inputs(original_model, seed_images, seed_labels)

    # DDV 계산
    original_ddv = compute_ddv(original_model, seed_images, adv_inputs, original_outputs)
    quantized_ddv = compute_ddv(quantized_model, seed_images, adv_inputs, quantized_outputs)

    # 유사도 계산 및 출력
    print("Similarities between original and quantized model:")
    calculate_and_print_similarities(original_ddv, quantized_ddv)

    for key in quantized_ddv.keys():
        print(quantized_ddv[key] - original_ddv[key])
if __name__ == '__main__':
    main()