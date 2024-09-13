# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import math
import warnings
from itertools import repeat

import torch
import torch.nn.functional as F
from torch import nn
from .plot_distrib import plot_distribution
from .ptq import QAct, QConv2d, QLinear

# alpha_pool = [0.35,0.4,0.5,0.55]
alpha_pool = [0.5]
bit_pool = [4,8]

def smoothquant_process(weight, act, ):
    def round_ln(x, type=None):
        if type == 'ceil':
            return torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
        elif type == 'floor':
            return torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
        else:
            y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            out = torch.gt((x-2**y),(2**(y+1)-x))
            return out+y
    c_out, c_in = weight.shape
    B, token, c_in = act.shape
    # channel-wise scaling factors
    local_max_x = torch.abs(act).max(axis=1).values
    global_max_x = local_max_x.max(axis=0).values
    max_weight = torch.abs(weight).max(axis=0).values
    channel_scale = (global_max_x**0.5)/(max_weight**0.5)
    aplha = round_ln(channel_scale, 'round')
    channel_scale = 2**aplha
    return channel_scale

def _ntuple(n):

    def parse(x):
        if isinstance(x, collections.abc.Iterable):
            return x
        return tuple(repeat(x, n))

    return parse


to_2tuple = _ntuple(2)


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # Cut & paste from PyTorch official master until it's in a few official releases - RW
    # Method based on https://people.sc.fsu.edu/~jburkardt/presentations/truncated_normal.pdf
    def norm_cdf(x):
        # Computes standard normal cumulative distribution function
        return (1.0 + math.erf(x / math.sqrt(2.0))) / 2.0

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn(
            'mean is more than 2 std from [a, b] in nn.init.trunc_normal_. '
            'The distribution of values may be incorrect.',
            stacklevel=2,
        )

    with torch.no_grad():
        # Values are generated by using a truncated uniform distribution and
        # then using the inverse CDF for the normal distribution.
        # Get upper and lower cdf values
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        # Uniformly fill tensor with values from [l, u], then translate to
        # [2l-1, 2u-1].
        tensor.uniform_(2 * l - 1, 2 * u - 1)

        # Use inverse cdf transform for normal distribution to get truncated
        # standard normal
        tensor.erfinv_()

        # Transform to proper mean, std
        tensor.mul_(std * math.sqrt(2.0))
        tensor.add_(mean)

        # Clamp to ensure it's in the proper range
        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0.0, std=1.0, a=-2.0, b=2.0):
    # type: (Tensor, float, float, float, float) -> Tensor
    r"""Fills the input Tensor with values drawn from a truncated
    normal distribution. The values are effectively drawn from the
    normal distribution :math:`\mathcal{N}(\text{mean}, \text{std}^2)`
    with values outside :math:`[a, b]` redrawn until they are within
    the bounds. The method used for generating the random values works
    best when :math:`a \leq \text{mean} \leq b`.
    Args:
        tensor: an n-dimensional `torch.Tensor`
        mean: the mean of the normal distribution
        std: the standard deviation of the normal distribution
        a: the minimum cutoff value
        b: the maximum cutoff value
    Examples:
        >>> w = torch.empty(3, 5)
        >>> nn.init.trunc_normal_(w)
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


def drop_path(x, drop_prob: float = 0.0, training: bool = False):
    """Drop paths (Stochastic Depth) per sample (when applied in main path of residual blocks).
    This is the same as the DropConnect impl I created for EfficientNet, etc networks, however,
    the original name is misleading as 'Drop Connect' is a different form of dropout in a separate paper...
    See discussion: https://github.com/tensorflow/tpu/issues/494#issuecomment-532968956 ... I've opted for
    changing the layer and argument names to 'drop path' rather than mix DropConnect as a layer name and use
    'survival rate' as the argument.
    """
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0], ) + (1, ) * (
        x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + \
        torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample  (when applied in main path of residual blocks)."""

    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class Mlp(nn.Module):

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        # self.fc1 = nn.Linear(in_features, hidden_features)
        self.qact0 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.fc1 = QLinear(in_features,
                           hidden_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.act = act_layer()
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        # self.fc2 = nn.Linear(hidden_features, out_features)
        self.fc2 = QLinear(hidden_features,
                           out_features,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        # self.qact2 = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A,
        #                   observer_str=cfg.OBSERVER_A,
        #                   quantizer_str=cfg.QUANTIZER_A)
        self.drop = nn.Dropout(drop)
        self.channel_scale = None
        self.fc1_output = None

    def forward(self, x, FLOPs, global_distance, ffn_bit_config, plot=False, quant=True, smoothquant=True, activation=[], hessian_statistic=False):
        # x = self.fc1(x)
        # x[0] = self.act(x[0])
        # x[1] = self.act(x[1])
        # x = self.qact1(x)
        # x[0] = self.drop(x[0])
        # x[1] = self.drop(x[1])
        # x = self.fc2(x)
        # x = self.qact2(x)
        # x[0] = self.drop(x[0])
        # x[1] = self.drop(x[1])
        B, N, C = x.shape
        activation.append(x)
        if ffn_bit_config:
            bit_config = ffn_bit_config[0]
        else:
            bit_config = None
        
        # FIXME: smoothquant
        if smoothquant and not hessian_statistic:
            if self.channel_scale == None or bit_config == -1:
                def round_ln(x, type=None):
                    if type == 'ceil':
                        return torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                    elif type == 'floor':
                        return torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                    else:
                        y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                        out = torch.gt((x-2**y),(2**(y+1)-x))
                        return out+y
                c_out, c_in = self.fc1.weight.shape
                B, token, c_in = x.shape
                # channel-wise scaling factors
                local_max_x = torch.abs(x).max(axis=1).values
                global_max_x = local_max_x.max(axis=0).values
                max_weight = torch.abs(self.fc1.weight).max(axis=0).values
                #         self.channel_scale = (global_max_x**0.5)/(max_weight**0.5)
                #         aplha = round_ln(self.channel_scale, 'round')
                #         self.channel_scale = 2**aplha
                #     x_smoothed = x/self.channel_scale.reshape((1,1,-1))
                #     weight_smoothed = self.fc1.weight*self.channel_scale.reshape((1,-1))
                # else:
                #     x_smoothed = x
                #     weight_smoothed = None
                # x = self.qact0(x_smoothed)
                # activation.append(x)
                # x = self.fc1(x, global_distance, bit_config, weight_smoothed)
                channel_scale_pool = []

                # gt = F.linear(x, self.qkv.weight, self.qkv.bias)
                loss_pool = [[],[]]
                act_scale = []
                act_zp = []
                weight_scale = []
                weight_zp = []
                if self.channel_scale == None:
                    self.best_scale = []
                    self.best_act_scale = []
                    self.best_act_zp = []
                    self.best_weight_scale = []
                    self.best_weight_zp = []
                for i, alpha in enumerate(alpha_pool):
                    channel_scale = global_max_x**alpha/(max_weight**(1-alpha))
                    aplha = round_ln(channel_scale, 'round')
                    channel_scale = 2**aplha
                    channel_scale_pool.append(channel_scale)
                    x_smoothed = x/channel_scale.reshape((1,1,-1))
                    weight_smoothed = self.fc1.weight*channel_scale.reshape((1,-1))
                    gt = F.linear(x_smoothed, weight_smoothed, self.fc1.bias)
                    
                    # observe to obtaion scaling factors
                    middle_out = self.qact0(x_smoothed)
                    if self.qact0.last_calibrate and bit_config != -1:
                        act_scale.append(self.qact0.quantizer.scale)
                        act_zp.append(self.qact0.quantizer.zero_point)
                        middle_out = self.fc1(middle_out, global_distance, bit_config, weight_smoothed)
                        weight_scale.append(self.fc1.quantizer.dic_scale)
                        weight_zp.append(self.fc1.quantizer.dic_zero_point)
                        # compute loss
                        self.qact0.calibrate = False
                        self.qact0.quant = True
                        middle_out = self.qact0(x_smoothed)
                        self.fc1.calibrate = False
                        self.fc1.quant = True
                        for j, bit in enumerate(bit_pool):
                            quant_out = self.fc1(middle_out, global_distance, bit, weight_smoothed)
                            loss_pool[j].append((gt - quant_out).abs().pow(2.0).mean())
                        self.qact0.quant = False
                        self.qact0.calibrate = True
                        self.fc1.quant = False
                        self.fc1.calibrate = True
                if self.qact0.last_calibrate and bit_config != -1:
                    for loss in loss_pool:
                        indx = loss.index(min(loss))
                        self.channel_scale = channel_scale_pool[indx]
                        self.best_scale.append(channel_scale_pool[indx])
                        self.best_act_scale.append(act_scale[indx])
                        self.best_act_zp.append(act_zp[indx])
                        self.best_weight_scale.append(weight_scale[indx])
                        self.best_weight_zp.append(weight_zp[indx])            
                    activation.append(x_smoothed)
                x = gt
            else:
                indx = bit_pool.index(bit_config)
                self.channel_scale = self.best_scale[indx]    
                x_smoothed = x/self.channel_scale.reshape((1,1,-1))
                weight_smoothed = self.fc1.weight*self.channel_scale.reshape((1,-1))

                self.qact0.quantizer.scale = self.best_act_scale[indx] 
                self.qact0.quantizer.zero_point = self.best_act_zp[indx] 
                x = self.qact0(x_smoothed)
                activation.append(x)
                self.fc1.quantizer.dic_scale = self.best_weight_scale[indx] 
                self.fc1.quantizer.dic_zero_point = self.best_weight_zp[indx] 
                x = self.fc1(x, global_distance, bit_config, weight_smoothed)
                
        else:
            x_smoothed = x
            weight_smoothed = None
            x = self.qact0(x_smoothed)
            activation.append(x)
            x = self.fc1(x, global_distance, bit_config, weight_smoothed)

        # activation.append(x_smoothed)
        self.fc1_output = x.detach().clone()  # qkv 출력 저장

        B, N, M = x.shape
        FLOPs.append(N*C*M)
        
        x = self.act(x)
        # TODO:
        x = self.qact1(x, asymmetric=False)
        # activation.append(x)
        x = self.drop(x)
        
        B, N, C = x.shape
        if ffn_bit_config:
            bit_config = ffn_bit_config[1]
        else:
            bit_config = None
        x = self.fc2(x, global_distance, bit_config)
        B, N, M = x.shape
        FLOPs.append(N*C*M)
        
        x = self.qact2(x)
        activation.append(x)
        if plot:
            plot_distribution(activation, 'MLP', quant)
        x = self.drop(x)
        return x


class PatchEmbed(nn.Module):
    """Image to Patch Embedding"""

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 embed_dim=768,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        # 이미지 크기와 패치 크기를 2D 튜플로 반환
        #to_2tuple 함수는 _ntuple(2)를 호출하여 생성됩니다.
        #이 함수는 입력이 이터러블(iterable)이 아닌 경우, 해당 값을 2번 반복하여 튜플로 만듭니다.
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        self.img_size = img_size
        self.patch_size = patch_size

        self.grid_size = (img_size[0] // patch_size[0],#224x224 크기의 이미지를 16x16크기의
                          img_size[1] // patch_size[1]) #패치로 나누면 14x14개의 패치 그리드가 생성됨.
        self.num_patches = self.grid_size[0] * self.grid_size[1] # 14* 14 =196 => 총 패치의 갯수.
        #N, 3 ,224, 224가 들어가서 (16, 16) weight, (16, 16) stride가 적용되었다고 가정하면
        #N, 768, 14, 14가 나올 것이라고 예상이 가능함.
        self.proj = QConv2d(in_chans,
                            embed_dim,
                            kernel_size=patch_size,
                            stride=patch_size,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)
        self.patch_size = patch_size
        if norm_layer:
            self.qact_before_norm = QAct(
                quant=quant,
                calibrate=calibrate,
                bit_type=cfg.BIT_TYPE_A,
                calibration_mode=cfg.CALIBRATION_MODE_A,
                observer_str=cfg.OBSERVER_A,
                quantizer_str=cfg.QUANTIZER_A)
            self.norm = norm_layer(embed_dim)
            self.qact = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        else:
            self.qact_before_norm = nn.Identity()
            self.norm = nn.Identity()
            self.qact = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)

    def forward(self, x, FLOPs, bit_config):
        # B, C, H, W = x[0].shape
        # # FIXME look at relaxing size constraints
        # assert (
        #     H == self.img_size[0] and W == self.img_size[1]
        # ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        # # x = self.proj(x).flatten(2).transpose(1, 2)
        # x = self.proj(x)
        # x[0] = x[0].flatten(2).transpose(1, 2)
        # x[1] = x[1].flatten(2).transpose(1, 2)
        # x[0] = self.qact_before_norm(x[0])
        # x[1] = self.qact_before_norm(x[1])
        # if isinstance(self.norm, nn.Identity):
        #     x[0] = self.norm(x[0])
        #     x[1] = self.norm(x[1])
        # else:
        #     x = self.norm(x, self.qact_before_norm.quantizer,
        #                   self.qact.quantizer)
        # x = self.qact(x)
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert (
            H == self.img_size[0] and W == self.img_size[1]
        ), f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        #2D 합성곱 적용 후, flatten 및 transpose
        """
        flatten(2)의 의미:

        flatten(start_dim=2)는 텐서의 차원 중 인덱스 2부터 끝까지의 모든 차원을 하나의 차원으로 평탄화(flatten)합니다.
        여기서 2는 시작 차원 인덱스를 나타냅니다. (파이썬에서 인덱스는 0부터 시작)


        연산 전후의 텐서 형태 변화:

        연산 전 (Conv2D 출력): (B, C, H, W)

        B: 배치 크기
        C: 출력 채널 수 (embed_dim)
        H: 높이
        W: 너비


        flatten(2) 후: (B, C, H*W)

        높이와 너비 차원이 하나로 합쳐집니다.


        transpose(1, 2) 후: (B, H*W, C)

        채널 차원과 평탄화된 공간 차원의 위치가 바뀝니다.
        """
        x = self.proj(x, bit_config) #proj를 위해 conv2d에서 bit_finetune를 받아옴.
        B, M, H, W = x.shape
        """
        FLOPs.append(C*self.patch_size[0]*self.patch_size[1]*M*H*W)
        이 라인은 합성곱 연산(self.proj)의 FLOPs를 계산하여 리스트에 추가합니다.

        C: 입력 채널 수
        self.patch_size[0], self.patch_size[1]: 패치의 높이와 너비
        M: 출력 채널 수 (embed_dim)
        H, W: 출력 특징 맵의 높이와 너비


        이 계산은 2D 합성곱 연산의 FLOPs를 나타냅니다. 각 출력 픽셀마다 입력 채널 * 커널 크기 * 출력 채널 만큼의 곱셈-덧셈 연산이 필요합니다.
        FLOPs 리스트에 이 값을 추가함으로써, 모델의 다른 부분에서 전체 연산량을 집계할 수 있게 됩니다.
        """
        FLOPs.append(C*self.patch_size[0]*self.patch_size[0]*M*H*W) #FLOPs 계산
        x = x.flatten(2).transpose(1, 2)
        
        x = self.qact_before_norm(x)
        if isinstance(self.norm, nn.Identity):
            x = self.norm(x)
        else:
            x = self.norm(x, self.qact_before_norm.quantizer,
                          self.qact.quantizer)
        x = self.qact(x)
        return x


class HybridEmbed(nn.Module):
    """CNN Feature Map Embedding
    Extract feature map from CNN, flatten, project to embedding dim.
    """

    def __init__(self,
                 backbone,
                 img_size=224,
                 feature_size=None,
                 in_chans=3,
                 embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                # FIXME this is hacky, but most reliable way of determining the exact dim of the output feature
                # map for all networks, the feature metadata has reliable channel and stride info, but using
                # stride to calc feature dim requires info about padding of each stage that isn't captured.
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(
                    torch.zeros(1, in_chans, img_size[0], img_size[1]))
                if isinstance(o, (list, tuple)):
                    # last feature if backbone outputs list/tuple of features
                    o = o[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            if hasattr(self.backbone, 'feature_info'):
                feature_dim = self.backbone.feature_info.channels()[-1]
            else:
                feature_dim = self.backbone.num_features
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Conv2d(feature_dim, embed_dim, 1)

    def forward(self, x):
        x = self.backbone(x)
        if isinstance(x, (list, tuple)):
            x = x[
                -1]  # last feature if backbone outputs list/tuple of features
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x
