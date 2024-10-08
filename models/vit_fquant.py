# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import collections.abc
import math
import os
import re
import warnings
import copy
from collections import OrderedDict
from functools import partial
from itertools import repeat
from .ptq.bit_type import BIT_TYPE_DICT, BIT_TYPE_LIST
import torch
import torch.nn.functional as F
from torch import nn
from .plot_distrib import plot_distribution
from .layers_quant import DropPath, HybridEmbed, Mlp, PatchEmbed, trunc_normal_
# from .layers import Mlp_woq, PatchEmbed_woq
from .ptq import QAct, QConv2d, QIntLayerNorm, QIntSoftmax, QLinear
from .utils import load_weights_from_npz
# from .ptq.bit_type import BIT_TYPE_DICT, BIT_TYPE_LIST

__all__ = [
    'deit_tiny_patch16_224', 'deit_small_patch16_224', 'deit_base_patch16_224',
    'vit_base_patch16_224', 'vit_large_patch16_224'
]

# alpha_pool = [0.35,0.4,0.5]
# deit_small_4: 0.35; deit_small_8: 0.55 (no)
# deit_base_4: 0.35; deit_base_8: 0.4
# deit_tiny: smoothquant-False; Res-True
# vit-base_4: 0.35/0.35; vit-base_8: 0.5
alpha_pool = [0.35] 
bit_pool = [4,8]


def smoothquant_process(weight, act, alpha):
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
    channel_scale = (global_max_x*alpha)/(max_weight**(1-alpha))
    aplha = round_ln(channel_scale, 'round')
    channel_scale = 2**aplha
    return channel_scale

class Attention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads=8,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.0,
                 proj_drop=0.0,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.num_heads = num_heads #멀티헤드 어텐션의 헤드 수 결정
        head_dim = dim // num_heads # 각 헤드의 차원 계산
        self.calibrate =calibrate
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim**-0.5 #스케일 팩터 설정.

        self.qkv = QLinear(dim,
                           dim * 3,
                           bias=qkv_bias,
                           quant=quant,
                           calibrate=calibrate,
                           bit_type=cfg.BIT_TYPE_W,
                           calibration_mode=cfg.CALIBRATION_MODE_W,
                           observer_str=cfg.OBSERVER_W,
                           quantizer_str=cfg.QUANTIZER_W) #Q, K, V 계산을 위한 양자화된 선형 곱 층
        # self.qacts = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A_LN,
        #                   observer_str=cfg.OBSERVER_A_LN,
        #                   quantizer_str=cfg.QUANTIZER_A_LN)  
        self.qact0 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)               
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)
        #출력 프로젝션을 위한 양자화된 선형 층
        self.proj = QLinear(dim,
                            dim,
                            quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_W,
                            calibration_mode=cfg.CALIBRATION_MODE_W,
                            observer_str=cfg.OBSERVER_W,
                            quantizer_str=cfg.QUANTIZER_W)
        self.qact3 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        # self.qact3 = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A,
        #                   observer_str=cfg.OBSERVER_A,
        #                   quantizer_str=cfg.QUANTIZER_A)
        
        #어텐션 스코어 양자화를 위한 활성화함수.
        self.qact_attn1 = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.attn_drop = nn.Dropout(attn_drop) #어텐션 드롭 아웃
        self.proj_drop = nn.Dropout(proj_drop) #프로젝션 드롭아웃
        self.log_int_softmax = QIntSoftmax(
            log_i_softmax=cfg.INT_SOFTMAX,
            quant=quant,
            calibrate=calibrate,
            bit_type=cfg.BIT_TYPE_S,
            calibration_mode=cfg.CALIBRATION_MODE_S,
            observer_str=cfg.OBSERVER_S,
            quantizer_str=cfg.QUANTIZER_S)
        self.channel_scale = None #채널 스케일링 팩터(Smooth Quant에서 사용)
        self.qkv_output = None

    def forward(self, x, FLOPs, global_distance, atten_bit_config, plot=False, quant=False, smoothquant=True, hessian_statistic=False):
        # B, N, C = x[0].shape
        # x = self.qkv(x)
        # x = self.qact1(x)
        # qkv = x[0].reshape(B, N, 3, self.num_heads,
        #                 C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        # qkv_q = x[1].reshape(B, N, 3, self.num_heads,
        #                 C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        # q, k, v = (
        #     qkv[0],
        #     qkv[1],
        #     qkv[2],
        # )  # make torchscript happy (cannot use tensor as tuple)
        # q_q, k_q, v_q = (
        #     qkv_q[0],
        #     qkv_q[1],
        #     qkv_q[2],
        # )  # make torchscript happy (cannot use tensor as tuple)
        # attn = (q @ k.transpose(-2, -1)) * self.scale
        # attn_q = (q_q @ k_q.transpose(-2, -1)) * self.scale
        # [attn, attn_q] = self.qact_attn1([attn, attn_q])
        # # TODO:
        # [attn, attn_q] = self.log_int_softmax([attn, attn_q], self.qact_attn1.quantizer.scale)
        # # attn = attn.softmax(dim=-1)
        # attn = self.attn_drop(attn)
        # attn_q = self.attn_drop(attn_q)
        # x[0] = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # x[1] = (attn_q @ v_q).transpose(1, 2).reshape(B, N, C)
        # x = self.qact2(x)
        # x = self.proj(x)
        # x = self.qact3(x)
        # x[0] = self.proj_drop(x[0])
        # x[1] = self.proj_drop(x[1])
        # return x
        self.atten_bit_config = atten_bit_config #어텐션 비트 설정 저장
        
        activation = [] #활성화 값 저장을 위한 리스트
        B, N, C = x.shape #입력 텐서의 shape추출
        if atten_bit_config:
            bit_config = atten_bit_config[0] #첫번쨰 어텐션 비트 설정 추출
        else:
            bit_config = None
        
        # FIXME: smoothquant
        # out = self.qacts(x)
        if smoothquant and not hessian_statistic:
            # SmoothQuant 로직 구현
            
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

                c_out, c_in = self.qkv.weight.shape
                B, token, c_in = x.shape
                # # scale_factor_v1 = smoothquant_process(self.qkv.weight, x, 0.5)
                local_max_x = torch.abs(x).max(axis=1).values
                global_max_x = local_max_x.max(axis=0).values
                max_weight = torch.abs(self.qkv.weight).max(axis=0).values
                # scale_factor_v1 = (global_max_x**0.5)/(max_weight**0.5)
                # aplha = round_ln(scale_factor_v1, 'round')
                # scale_factor_v1 = 2**aplha
                # x_smoothed_v1 = x/scale_factor_v1.reshape((1,1,-1))
                # weight_smoothed_v1 = self.qkv.weight*scale_factor_v1.reshape((1,-1))
        
                # channel-wise scaling factors
                # local_max_x = torch.abs(x).max(axis=1).values
                # global_max_x = local_max_x.max(axis=0).values
                # max_weight = torch.abs(self.qkv.weight).max(axis=0).values
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
                    weight_smoothed = self.qkv.weight*channel_scale.reshape((1,-1))
                    gt = F.linear(x_smoothed, weight_smoothed, self.qkv.bias)
                    
                    # observe to obtaion scaling factors
                    middle_out = self.qact0(x_smoothed)
                    if self.qact0.last_calibrate and bit_config != -1:
                        act_scale.append(self.qact0.quantizer.scale)
                        act_zp.append(self.qact0.quantizer.zero_point)
                        middle_out = self.qkv(middle_out, global_distance, bit_config, weight_smoothed, attn=False, attn_para=[self.num_heads, C, self.scale])
                        weight_scale.append(self.qkv.quantizer.dic_scale)
                        weight_zp.append(self.qkv.quantizer.dic_zero_point)
                        # compute loss
                        self.qact0.calibrate = False
                        self.qact0.quant = True
                        middle_out = self.qact0(x_smoothed)
                        self.qkv.calibrate = False
                        self.qkv.quant = True
                        for j, bit in enumerate(bit_pool):
                            quant_out = self.qkv(middle_out, global_distance, bit, weight_smoothed, attn=False, attn_para=[self.num_heads, C, self.scale])
                            loss_pool[j].append((gt - quant_out).abs().pow(2.0).mean())
                        self.qact0.quant = False
                        self.qact0.calibrate = True
                        self.qkv.quant = False
                        self.qkv.calibrate = True
                if self.qact0.last_calibrate and bit_config != -1:
                    for loss in loss_pool:
                        indx = loss.index(min(loss))
                        self.channel_scale = channel_scale_pool[indx]
                        self.best_scale.append(channel_scale_pool[indx])
                        self.best_act_scale.append(act_scale[indx])
                        self.best_act_zp.append(act_zp[indx])
                        self.best_weight_scale.append(weight_scale[indx])
                        self.best_weight_zp.append(weight_zp[indx])            

                x = gt
            else:
                indx = bit_pool.index(bit_config)
                self.channel_scale = self.best_scale[indx]    
                x_smoothed = x/self.channel_scale.reshape((1,1,-1))
                weight_smoothed = self.qkv.weight*self.channel_scale.reshape((1,-1))

                self.qact0.quantizer.scale = self.best_act_scale[indx] 
                self.qact0.quantizer.zero_point = self.best_act_zp[indx] 
                x = self.qact0(x_smoothed)
                activation.append(x)
                self.qkv.quantizer.dic_scale = self.best_weight_scale[indx] 
                self.qkv.quantizer.dic_zero_point = self.best_weight_zp[indx] 
                x = self.qkv(x, global_distance, bit_config, weight_smoothed, attn=False, attn_para=[self.num_heads, C, self.scale])
        else:
            x_smoothed = x
            weight_smoothed = None
            x = self.qact0(x_smoothed)
            activation.append(x)
            x = self.qkv(x, global_distance, bit_config, weight_smoothed, attn=False, attn_para=[self.num_heads, C, self.scale])
           
        self.qkv_output = x.detach().clone()  # qkv 출력 저장

        B, N, M = x.shape
        FLOPs.append(N*C*M) # FLOPs 계산 및 추가
        # TODO:
        #입력 양자화
        x = self.qact1(x, attn=False, attn_para=[self.num_heads, C, self.scale])
        activation.append(x) #양자화된 입력을 activation에 추가
        qkv = x.reshape(B, N, 3, self.num_heads, #qkv 텐서 재구성
                        C // self.num_heads).permute(2, 0, 3, 1, 4)  # (BN33)
        q, k, v = ( #q, k, v 텐서로 분리
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)
        attn = (q @ k.transpose(-2, -1)) * self.scale #어텐션 스케일링
        attn = self.qact_attn1(attn) #어텐션 스코어 양자화
        activation.append(x) #양자화된 어텐션 스코어를 activation에 추가
        # TODO:
        #log int softmax 적용.
        attn = self.log_int_softmax(attn, self.qact_attn1.quantizer.scale)
        # attn = attn.softmax(dim=-1)
        
        attn = self.attn_drop(attn) #어텐션 드롭아웃
        x = (attn @ v).transpose(1, 2).reshape(B, N, C) #어텐션 값 계산
        x = self.qact2(x) #어텐션 값 양자화
        activation.append(x) #양자화된 어텐션 값을 activation에 추가
        
        B, N, C = x.shape
        if atten_bit_config:
            bit_config = atten_bit_config[1]
        else:
            bit_config = None
        x = self.proj(x, global_distance, bit_config) #최종 프로젝션에서 양자화 비트를 사용한다.
        B, N, M = x.shape
        FLOPs.append(N*C*M) #FLOPs 계산 및 추가
        
        x = self.qact3(x) #프로젝션 값 양자화
        activation.append(x) #양자화된 프로젝션 값을 activation에 추가
        if plot:
            plot_distribution(activation, 'attn', quant) #활성화 분포 플로팅 (디버깅용)
        # exit()
        x = self.proj_drop(x) # 프로젝션 드롭아웃 적용
        return x  # 최종 어텐션 결과 반환

    def get_requant_scale(self):
        bit_config = self.atten_bit_config[1]
        bit_type = 'int' + str(bit_config)
        return (self.qact2.quantizer.scale*self.proj.quantizer.dic_scale[BIT_TYPE_DICT[bit_type].name])/self.qact3.quantizer.scale


class Block(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.0,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.0,
                 attn_drop=0.0,
                 drop_path=0.0,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 quant=False,
                 calibrate=False,
                 cfg=None):
        super().__init__()
        self.norm1 = norm_layer(dim)
        # self.qact1 = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A,
        #                   observer_str=cfg.OBSERVER_A,
        #                   quantizer_str=cfg.QUANTIZER_A)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              cfg=cfg)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0.0 else nn.Identity()
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        self.norm2 = norm_layer(dim)
        # self.qact3 = QAct(quant=quant,
        #                   calibrate=calibrate,
        #                   bit_type=cfg.BIT_TYPE_A,
        #                   calibration_mode=cfg.CALIBRATION_MODE_A,
        #                   observer_str=cfg.OBSERVER_A,
        #                   quantizer_str=cfg.QUANTIZER_A)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop,
                       quant=quant,
                       calibrate=calibrate,
                       cfg=cfg)
        self.qact4 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)

    def forward(self, x, last_quantizer=None, FLOPs=[], global_distance=[], local_bit_config=None, plot=False, quant=False, hessian_statistic=False):
        # x = self.qact2(x + self.drop_path(
        #     self.attn(
        #         self.qact1(self.norm1(x, last_quantizer,
        #                               self.qact1.quantizer)))))
        activation = []
        # activation.append(x)
        if local_bit_config:
            atten_bit_config = local_bit_config[0:2]
        else:
            atten_bit_config = None
        # x = self.qact2(x + self.drop_path(
        #     self.attn(
        #         self.qact1(self.norm1(x, last_quantizer,
        #                               self.qact1.quantizer)), FLOPs, global_distance, atten_bit_config, plot, quant)))
        if atten_bit_config is not None and -1 in atten_bit_config:
            self.norm1.mode = 'ln'
        x = self.qact2(x + self.drop_path(
            self.attn(
                (self.norm1(x, last_quantizer,
                                      self.attn.qact0.quantizer, self.attn.channel_scale)), FLOPs, global_distance, atten_bit_config, plot=False, quant=quant, hessian_statistic=hessian_statistic)))
        activation.append(x)
        # x_old = copy.deepcopy(x)
        # x = self.attn(
        #         self.qact1(self.norm1(x, last_quantizer,
        #                               self.qact1.quantizer)))
        # x[0] = self.drop_path(x[0])
        # x[1] = self.drop_path(x[1])
        # # x = self.qact2(x_old + x)
        # x[0] = x_old[0] + x[0]
        # x[1] = x_old[1] + x[1]
        # x = self.qact2(x)
        
        # x = self.qact4(x + self.drop_path(
        #     self.mlp(
        #         self.qact3(
        #             self.norm2(x, self.qact2.quantizer,
        #                        self.qact3.quantizer)))))
        if local_bit_config:
            ffn_bit_config = local_bit_config[2:4]
        else:
            ffn_bit_config = None
        # x = self.qact4(x + self.drop_path(
        #     self.mlp(
        #         self.qact3(
        #             self.norm2(x, self.qact2.quantizer,
        #                        self.qact3.quantizer)), FLOPs, global_distance, ffn_bit_config, plot, quant)))
        # ffn_bit_config중 하나라도 -1이 있다면
        if ffn_bit_config is not None and -1 in ffn_bit_config:
            self.norm2.mode = 'ln'
        norm2_result = self.norm2(x, self.qact2.quantizer, self.mlp.qact0.quantizer, self.attn.channel_scale)
        #int4_model: mlp_result가 정상적으로 나왔음.
        #int-1model: mlp_result가 다 nan으로 나옴. mlp에서 이상이 생긴 것 같음.
        mlp_result =self.mlp(norm2_result, FLOPs, global_distance, ffn_bit_config, plot, quant, activation=activation, hessian_statistic=hessian_statistic)
        x = self.qact4(x + self.drop_path(mlp_result))
        # activation.append(x)
        # if plot:
        #     plot_distribution(activation, 'block', quant)
        # exit()
        # x_old = copy.deepcopy(x)
        # x = self.mlp(
        #         self.qact3(
        #             self.norm2(x, self.qact2.quantizer,
        #                        self.qact3.quantizer)))
        # x[0] = self.drop_path(x[0])
        # x[1] = self.drop_path(x[1])
        # # x = self.qact4(x_old + x)
        # x[0] = x_old[0] + x[0]
        # x[1] = x_old[1] + x[1]
        # x = self.qact4(x)
        return x


class VisionTransformer(nn.Module):
    """Vision Transformer
    A PyTorch impl of : `An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale`  -
        https://arxiv.org/abs/2010.11929
    """

    def __init__(self,
                 img_size=224,
                 patch_size=16,
                 in_chans=3,
                 num_classes=1000,
                 embed_dim=768,
                 depth=12,
                 num_heads=12,
                 mlp_ratio=4.0,
                 qkv_bias=True,
                 qk_scale=None,
                 representation_size=None,
                 drop_rate=0.0,
                 attn_drop_rate=0.0,
                 drop_path_rate=0.0,
                 hybrid_backbone=None,
                 norm_layer=None,
                 quant=False,
                 calibrate=False,
                 input_quant=False,
                 cfg=None):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = (
            self.embed_dim
        ) = embed_dim  # num_features for consistency with other models
        norm_layer = norm_layer or partial(nn.LayerNorm, eps=1e-6)

        self.cfg = cfg
        self.quant = False
        self.input_quant = input_quant
        if input_quant:
            self.qact_input = QAct(quant=quant,
                                   calibrate=calibrate,
                                   bit_type=cfg.BIT_TYPE_A,
                                   calibration_mode=cfg.CALIBRATION_MODE_A,
                                   observer_str=cfg.OBSERVER_A,
                                   quantizer_str=cfg.QUANTIZER_A)

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone,
                img_size=img_size,
                in_chans=in_chans,
                embed_dim=embed_dim,
            )
        else:
            self.patch_embed = PatchEmbed(img_size=img_size,
                                          patch_size=patch_size,
                                          in_chans=in_chans,
                                          embed_dim=embed_dim,
                                          quant=quant,
                                          calibrate=calibrate,
                                          cfg=cfg)
            # self.patch_embed_woq = PatchEmbed_woq(img_size=img_size,
            #                               patch_size=patch_size,
            #                               in_chans=in_chans,
            #                               embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches
        """
        ViT는 이미지 분류를 위해 "클래스 토큰"이라는 특별한 토큰을 추가합니다.
        이 클래스 토큰은 패치 임베딩 시퀀스의 맨 앞에 추가됩니다.
        최종적으로 이 클래스 토큰의 출력이 분류에 사용됩니다.
        """
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        #위치 인코딩 설정
        self.pos_embed = nn.Parameter(
            torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        #임베딩과 위치 인코딩을 위한 양자화 액티베이션 레이어
        self.qact_embed = QAct(quant=quant,
                               calibrate=calibrate,
                               bit_type=cfg.BIT_TYPE_A,
                               calibration_mode=cfg.CALIBRATION_MODE_A,
                               observer_str=cfg.OBSERVER_A,
                               quantizer_str=cfg.QUANTIZER_A)
        self.qact_pos = QAct(quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_A,
                             calibration_mode=cfg.CALIBRATION_MODE_A,
                             observer_str=cfg.OBSERVER_A,
                             quantizer_str=cfg.QUANTIZER_A)
        self.qact1 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A_LN,
                          observer_str=cfg.OBSERVER_A_LN,
                          quantizer_str=cfg.QUANTIZER_A_LN)
        #transformer 블록 설정
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)
               ]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(dim=embed_dim,
                  num_heads=num_heads,
                  mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias,
                  qk_scale=qk_scale,
                  drop=drop_rate,
                  attn_drop=attn_drop_rate,
                  drop_path=dpr[i],
                  norm_layer=norm_layer,
                  quant=quant,
                  calibrate=calibrate,
                  cfg=cfg) for i in range(depth)
        ])
        self.depth =depth
        self.norm = norm_layer(embed_dim)
        self.qact2 = QAct(quant=quant,
                          calibrate=calibrate,
                          bit_type=cfg.BIT_TYPE_A,
                          calibration_mode=cfg.CALIBRATION_MODE_A,
                          observer_str=cfg.OBSERVER_A,
                          quantizer_str=cfg.QUANTIZER_A)

        # Representation layer
        if representation_size:
            self.num_features = representation_size
            self.pre_logits = nn.Sequential(
                OrderedDict([
                    ('fc', nn.Linear(embed_dim, representation_size)),
                    ('act', nn.Tanh()),
                ]))
        else:
            self.pre_logits = nn.Identity()

        # Classifier head
        self.head = (QLinear(self.num_features,
                             num_classes,
                             quant=quant,
                             calibrate=calibrate,
                             bit_type=cfg.BIT_TYPE_W,
                             calibration_mode=cfg.CALIBRATION_MODE_W,
                             observer_str=cfg.OBSERVER_W,
                             quantizer_str=cfg.QUANTIZER_W)
                     if num_classes > 0 else nn.Identity())
        self.act_out = QAct(quant=quant,
                            calibrate=calibrate,
                            bit_type=cfg.BIT_TYPE_A,
                            calibration_mode=cfg.CALIBRATION_MODE_A,
                            observer_str=cfg.OBSERVER_A,
                            quantizer_str=cfg.QUANTIZER_A)
        #가중치 초기화.
        """
        pos_embed는 위치 인코딩(positional encoding)을 위한 텐서입니다.
        평균 0, 표준편차 0.02의 절단된 정규 분포로 초기화됩니다.
        이는 위치 정보를 미세하게 인코딩하면서도 너무 큰 값을 가지지 않도록 합니다.
        """
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)
        

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = (nn.Linear(self.embed_dim, num_classes)
                     if num_classes > 0 else nn.Identity())

    def model_quant(self, flag='on'):
        if flag == 'on':
            self.quant = True
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = True
            if self.cfg.INT_NORM:
                if type(m) in [QIntLayerNorm]:
                    if flag == 'off':
                        pass
                    else:
                        m.mode = 'int'

    def model_dequant(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.quant = False

    def model_open_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = True

    def model_open_last_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.last_calibrate = True

    def model_close_calibrate(self):
        for m in self.modules():
            if type(m) in [QConv2d, QLinear, QAct, QIntSoftmax]:
                m.calibrate = False

    def forward_features(self, x, FLOPs, global_distance, bit_config, global_plot, hessian_statistic=False):
        # B = x[0].shape[0]
        B = x.shape[0] #배치 크기 추출
        activation = []
        #입력 양자화
        if self.input_quant:
            x = self.qact_input(x)
        activation.append(x) #입력을 양자화한 것을 activation 배열에 append.
            # print()
        # x_original = x
        if bit_config: #bit config가 설정되어있으면 patch_bit을 설정.
            patch_bit = bit_config[0]
        else:
            patch_bit = None
        #이미지를 패치로 분할하고 임베딩하는데 patch_bit와 FLOPs를 같이 줌.
        x = self.patch_embed(x, FLOPs, patch_bit) 
        # y = self.patch_embed_woq(x_original[0])
        # print(out[0]==y)
        cls_tokens = self.cls_token.expand( #배치크기만큼 cls_token을 확장
            B, -1, -1)  
        # x[0] = torch.cat((cls_tokens, x[0]), dim=1)
        # x[1] = torch.cat((cls_tokens, x[1]), dim=1)
        x = torch.cat((cls_tokens, x), dim=1) #cls_token과 패치 임베딩을 합침.
        x = self.qact_embed(x) #임베딩 결과를 양자화한다.
        activation.append(x) #양자화한 것을 저장.
        x = x + self.qact_pos(self.pos_embed) #position 정보도 같이 저장함.
        
        # out = self.qact_pos([self.pos_embed,self.pos_embed])
        # # print(out[0]==self.pos_embed)
        # x[0] = x[0] + out[0]
        # x[1] = x[1] + out[1]

        x = self.qact1(x)
        activation.append(x)

        # x[0] = self.pos_drop(x[0])
        # x[1] = self.pos_drop(x[1])
        x = self.pos_drop(x) #드롭아웃 적용.
        #각 블록을 순회하면서 forward 수행.
        for i, blk in enumerate(self.blocks):
            if bit_config:
                #각 블록당 4개의 quantizer configure이 필요함.
                """
                Attention의 Query, Key, Value 계산을 위한 비트 수
                Attention의 출력 프로젝션을 위한 비트 수
                FFN(Feed-Forward Network)의 첫 번째 선형 레이어를 위한 비트 수
                FFN의 두 번째 선형 레이어를 위한 비트 수
                """
                local_bit_config = bit_config[i*4+1:i*4+5]
            else:
                local_bit_config = None
            #이전 블록의 quantizer정보 전달.
            last_quantizer = self.qact1.quantizer if i == 0 else self.blocks[
                i - 1].qact4.quantizer
            if i == self.depth-1 and global_plot:
            # if i == 0 and global_plot:
                plot = True
            else:
                plot = False
            # print(i)
            x = blk(x, last_quantizer, FLOPs, global_distance, local_bit_config, plot, self.quant, hessian_statistic)
            # TODO:
            # if bit_config:
            #     scale = blk.attn.get_requant_scale()
            #     print(torch.floor(torch.div(torch.log(scale),torch.log(torch.Tensor([2]).cuda()))))
        
        x = self.norm(x, self.blocks[-1].qact4.quantizer,
                      self.qact2.quantizer)[:, 0]
        # x[0] = x[0][:, 0]
        # x[1] = x[1][:, 0]

        x = self.qact2(x)
        # activation.append(x)
        # plot_distribution(activation, 'vit', self.quant)
        # x[0] = self.pre_logits(x[0])
        # x[1] = self.pre_logits(x[1])
        x = self.pre_logits(x)
        # exit()
        return x

    def forward(self, x, bit_config=None, plot=False, hessian_statistic=False):
        # new_x = [x, x]
        # new_x = x
        FLOPs = []
        global_distance = []
        x = self.forward_features(x, FLOPs, global_distance, bit_config, plot, hessian_statistic)
        
        B, C = x.shape
        if bit_config:
            head_bit = bit_config[-1]
        else:
            head_bit = None
        x = self.head(x, global_distance, head_bit)
        B, M = x.shape
        FLOPs.append(C*M)

        x = self.act_out(x)
        # print(new_x[0]==y[0])
        # return new_x[1]
        return x, FLOPs, global_distance


def deit_tiny_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(
        patch_size=16,
        embed_dim=192,
        depth=12,
        num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(QIntLayerNorm, eps=1e-6),
        quant=quant,
        calibrate=calibrate,
        input_quant=True,
        cfg=cfg,
        **kwargs,
    )
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_tiny_patch16_224-a1311bcf.pth',
            map_location='cpu',
            check_hash=True,
        )
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_small_patch16_224(pretrained=False,
                           quant=False,
                           calibrate=False,
                           cfg=None,
                           **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=384,
                              depth=12,
                              num_heads=6,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_small_patch16_224-cd65a155.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def deit_base_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        checkpoint = torch.hub.load_state_dict_from_url(
            url=
            'https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth',
            map_location='cpu',
            check_hash=True)
        model.load_state_dict(checkpoint['model'], strict=False)
    return model


def vit_base_patch16_224(pretrained=False,
                         quant=False,
                         calibrate=False,
                         cfg=None,
                         **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=768,
                              depth=12,
                              num_heads=12,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=True,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'B_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.0-sd_0.0--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model


def vit_large_patch16_224(pretrained=False,
                          quant=False,
                          calibrate=False,
                          cfg=None,
                          **kwargs):
    model = VisionTransformer(patch_size=16,
                              embed_dim=1024,
                              depth=24,
                              num_heads=16,
                              mlp_ratio=4,
                              qkv_bias=True,
                              norm_layer=partial(QIntLayerNorm, eps=1e-6),
                              quant=quant,
                              calibrate=calibrate,
                              input_quant=False,
                              cfg=cfg,
                              **kwargs)
    if pretrained:
        url = 'https://storage.googleapis.com/vit_models/augreg/' + \
            'L_16-i21k-300ep-lr_0.001-aug_medium1-wd_0.1-do_0.1-sd_0.1--imagenet2012-steps_20k-lr_0.01-res_224.npz'

        load_weights_from_npz(model, url, check_hash=True)
    return model
