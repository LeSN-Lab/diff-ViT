# Copyright (c) MEGVII Inc. and its affiliates. All Rights Reserved.
import torch

from .base import BaseObserver
from .utils import lp_loss
from torch.nn import functional as F


class MinmaxObserver(BaseObserver):

    def __init__(self, module_type, bit_type, calibration_mode):
        super(MinmaxObserver, self).__init__(module_type, bit_type,
                                             calibration_mode)
        self.symmetric = self.bit_type.signed

    def update(self, v):
        self.v = v
        #입력 텐서 v의 최대값과 최소값을 계산하고, 이를 갱신합니다
        v = self.reshape_tensor(v) #입력 텐서를 모듈 타입에 맞게 변환함.
        # self.v = v
        cur_max = v.max(axis=1).values #임베딩 벡터 (D, N*P)의 각 차원당 max값을 구함.
        #혹은 각 채널당.
        
        #계속 최댓값을 갱신시켜나감.
        if self.max_val is None:
            self.max_val = cur_max
        else:
            self.max_val = torch.max(cur_max, self.max_val)
            
        #self.max_val, self.min_val: 현재까지 관찰된 최대값과 최소값을 업데이트합니다
        cur_min = v.min(axis=1).values
        if self.min_val is None:
            self.min_val = cur_min
        else:
            self.min_val = torch.min(cur_min, self.min_val)

        if self.calibration_mode == 'layer_wise':
            self.max_val = self.max_val.max()
            self.min_val = self.min_val.min()


    def get_quantization_params(self, x, others=None, attn=False, attn_para=None, *args, **kwargs):
        """
         양자화에 필요한 스케일(scale)과 영점(zero_point)을 계산합니다.
	        •	qmax, qmin: 비트 타입에 따른 양자화 값의 최대, 최소 범위를 나타냅니다.
	        •	scale: 양자화 스케일을 저장할 변수로 초기화합니다.
	        •	zero_point: 영점을 저장할 변수로 초기화합니다.
        
        """
        max_val = self.max_val
        min_val = self.min_val
        self.input = x
        self.others = others
        self.attn = attn
        self.attn_para = attn_para

        qmax = self.bit_type.upper_bound
        qmin = self.bit_type.lower_bound

        # 스케일과 영점을 초기화합니다.
        scale = torch.ones_like(max_val, dtype=torch.float32)
        zero_point = torch.zeros_like(max_val, dtype=torch.int64)

        # 로그를 이용한 반올림 함수를 정의합니다.
        def round_ln(x, type=None):
            if type == 'ceil':
                return torch.ceil(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            elif type == 'floor':
                return torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            else:
                y = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
                out = torch.gt((x-2**y),(2**(y+1)-x))
                return out+y
            # floor = torch.floor(torch.div(torch.log(x),torch.log(torch.Tensor([2]).cuda())))
            # for j in range(self.v.shape[0]):
        
        def get_attn(x):
            B, N, M = x.shape 
            qkv = x.reshape(B, N, 3, self.attn_para[0],
            self.attn_para[1] // self.attn_para[0]).permute(2, 0, 3, 1, 4)
            #  (3, B, num_heads, N, head_dim)
            q, k, v = (
                qkv[0],
                qkv[1],
                qkv[2],
            )
            #QKV를 분리하고 형상변환
            attn = (q @ k.transpose(-2, -1)) * self.attn_para[2]
            """
            self.scale을 곱해주는 이유
            그래디언트 안정성:

            주된 목적은 dot product의 결과값이 너무 크지 않도록 하는 것입니다.
            입력 차원이 커질수록 dot product의 값도 커지는 경향이 있어, 이를 조절할 필요가 있습니다.


            소프트맥스 함수의 안정화:

            스케일링을 하지 않으면 소프트맥스 함수의 입력값이 매우 커질 수 있습니다.
            큰 입력값은 소프트맥스 함수를 포화상태로 만들어 그래디언트가 사라지는 문제를 일으킬 수 있습니다.


            수학적 근거:

            일반적으로 scale = 1 / sqrt(d_k)를 사용합니다. 여기서 d_k는 key의 차원입니다.
            이는 dot product의 분산을 1로 유지하기 위한 것입니다.


            어텐션 분포의 개선:

            스케일링을 통해 어텐션 가중치의 분포가 더 부드러워집니다.
            이는 특정 토큰에 과도하게 집중되는 것을 방지하고, 더 균형 잡힌 어텐션을 가능하게 합니다.
            """
            #projection의 출력을 quantization 시키지 않았음., logint softmax를 사용하지 않았음.
            attn = attn.softmax(dim=-1)
            #단순히 attention의 출력만 구함. B, N, C.
            return (attn @ v).transpose(1, 2).reshape(B, N, self.attn_para[1])
        
        def get_out(x, j, quant=False):
            
            if self.calibration_mode == 'channel_wise':
                #j 번째 채널의 가중치를 선택하고 차원을 추가.
                weight = self.v[j,...].unsqueeze(0)
                if self.others:
                    # j번째 채널의 bias를 선택하고 차원을 추가.
                    bias = self.others[0][j].unsqueeze(0)
            else:
                #전체 가중치 사용
                weight = self.v
                if self.others:
                    #전체 편향 사용
                    bias = self.others[0]
            # FIXME:
            # input_gt = self.input[0]
            # input_q = self.input[0]
            #입력 설정 (GT와 양자화된 입력)
            input_gt = self.input
            input_q = self.input
            if self.module_type == 'activation':
                if quant == True:
                    #layerwise calibration을 하는 경우
                    if self.attn and self.calibration_mode == 'layer_wise': 
                        return get_attn(x)
                    else:
                        return x
                else:
                    # return self.input[0]
                    #quant가 True가 아니면 self.input에 대한 어텐션을 반환한다.
                    if self.attn and self.calibration_mode == 'layer_wise': 
                        return get_attn(self.input)
                    else:
                        return self.input
            elif self.module_type == 'conv_weight':
                if quant == True:
                    #quant가 True이면, 양자화된 입력으로 2D합성곱을 수행한다.
                    return F.conv2d(input_q, x, bias, self.others[1], self.others[2], self.others[3], self.others[4])
                    # return x
                else:
                    #비양자회된 경우, 원본 입력으로 2D 합성곱을 수행한다.
                    return F.conv2d(input_gt, weight, bias, self.others[1], self.others[2], self.others[3], self.others[4])
                    # return weight
            #선형 가중치 모듈인 경우
            elif self.module_type == 'linear_weight': 
                if quant == True:
                    #양자화된 경우, 양자화된 입력으로 선형 연산 수행.
                    out = F.linear(input_q, x, bias)
                    # return x
                else:
                    #비양자화된 경우, 원본 입력으로 선형 연산 수행.
                    out = F.linear(input_gt, weight, bias)
                
                #layerwise calibration을 하는 경우
                if self.attn and self.calibration_mode == 'layer_wise': 
                    return get_attn(out)
                else:
                    return out 

                    # return weight 

        def round_x(scale, x, zero_point=False):
            #PoT 스케일링 팩터의 후보값들을 계산한다.
            alpha_round = round_ln(scale, 'round').cuda() #log2x를 반올림한 값
            alpha_floor = round_ln(scale, 'floor').cuda() #마찬가지임.
            alpha = alpha_round
            
            #zero point를 설정한다.(symmetric quantization)
            if not zero_point:
                zero_point = torch.Tensor([0]).cuda()
            # print(scale.shape)
            
            #layerwise이면 전체 다, channelwise이면 각 채널마다
            if self.calibration_mode == 'layer_wise':
                dim = 1
            else:
                dim = scale.shape[0]
                
            #각 차원에 대해 반복한다.
            for j in range(dim):
                #가중치 설정
                if dim == 1: #layerwise
                    weight = x.cuda()
                    if self.module_type == 'activation':
                        # FIXME:
                        # weight = self.input[0]
                        weight = self.input
                else:
                    #각 차원 j에 대해 가중치를 선택함. laywerwise면 그냥 올리면 되는데
                    #채널별이면 각 채널 j에 대해 나머지 [j, :, :, :, ...]를 선택함.
                    weight = x[j,...].unsqueeze(0).cuda()
                    
                    
                    
                #네 가지 다른 PoT 스케일링 팩터로 양자화를 수행한다.
                #이는 논문의 식 (11)에서 설명된 확장된 search space[α_f - 1, α_c + 1]을 구현한 것
                weight_1 = ((weight / 2**(alpha_floor[j]-1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]-1)
                out_1 = get_out(weight_1, j, quant=True)
                # out_1 = weight_1
                weight_2 = ((weight / 2**(alpha_floor[j]) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j])
                out_2 = get_out(weight_2, j, quant=True)
                # TODO: expand the range
                #round를 쓴게 아니라 그냥 floor에서 1을 더해서 계속 확장한 것 같음.
                weight_3 = ((weight / 2**(alpha_floor[j]+1) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+1)
                out_3 = get_out(weight_3, j, quant=True)
                weight_4 = ((weight / 2**(alpha_floor[j]+2) + zero_point).round().clamp(qmin, qmax) -
                zero_point) * 2**(alpha_floor[j]+2)
                out_4 = get_out(weight_4, j, quant=True)
                # out_2 = weight_2
                out = get_out(weight, j, quant=False)
                # out = weight
                score1 = lp_loss(out, out_1, p=2.0, reduction='all') #mse
                score2 = lp_loss(out, out_2, p=2.0, reduction='all')
                score3 = lp_loss(out, out_3, p=2.0, reduction='all')
                score4 = lp_loss(out, out_4, p=2.0, reduction='all')
                score = [score1, score2, score3, score4]
                # score = [score2, score3]
                indx = score.index(min(score))
                alpha[j] = alpha_floor[j] -1 + indx
                # alpha[j] = alpha_floor[j] + indx
            return alpha

        if self.symmetric:
            zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            max_val = torch.max(-min_val, max_val)
            scale = max_val / (float(qmax - qmin) / 2)
            # TODO: ########### 2^n ############
            alpha_x = round_x(scale, self.v)
            # alpha_x = round_ln(scale, self.v)
            scale = 2**alpha_x
            # if self.module_type in ['conv_weight', 'linear_weight']:
            #     # alpha_x = round_x(scale, self.v)
            #     # scale = 2**alpha_x
            #     pass
            # elif self.module_type == 'activation':
            #     alpha_x = round_x(scale, self.v)
            #     scale = 2**alpha_x
            #     # pass
            # ####################################
            scale.clamp_(self.eps)
        else:
            # zero_point = torch.zeros_like(max_val, dtype=torch.int64)
            scale = (max_val - min_val) / float(qmax - qmin)
            zero_point = qmin - torch.round(min_val / scale)
            zero_point.clamp_(qmin, qmax)
            # TODO: ########### 2^n ############
            alpha_x = round_x(scale, self.v, zero_point)
            scale = 2**alpha_x
            ####################################
            scale.clamp_(self.eps)
        return scale, zero_point
