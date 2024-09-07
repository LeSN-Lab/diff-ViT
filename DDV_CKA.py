import torch
import torch.nn as nn

class MinibatchAdvCKA(nn.Module):
    def __init__(self,
                 num_layers,
                 num_layers2=None,
                 across_models=False,
                 dtype=torch.float32):
        super(MinibatchAdvCKA, self).__init__()
        # TensorFlow의 add_weight를 PyTorch의 nn.Parameter로 변환
        if num_layers2 is None:
            num_layers2 = num_layers
        self.hsic_accumulator = nn.Parameter(torch.zeros((num_layers, num_layers2), dtype=dtype), requires_grad=False).cuda()
        self.across_models = across_models
        
        self.hsic_accumulator_model1 = nn.Parameter(torch.zeros((num_layers,), dtype=dtype), requires_grad=False).cuda()
        self.hsic_accumulator_model2 = nn.Parameter(torch.zeros((num_layers2,), dtype=dtype), requires_grad=False).cuda()
        

    def _generate_adv_gram_matrix(self, x, adv_x):
        # reshape 연산을 PyTorch 스타일로 변경
        # x = x.view(x.size(0), -1)
        x = x.reshape(x.size(0), -1)
        adv_x = adv_x.reshape(adv_x.size(0), -1)
        # matmul 연산을 PyTorch에 맞게 변경
        gram = torch.matmul(x, adv_x.t())
        n = gram.size(0)
        # set_diag 대신 diagonal 메서드 사용
        gram.diagonal().fill_(0)
        # dtype 캐스팅을 PyTorch 스타일로 변경
        gram = gram.to(self.hsic_accumulator.dtype)
        # reduce_sum을 sum으로 변경, 캐스팅 방식 변경
        means = gram.sum(0) / (n - 2)
        means -= means.sum() / (2 * (n - 1))
        gram -= means.unsqueeze(0)
        gram -= means.unsqueeze(1)
        gram.diagonal().fill_(0)
        # reshape를 view로 변경
        gram = gram.view(-1)
        return gram

    def update_state(self, model1_activations, model1_adv_activations, model2_activations, model2_adv_activations):
        # list comprehension 유지, PyTorch 텐서 연산으로 변경
        model1_layer_grams = torch.stack([self._generate_adv_gram_matrix(x, adv_x) for x, adv_x in zip(model1_activations, model1_adv_activations)])
        # assign_add를 PyTorch의 in-place 덧셈으로 변경
        model2_layer_grams = torch.stack([self._generate_adv_gram_matrix(x, adv_x) for x, adv_x in zip(model2_activations, model2_adv_activations)])
        
        
        #평균이 0인 것을 계속 더한다. 레이어별 유사도 (50, 2500), (50, 2500)을 내적하면 각 레이어마다의 유사도가 나온다.
        self.hsic_accumulator.data.add_(torch.matmul(model1_layer_grams, model2_layer_grams.t())) 
        
        self.hsic_accumulator_model1.data.add_(torch.einsum('ij,ij->i', model1_layer_grams, model1_layer_grams))
        self.hsic_accumulator_model2.data.add_(torch.einsum('ij,ij->i', model2_layer_grams, model2_layer_grams))
        

    def update_state_across_models(self, activations1, activations2):
        # assert 문을 PyTorch 스타일로 변경
        assert self.hsic_accumulator.size(0) == len(activations1), 'Number of activation vectors does not match num_layers.'
        assert self.hsic_accumulator.size(1) == len(activations2), 'Number of activation vectors does not match num_layers.'
        # list comprehension 및 stack 연산 유지, PyTorch 텐서로 변경
        layer_grams1 = torch.stack([self._generate_gram_matrix(x) for x in activations1])
        layer_grams2 = torch.stack([self._generate_gram_matrix(x) for x in activations2])
        # assign_add를 PyTorch의 in-place 덧셈으로 변경
        self.hsic_accumulator.data.add_(torch.matmul(layer_grams1, layer_grams2.t()))
        # einsum 연산을 PyTorch에서 지원하는 방식으로 변경
        self.hsic_accumulator_model1.data.add_(torch.einsum('ij,ij->i', layer_grams1, layer_grams1))
        self.hsic_accumulator_model2.data.add_(torch.einsum('ij,ij->i', layer_grams2, layer_grams2))
    
        
    def result(self):
        # convert_to_tensor 필요 없음, 이미 PyTorch 텐서임
        mean_hsic = self.hsic_accumulator
        
        # sqrt 연산을 PyTorch 함수로 변경
        normalization1 = torch.sqrt(self.hsic_accumulator_model1)
        normalization2 = torch.sqrt(self.hsic_accumulator_model2)
        # 인덱싱 방식을 PyTorch에 맞게 변경
        mean_hsic = mean_hsic / normalization1.unsqueeze(1)
        mean_hsic = mean_hsic / normalization2.unsqueeze(0)
        
        return mean_hsic