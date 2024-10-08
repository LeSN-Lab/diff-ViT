#*
# @file Different utility functions
# Copyright (c) Zhewei Yao, Amir Gholami
# All rights reserved.
# This file is part of PyHessian library.
#
# PyHessian is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# PyHessian is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with PyHessian.  If not, see <http://www.gnu.org/licenses/>.
#*

import torch
import math
from torch.autograd import Variable
import numpy as np

from pyhessian.utils import group_product, group_add, normalization, get_params_grad, hessian_vector_product, orthnormal


class DDVHessian():
    """
    The class used to compute :
        i) the top 1 (n) eigenvalue(s) of the neural network
        ii) the trace of the entire neural network
        iii) the estimated eigenvalue density
    """

    def __init__(self, 
                 model,
                 q_model,
                 criterion,
                 data=None,
                 adv_data = None,
                 original_ddv=None,
                 dataloader=None,
                 adv_dataloader = None,
                 attack_net = None,
                 cuda=True):
        """
        model: the model that needs Hessain information
        criterion: the loss function
        data: a single batch of data, including inputs and its corresponding labels
        dataloader: the data loader including bunch of batches of data
        """

        # make sure we either pass a single batch or a dataloader
        assert (data != None and dataloader == None) or (data == None and
                                                         dataloader != None)
        assert (adv_data != None and adv_dataloader == None) or (adv_data == None and
                                                         adv_dataloader != None)
        
        self.model = model.eval()  # make model is in evaluation model
        self.q_model = q_model.eval()
        self.criterion = criterion
        self.attack_net = attack_net
        
        if data != None:
            self.data = data
            self.full_dataset = False
        else:
            self.data = dataloader
            self.full_dataset = True
            
        if adv_data != None:
            self.adv_data = adv_data
            self.full_dataset = False
        else:
            self.adv_data = adv_dataloader
            self.full_dataset = True
            
            
        if cuda:
            self.device = 'cuda'
        else:
            self.device = 'cpu'

        # pre-processing for single batch case to simplify the computation.
        if not self.full_dataset:
            self.inputs, self.targets = self.data
            self.adv_inputs, self.adv_targets = self.adv_data
            if self.device == 'cuda':
                self.inputs = self.inputs.cuda()
                self.adv_inputs = self.adv_inputs.cuda()
                self.targets = self.targets.cuda()

            # if we only compute the Hessian information for a single batch data, we can re-use the gradients.
            outputs, _, _ = self.model(self.inputs,  hessian_statistic=True)
            adv_outputs, _, _ = self.model(self.adv_inputs, hessian_statistic=True)
            ddv = torch.matmul(outputs, adv_outputs.t())
            # loss = self.criterion(outputs[0], self.targets)
            loss = self.criterion(ddv, original_ddv)
            loss.backward(create_graph=True)

        # this step is used to extract the parameters from the model
        params, names, gradsH = get_params_grad(self.model)
        self.params = params
        self.names = names
        self.gradsH = gradsH  # gradient used for Hessian computation

    def dataloader_hv_product(self, v):

        device = self.device
        num_data = 0  # count the number of datum points in the dataloader

        THv = [torch.zeros(p.size()).to(device) for p in self.params
              ]  # accumulate result
        for (inputs, targets) in self.data:
            adv_inputs = self.attack_net.gen_adv_inputs(inputs, targets)
            inputs, targets = inputs.cuda(), targets.cuda()
            self.model.zero_grad()
            tmp_num_data = inputs.size(0)
            
            
            outputs, _, _ = self.model(inputs)
            adv_outputs, _, _ = self.model(adv_inputs)
            ddv = torch.matmul(outputs, adv_outputs.t())
            original_ddv = ddv.detach()
            
            q_outputs, _, _ = self.q_model(inputs)
            q_adv_outputs, _, _ = self.q_model(adv_inputs)
            q_ddv = torch.matmul(q_outputs, q_adv_outputs.t())
            
            loss = self.criterion(q_ddv, original_ddv)
            loss.backward(create_graph=True)
            params, gradsH = get_params_grad(self.model)
            self.model.zero_grad()
            Hv = torch.autograd.grad(gradsH,
                                     params,
                                     grad_outputs=v,
                                     only_inputs=True,
                                     retain_graph=False)
            THv = [
                THv1 + Hv1 * float(tmp_num_data) + 0.
                for THv1, Hv1 in zip(THv, Hv)
            ]
            num_data += float(tmp_num_data)

        THv = [THv1 / float(num_data) for THv1 in THv]
        eigenvalue = group_product(THv, v).cpu().item()
        return eigenvalue, THv
    
    def eigenvalues(self, maxIter=100, tol=1e-3, top_n=1):
        """
        compute the top_n eigenvalues using power iteration method
        maxIter: maximum iterations used to compute each single eigenvalue
        tol: the relative tolerance between two consecutive eigenvalue computations from power iteration
        top_n: top top_n eigenvalues will be computed
        """

        assert top_n >= 1

        device = self.device

        eigenvalues = []
        eigenvectors = []

        computed_dim = 0

        while computed_dim < top_n:
            eigenvalue = None
            v = [torch.randn(p.size()).to(device) for p in self.params
                ]  # generate random vector
            v = normalization(v)  # normalize the vector

            for i in range(maxIter):
                v = orthnormal(v, eigenvectors)
                self.model.zero_grad()

                if self.full_dataset:
                    tmp_eigenvalue, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(self.gradsH, self.params, v)
                    tmp_eigenvalue = group_product(Hv, v).cpu().item()

                v = normalization(Hv)

                if eigenvalue == None:
                    eigenvalue = tmp_eigenvalue
                else:
                    if abs(eigenvalue - tmp_eigenvalue) / (abs(eigenvalue) +
                                                           1e-6) < tol:
                        break
                    else:
                        eigenvalue = tmp_eigenvalue
            eigenvalues.append(eigenvalue)
            eigenvectors.append(v)
            computed_dim += 1

        return eigenvalues, eigenvectors
    
    def trace(self, maxIter=150, tol=5e-3):
        """
        compute the trace of hessian using Hutchinson's method
        maxIter: maximum iterations used to compute trace
        tol: the relative tolerance
        """
        global_trace_vhv = []
        device = self.device
        for (i_grad, i_param, module_name) in zip(self.gradsH, self.params, self.names):
            trace_pair={"layer_name":" ", "trace":0}
            trace_pair["layer_name"] = module_name
            trace_vhv = []
            trace = 0.

            for i in range(maxIter):
                
                self.model.zero_grad()
                # v = [
                #     torch.randint_like(p, high=2, device=device)
                #     for p in self.params
                # ]
                v = [torch.randint_like(i_param, high=2, device=device)]   
                
                # generate Rademacher random variables
                # for v_i in v:
                #     v_i[v_i == 0] = -1
                for v_i in v:
                    v_i[v_i==0] = -1 

                if self.full_dataset:
                    _, Hv = self.dataloader_hv_product(v)
                else:
                    Hv = hessian_vector_product(i_grad, i_param, v)
                trace_vhv.append(group_product(Hv, v).cpu().item())
                if abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6) < tol:
                    trace_pair["trace"] = trace
                    global_trace_vhv.append(trace)
                    # global_trace_vhv.append(trace_pair)
                    # print(trace_pair)
                    break
                else:
                    trace = np.mean(trace_vhv)
            if trace_pair["trace"] == 0:
                trace_pair["trace"] = trace
                global_trace_vhv.append(trace)
                # print(trace_pair)
                print("The gap is ", abs(np.mean(trace_vhv) - trace) / (abs(trace) + 1e-6))

        return self.names, global_trace_vhv