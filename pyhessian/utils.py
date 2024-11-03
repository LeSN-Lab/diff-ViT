# *
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
# *

import torch
import math
from torch.autograd import Variable
import numpy as np


def group_product(xs, ys):
    """
    the inner product of two lists of variables xs,ys
    :param xs:
    :param ys:
    :return:
    """
    return sum([torch.sum(x * y) for (x, y) in zip(xs, ys)])


def group_add(params, update, alpha=1):
    """
    params = params + update*alpha
    :param params: list of variable
    :param update: list of data
    :return:
    """
    for i, p in enumerate(params):
        params[i].data.add_(update[i] * alpha)
    return params


def normalization(v):
    """
    normalization of a list of vectors
    return: normalized vectors v
    """
    s = group_product(v, v)
    s = s**0.5
    s = s.cpu().item()
    v = [vi / (s + 1e-6) for vi in v]
    return v


def get_params_grad(model, target_layer_idx=None):
    """
    get model parameters and corresponding gradients up to the target layer
    Args:
        model: the neural network model
        target_layer_idx: the index of the target layer (None means all layers)
    """
    names = []
    params = []
    grads = []

    current_layer = 0
    for name, param in model.named_parameters():
        # Skip if parameter doesn't require gradient
        if not param.requires_grad:
            continue

        # Skip certain layers based on name
        if (
            "norm" in name
            or "bias" in name
            or "cls_token" in name
            or "pos_embed" in name
            or "patch_embed" in name
        ):
            continue

        # Extract block number from parameter name
        if "blocks." in name:
            block_num = int(name.split("blocks.")[1].split(".")[0])

            # Skip if we've passed the target layer
            if target_layer_idx is not None and block_num > target_layer_idx:
                continue

        params.append(param)
        names.append(name)
        grads.append(0.0 if param.grad is None else param.grad + 0.0)

        current_layer += 1

    return params, names, grads


def hessian_vector_product(gradsH, params, v):
    """
    compute the hessian vector product of Hv, where
    gradsH is the gradient at the current point,
    params is the corresponding variables,
    v is the vector.
    """
    try:
        hv = torch.autograd.grad(
            gradsH, params, grad_outputs=v, only_inputs=True, retain_graph=True
        )
        return hv
    except RuntimeError as e:
        print("Error in hessian_vector_product", e)
        return None


def orthnormal(w, v_list):
    """
    make vector w orthogonal to each vector in v_list.
    afterwards, normalize the output w
    """
    for v in v_list:
        w = group_add(w, v, alpha=-group_product(w, v))
    return normalization(w)
