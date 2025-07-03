import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from lavis.attack_methods.compute_metrics import *
import json
import os


iterations = 10
# eps = 16 / 1
# alpha = 2

max_clamp = torch.tensor([1.9303, 2.0749, 2.1459])
min_clamp = torch.tensor([-1.7923, -1.7521, -1.4802])

mean_r = 0.48145466
std_r = 0.26862954

mean_g = 0.4578275
std_g = 0.26130258

mean_b = 0.40821073
std_b = 0.27577711

std = torch.tensor([std_r, std_g, std_b])
mean = torch.tensor([mean_r, mean_g, mean_b])

def pixel_to_normalized(pixels, std):
    normed = (pixels / 255.0) / std
    normed = torch.round(normed * 10000) / 10000  # 四舍五入到四位小数
    return normed

def clip_norm(eta, norm, eps):
    if norm not in [np.inf, 1, 2]:
        raise ValueError("norm must be np.inf, 1, or 2.")
    if norm == np.inf:
        eta = torch.clamp(eta, -eps, eps)
    return eta

def attack_GI_PreWeight(model, modalities, samples, norm=np.inf):

    eps = 16
    weight = {}
    iterations = 20
    norm_method = 'softmax'
    weight_method = 'grad-cam' 

    alpha = 2

    max_clamp = torch.tensor([1.9303, 2.0749, 2.1459])
    min_clamp = torch.tensor([-1.7923, -1.7521, -1.4802])

    adv_samples = deepcopy(samples)
    torch.set_grad_enabled(True)

    std_tensor = std.to(model.device)
    alpha_tensor = pixel_to_normalized(alpha, std_tensor)

    max_clamp = max_clamp.to(model.device)
    min_clamp = min_clamp.to(model.device)

    g = {}
    for modality in modalities:
        g[modality] = torch.zeros_like(samples[modality]).to(model.device)

    loss_dict = []

    if weight_method != 'grad-cam':
        for modality in modalities:
            eps_tensor = {}
            eps_tensor[modality] = pixel_to_normalized(eps, std_tensor)
            adv_samples = deepcopy(samples)

            adv_samples[modality] = adv_samples[modality].detach().requires_grad_()

            loss = model(samples=adv_samples, methods="MI")['loss'] # encode_input, t5_input, t5_query, fusion_feature
            model.zero_grad()     
            loss.backward(retain_graph=True)
            grad = deepcopy(adv_samples[modality].grad)
            # import pdb; pdb.set_trace()
            if modality == 'rgb':
                shape_eps = eps_tensor[modality].view(1,1,3,1,1)
                shape_alpha = alpha_tensor.view(1,1,3,1,1)
                shape_max = max_clamp.view(1, 1, 3, 1, 1)
                shape_min = min_clamp.view(1, 1, 3, 1, 1)
            else:
                shape_eps = eps_tensor[modality].view(1,3,1,1,1)
                shape_alpha = alpha_tensor.view(1,3,1,1,1)
                shape_max = max_clamp.view(1, 3, 1, 1, 1)
                shape_min = min_clamp.view(1, 3, 1, 1, 1)

            grad = grad / (torch.mean(torch.abs(grad), dim=list(range(1, grad.dim())), keepdim=True) + 1e-8)
            perturb = shape_alpha * torch.sign(grad)
            adv_samples[modality] = adv_samples[modality] + perturb
            eta = adv_samples[modality] - samples[modality]
            eta = clip_norm(eta, np.inf, shape_eps)
            adv_samples[modality] = samples[modality] + eta
            adv_samples[modality] = torch.max(torch.min(adv_samples[modality], shape_max), shape_min)

            loss_value = loss.item()
            loss = model(samples=adv_samples, methods="MI")['loss']
            loss_dict.append(loss.item() - loss_value)
    else:
        adv_samples = deepcopy(samples)
        for modality in modalities:
            adv_samples[modality] = adv_samples[modality].detach().requires_grad_()

        loss = model(samples=adv_samples, methods="MI")['loss'] # encode_input, t5_input, t5_query, fusion_feature
        model.zero_grad()     
        loss.backward(retain_graph=True)
        for modality in modalities:
            loss = torch.sum(adv_samples[modality]*adv_samples[modality].grad)
            loss_value = loss.item()
            loss_dict.append(loss_value)

    if norm_method == "one-hot":
        max_loss = max(loss_dict)
        max_index = loss_dict.index(max_loss)
        for modal in modalities:
            if modal == modalities[max_index]:
                weight[modal] = torch.full((1,), 1.0, device=model.device)
            else:
                weight[modal] = torch.full((1,), 0.0, device=model.device)
    elif norm_method == "softmax":
        temperature = 0.2
        loss_gains = torch.tensor(loss_dict, device=model.device)
        weight_ratios = F.softmax(loss_gains / temperature, dim=0)

        for i, modal in enumerate(modalities):
            weight[modal] = weight_ratios[i]
    elif norm_method == "min-max":
        loss_tensor = torch.tensor(loss_dict, device=model.device)
        min_val = loss_tensor.min()
        max_val = loss_tensor.max()

        if max_val == min_val:
            # 所有loss相同，默认均匀分配
            norm_losses = torch.full_like(loss_tensor, 1.0 / len(loss_tensor))
        else:
            norm_losses = (loss_tensor - min_val) / (max_val - min_val)

        norm_losses = norm_losses / norm_losses.sum()  # 再归一化，使总和为1

        for i, modal in enumerate(modalities):
            weight[modal] = norm_losses[i]
    

    adv_samples = deepcopy(samples)

    pre_attack = 5
    S = 10

    for iter in range(pre_attack):
        
        eps_tensor = {}
        for modal in modalities:
            eps_tensor[modal] = pixel_to_normalized(eps * weight[modal], std_tensor)

        for modality in modalities:
                if iter == 0:
                     eta = torch.zeros_like(adv_samples[modality]).uniform_(-1.0, 1.0)
                else:
                     eta = torch.zeros_like(adv_samples[modality])
                
                if modality == 'rgb':
                     shape_eps = eps_tensor[modality].view(1,1,3,1,1)
                     shape_alpha = alpha_tensor.view(1,1,3,1,1)
                     shape_max = max_clamp.view(1,1,3,1,1)
                     shape_min = min_clamp.view(1,1,3,1,1)
                else:
                     shape_eps = eps_tensor[modality].view(1,3,1,1,1)
                     shape_alpha = alpha_tensor.view(1,3,1,1,1)
                     shape_max = max_clamp.view(1,3,1,1,1)
                     shape_min = min_clamp.view(1,3,1,1,1) 

                eta = eta * shape_eps
                eta = clip_norm(eta, np.inf, shape_eps)
                adv_samples[modality] = (adv_samples[modality] + eta)
                adv_samples[modality] = torch.max(torch.min(adv_samples[modality], shape_max), shape_min)

        for modality in modalities:
            adv_samples[modality] = adv_samples[modality].detach().requires_grad_()

        loss = model(samples=adv_samples, methods="MI")['loss'] # encode_input, t5_input, t5_query, fusion_feature
        model.zero_grad()     
        loss.backward(retain_graph=True)
        for modality in modalities:
            if weight[modality].item() == 0.0:
                continue

            grad = deepcopy(adv_samples[modality].grad)

            if modality == 'rgb':
                shape_eps = eps_tensor[modality].view(1,1,3,1,1)
                shape_alpha = alpha_tensor.view(1, 1, 3, 1, 1)
                shape_max = max_clamp.view(1, 1, 3, 1, 1)
                shape_min = min_clamp.view(1, 1, 3, 1, 1)
            else:
                shape_eps = eps_tensor[modality].view(1,3,1,1,1)
                shape_alpha = alpha_tensor.view(1, 3, 1, 1, 1)
                shape_max = max_clamp.view(1, 3, 1, 1, 1)
                shape_min = min_clamp.view(1, 3, 1, 1, 1)

            momentum = 1.0
            grad = grad / (torch.mean(torch.abs(grad), dim=list(range(1, grad.dim())), keepdim=True) + 1e-8)  # normalize grad
            g[modality] = momentum * g[modality] + grad  # momentum累计
            perturb = shape_alpha * torch.sign(g[modality]) * S

            adv_samples[modality] = adv_samples[modality] + perturb
            eta = adv_samples[modality] - samples[modality]
            eta = clip_norm(eta, np.inf, shape_eps)
            adv_samples[modality] = samples[modality] + eta
            adv_samples[modality] = torch.max(torch.min(adv_samples[modality], shape_max), shape_min)
            adv_samples[modality] = adv_samples[modality].detach()
    
    adv_samples = deepcopy(samples)

    for iter in range(iterations):
        
        eps_tensor = {}
        for modal in modalities:
            eps_tensor[modal] = pixel_to_normalized(eps * weight[modal], std_tensor)

        for modality in modalities:
                if iter == 0:
                     eta = torch.zeros_like(adv_samples[modality]).uniform_(-1.0, 1.0)
                else:
                     eta = torch.zeros_like(adv_samples[modality])
                
                if modality == 'rgb':
                     shape_eps = eps_tensor[modality].view(1,1,3,1,1)
                     shape_alpha = alpha_tensor.view(1,1,3,1,1)
                     shape_max = max_clamp.view(1,1,3,1,1)
                     shape_min = min_clamp.view(1,1,3,1,1)
                else:
                     shape_eps = eps_tensor[modality].view(1,3,1,1,1)
                     shape_alpha = alpha_tensor.view(1,3,1,1,1)
                     shape_max = max_clamp.view(1,3,1,1,1)
                     shape_min = min_clamp.view(1,3,1,1,1) 

                eta = eta * shape_eps
                eta = clip_norm(eta, np.inf, shape_eps)
                adv_samples[modality] = (adv_samples[modality] + eta)
                adv_samples[modality] = torch.max(torch.min(adv_samples[modality], shape_max), shape_min)

        for modality in modalities:
            adv_samples[modality] = adv_samples[modality].detach().requires_grad_()

        loss = model(samples=adv_samples, methods="MI")['loss'] # encode_input, t5_input, t5_query, fusion_feature
        model.zero_grad()     
        loss.backward(retain_graph=True)
        for modality in modalities:
            if weight[modality].item() == 0.0:
                continue

            grad = deepcopy(adv_samples[modality].grad)

            if modality == 'rgb':
                shape_eps = eps_tensor[modality].view(1,1,3,1,1)
                shape_alpha = alpha_tensor.view(1, 1, 3, 1, 1)
                shape_max = max_clamp.view(1, 1, 3, 1, 1)
                shape_min = min_clamp.view(1, 1, 3, 1, 1)
            else:
                shape_eps = eps_tensor[modality].view(1,3,1,1,1)
                shape_alpha = alpha_tensor.view(1, 3, 1, 1, 1)
                shape_max = max_clamp.view(1, 3, 1, 1, 1)
                shape_min = min_clamp.view(1, 3, 1, 1, 1)

            momentum = 1.0
            grad = grad / (torch.mean(torch.abs(grad), dim=list(range(1, grad.dim())), keepdim=True) + 1e-8)  # normalize grad
            g[modality] = momentum * g[modality] + grad  # momentum累计
            perturb = shape_alpha * torch.sign(g[modality])

            adv_samples[modality] = adv_samples[modality] + perturb
            eta = adv_samples[modality] - samples[modality]
            eta = clip_norm(eta, np.inf, shape_eps)
            adv_samples[modality] = samples[modality] + eta
            adv_samples[modality] = torch.max(torch.min(adv_samples[modality], shape_max), shape_min)
            adv_samples[modality] = adv_samples[modality].detach()
        
    torch.set_grad_enabled(False)

    print(weight)
    print(loss_dict)

    return adv_samples