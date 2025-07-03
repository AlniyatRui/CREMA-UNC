import torch
import torch.nn.functional as F
from copy import deepcopy
import numpy as np
import os

def pixel_to_normalized(pixels, std):
    normed = (pixels / 255.0) / std
    normed = torch.round(normed * 10000) / 10000
    return normed

def clip_norm(eta, norm, eps):
    if norm == np.inf:
        return torch.clamp(eta, -eps, eps)
    else:
        raise ValueError("Only L_inf is supported in this function.")

def FGSM_Attack(model, modalities, samples, norm=np.inf, iterations=5):
    eps = eval(os.environ["ATTACK_EPS"])
    alpha = eval(os.environ["ALPHA"])
    pre_alpha = eval(os.environ["PREATTACK_ALPHA"])
    temperature = eval(os.environ["SOFTMAX_TEMP"])
    weight = {}

    max_clamp = torch.tensor([1.9303, 2.0749, 2.1459]).to(model.device)
    min_clamp = torch.tensor([-1.7923, -1.7521, -1.4802]).to(model.device)
    std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).to(model.device)

    alpha_tensor = pixel_to_normalized(alpha, std).to(model.device)
    eps_tensor_base = pixel_to_normalized(eps, std).to(model.device)

    adv_samples = deepcopy(samples)
    batch_size = samples[modalities[0]].size(0)

    g = {mod: torch.zeros_like(samples[mod]) for mod in modalities}
    # import pdb; pdb.set_trace()

    # 预评估各模态贡献度（single step FGSM）
    loss_gains = []
    for modality in modalities:
        adv_samples = deepcopy(samples)
        adv_samples[modality] = adv_samples[modality].detach().requires_grad_()
        loss = model(samples=adv_samples)['loss']
        model.zero_grad()
        loss.backward()
        grad = adv_samples[modality].grad

        grad = grad / (torch.mean(torch.abs(grad), dim=list(range(1, grad.dim())), keepdim=True) + 1e-8)
        if modality == 'rgb':
            perturb = alpha_tensor.view(1, 1, -1, 1, 1) * torch.sign(grad) * pre_alpha / alpha
        else:
            perturb = alpha_tensor.view(1, -1, 1, 1, 1) * torch.sign(grad) * pre_alpha / alpha
        adv_samples[modality] = adv_samples[modality] + perturb

        loss_value = loss.item()
        loss = model(samples=adv_samples)['loss']
        loss_gains.append((loss.item() - loss_value))

    losses_tensor = torch.tensor([lg for lg in loss_gains], device=model.device)
    norm_weights = F.softmax(losses_tensor / temperature, dim=0)
    for i, mod in enumerate(modalities):
        weight[mod] = norm_weights[i]

    adv_samples = deepcopy(samples)

    eps_tensor = {mod: eps_tensor_base * weight[mod] for mod in modalities}
    for mod in modalities:
        eta = torch.zeros_like(samples[mod]).uniform_(-1.0, 1.0)
        if mod == 'rgb':
            eta = clip_norm(eta * eps_tensor[mod].view(1, 1, -1, 1, 1), norm, eps_tensor[mod].view(1, 1, -1, 1, 1))
        else:
            eta = clip_norm(eta * eps_tensor[mod].view(1, -1, 1, 1, 1), norm, eps_tensor[mod].view(1, -1, 1, 1, 1))
        adv_samples[mod] = samples[mod] + eta
        if mod == 'rgb':
            adv_samples[mod] = torch.max(torch.min(adv_samples[mod], max_clamp.view(1, 1, -1, 1, 1)), min_clamp.view(1, 1, -1, 1, 1))
        else:
            adv_samples[mod] = torch.max(torch.min(adv_samples[mod], max_clamp.view(1, -1, 1, 1, 1)), min_clamp.view(1, -1, 1, 1, 1))

    # 设置为可导
    for mod in modalities:
        adv_samples[mod] = adv_samples[mod].detach().requires_grad_()

    loss = model(samples=adv_samples)['loss']
    model.zero_grad()
    loss.backward()

    for mod in modalities:
        if weight[mod].item() == 0:
            continue
        grad = adv_samples[mod].grad
        grad = grad / (torch.mean(torch.abs(grad), dim=list(range(1, grad.dim())), keepdim=True) + 1e-8)
        g[mod] = 1.0 * g[mod] + grad
        if mod == 'rgb':
            perturb = eps_tensor[mod].view(1, 1, -1, 1, 1) * torch.sign(g[mod])
        else:
            perturb = eps_tensor[mod].view(1, -1, 1, 1, 1) * torch.sign(g[mod])

        eta = adv_samples[mod] + perturb - samples[mod]
        if mod == "rgb":
            eta = clip_norm(eta, norm, eps_tensor[mod].view(1, 1, -1, 1, 1))
        else:
            eta = clip_norm(eta, norm, eps_tensor[mod].view(1, -1, 1, 1, 1))
        adv_samples[mod] = samples[mod] + eta
        if mod == 'rgb':
            adv_samples[mod] = torch.max(torch.min(adv_samples[mod], max_clamp.view(1, 1, -1, 1, 1)), min_clamp.view(1, 1, -1, 1, 1))
        else:
            adv_samples[mod] = torch.max(torch.min(adv_samples[mod], max_clamp.view(1, -1, 1, 1, 1)), min_clamp.view(1, -1, 1, 1, 1))

    return adv_samples