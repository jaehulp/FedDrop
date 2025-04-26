import os
import copy
import torch
import torch.nn.functional as F

from utils.ops import features, hook, accuracy, AverageMeter, compute_cov
from utils.model import load_model
from utils.dataset import read_client_data

def return_client_model_list(epoch, num_client, num_join_client, saved_dir):
    model_list = []
    idx_list = []
    saved_dir = os.path.join(saved_dir, 'client')
    for i in range(0, num_client):
        client_dir = os.path.join(saved_dir, str(i))
        model_path = os.path.join(client_dir, f'epoch_{epoch}.pt')
        if os.path.isfile(model_path) == True:
            model = torch.load(model_path)
            if any(key.startswith("module.") for key in model.keys()):
                model = {key.replace("module.", ""): value for key, value in model.items()}
            model_list.append(model)
            idx_list.append(str(i))
    assert(len(model_list) == (num_join_client))

    return model_list, idx_list

def batched_angle_between(A, B, eps=1e-8):
    # Normalize each row
    A_norm = F.normalize(A, p=2, dim=1, eps=eps)
    B_norm = F.normalize(B, p=2, dim=1, eps=eps)

    # Compute cosine similarity (dot product per row)
    cos_theta = torch.sum(A_norm * B_norm, dim=1)
    cos_theta = torch.clamp(cos_theta, -1.0, 1.0)

    # Compute angle in radians
    angles_rad = torch.acos(cos_theta)  # shape: (N,)
    return angles_rad  # optional: .unsqueeze(1) if you want shape (N, 1)

def compute_cov_mean(server_output, client_output):
    server_mean = torch.mean(server_output, dim=0, keepdim=True)
    server_output = server_output - server_mean

    client_mean = torch.mean(client_output, dim=0, keepdim=True)
    client_output = client_output - client_mean

    U, S, Vt_s = torch.linalg.svd(server_output)
    svd_server_output = (Vt_s[:50] @ server_output.permute(1,0)).T
    U, S, Vt_c = torch.linalg.svd(client_output)

    svd_client_output = (Vt_c[:50] @ client_output.permute(1,0)).T

    output_mat = torch.stack([server_output, client_output])
    cov_val = compute_cov(output_mat)
    mean = torch.mean(cov_val)
    std = torch.std(cov_val)

    output_mat = torch.stack([svd_server_output, svd_client_output])
    svd_cov_val = compute_cov(output_mat)
    svd_cov_val = abs(svd_cov_val)
    svd_mean = torch.mean(svd_cov_val)
    svd_std = torch.std(svd_cov_val)

    topk_svd_mean = torch.mean(svd_cov_val[:5])

#    angle_rad = batched_angle_between(Vt_s[:5], Vt_c[:5])
#    print(angle_rad[:5] * 180 / torch.pi)

    return mean, svd_mean, topk_svd_mean

from utils.ops import features, accuracy, AverageMeter

def measure_client_cov(epoch, model_dir, data_dir, model_args, model_list, idx_list):

    server_model = load_model(model_args)
    server_state_dict = torch.load(os.path.join(model_dir, f'epoch_{epoch-1}.pt'))
    server_model.load_state_dict(server_state_dict)
    server_hook = server_model.fc.register_forward_hook(hook)
    server_model.cuda()

    client_model = load_model(model_args)
    client_model.cuda()
    
    server_acc_list = []
    client_acc_list = []
    acc1_meter = AverageMeter()

    cov_mean_list = []
    svd_cov_mean_list = []
    topk_svd_mean_list = []

    for m, idx in zip(model_list, idx_list):
        data = read_client_data(data_dir, idx)
        dataloader = torch.utils.data.DataLoader(data, 500, shuffle=False)
        
        acc1_meter.reset()
        server_output = []
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.cuda()
            ys = ys.cuda()
            logits = server_model(xs)
            acc1, _ = accuracy(logits, ys, topk=(1,5))
            acc1_meter.update(acc1)
            hook_data = features.value.data.view(xs.shape[0], -1).cuda()
            server_output.append(hook_data)
        server_acc_list.append(acc1_meter.result().item())

        client_model.load_state_dict(m)
        acc1_meter.reset()
        client_output = []
        client_hook = client_model.fc.register_forward_hook(hook)
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.cuda()
            ys = ys.cuda()
            logits = client_model(xs)
            acc1, _ = accuracy(logits, ys, topk=(1,5))
            acc1_meter.update(acc1)
            hook_data = features.value.data.view(xs.shape[0], -1).cuda()
            client_output.append(hook_data)
        client_hook.remove()
        client_acc_list.append(acc1_meter.result().item())

        server_output = torch.cat(server_output)
        client_output = torch.cat(client_output)
        cov_mean, svd_cov_mean, topk_svd_mean = compute_cov_mean(server_output, client_output)
        cov_mean_list.append(cov_mean.item())
        svd_cov_mean_list.append(svd_cov_mean.item())
        topk_svd_mean_list.append(topk_svd_mean.item())

    return client_acc_list, cov_mean_list, svd_cov_mean_list, topk_svd_mean_list

def measure_client_testset(epoch, model_dir, data_dir, dataloader, model_args, model_list, idx_list):

    server_model = load_model(model_args)
    server_state_dict = torch.load(os.path.join(model_dir, f'epoch_{epoch-1}.pt'))
    server_model.load_state_dict(server_state_dict)
    server_hook = server_model.fc.register_forward_hook(hook)
    server_model.cuda()

    client_model = load_model(model_args)
    client_model.cuda()
    
    client_acc_list = []
    acc1_meter = AverageMeter()

    cov_mean_list = []
    svd_cov_mean_list = []
    topk_svd_mean_list = []

    server_output = []
    
    pred_list = []
    pred_mat = []

    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()
        logits = server_model(xs)
        acc1, _ = accuracy(logits, ys, topk=(1,5))
        _, pred = logits.topk(1, 1, True, True)
        pred_list.append(pred.t().squeeze().detach())
        acc1_meter.update(acc1)
        hook_data = features.value.data.view(xs.shape[0], -1).cuda()
        server_output.append(hook_data)
    server_output = torch.cat(server_output)
    server_hook.remove()

    pred_list = torch.cat(pred_list)
    pred_mat.append(pred_list)

    for m, idx in zip(model_list, idx_list):
        data = torchvision.datasets.CIFAR10(
            root='./data/cifar10', train=False, download=True, transform=torchvision.transforms.ToTensor()
        )
        dataloader = torch.utils.data.DataLoader(data, 500, shuffle=False)
        
        acc1_meter.reset()

        client_model.load_state_dict(m)
        acc1_meter.reset()
        client_output = []
        pred_list = []
        client_hook = client_model.fc.register_forward_hook(hook)
        for i, (xs, ys) in enumerate(dataloader):
            xs = xs.cuda()
            ys = ys.cuda()
            logits = client_model(xs)
            acc1, _ = accuracy(logits, ys, topk=(1,5))
            _, pred = logits.topk(1, 1, True, True)
            acc1_meter.update(acc1)
            hook_data = features.value.data.view(xs.shape[0], -1).cuda()
            client_output.append(hook_data)
            pred_list.append(pred.t().squeeze().detach())
        client_hook.remove()
        client_acc_list.append(acc1_meter.result().item())

        pred_list = torch.cat(pred_list)
        pred_mat.append(pred_list)

        client_output = torch.cat(client_output)
        cov_mean, svd_cov_mean, topk_svd_mean = compute_cov_mean(server_output, client_output)
        cov_mean_list.append(cov_mean.item())
        svd_cov_mean_list.append(svd_cov_mean.item())
        topk_svd_mean_list.append(topk_svd_mean.item())

    pred_mat = torch.stack(pred_mat)
    pred_mat_1 = pred_mat.unsqueeze(0)
    pred_mat_2 = pred_mat.unsqueeze(1)
    pred_mat = (pred_mat_1 != pred_mat_2)

    dis_mat = torch.sum(pred_mat, dim=2) / pred_mat.shape[2]
    dis_list = dis_mat[0, 1:]

    return client_acc_list, cov_mean_list, svd_cov_mean_list, topk_svd_mean_list, dis_list

import torchvision

def add_parameters(w, agg_model, client_model):

    for server_param, client_param in zip(agg_model.to(0).parameters(), client_model.to(0).parameters()):
        server_param.data += client_param.data.clone() * w

def simple_model_aggregate(model, model_list, w, dataloader):

    w = torch.tensor(w).cuda()
    for param in model.parameters():
        param.data.zero_()
    client_model = copy.deepcopy(model)

    for m, w in zip(model_list, w):
        client_model.load_state_dict(m)
        add_parameters(w, model, client_model)

    acc1_meter = AverageMeter()
    for j, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()
        logits = model(xs)
        acc1, _ = accuracy(logits, ys, topk=(1,5))
        acc1_meter.update(acc1)

    return acc1_meter.result()
