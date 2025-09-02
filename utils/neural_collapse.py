import copy
import torch

import torch.distributed as dist
import numpy as np

class graph():
    def __init__(self, num_classes=10):

        self.num_classes = num_classes
        self.num_samples = [0 for _ in range(num_classes)]
        self.mean = [0 for _ in range(num_classes)]

        self.NCC_match = 0
        self.WC_cov = 0
        self.BC_cov = 0

        self.norm_M_Cov = 0
        self.norm_W_Cov = 0
        self.W_M_dist = 0

        self.cos_M_std = 0
        self.cos_M_avg = 0
        self.cos_W_std = 0
        self.cos_W_avg = 0

def calculate_mean(graph, hook_data, ys):

    for c in range(graph.num_classes):
        idxs = (ys == c).nonzero(as_tuple=True)[0]

        if len(idxs) == 0:
            continue
        
        hook_data_per_class = hook_data[idxs, :]
        graph.mean[c] += torch.sum(hook_data_per_class, dim=0)
        graph.num_samples[c] += hook_data_per_class.shape[0]
    

def calculate_cov(graph, rank, model, dataloader):

    NCC_match_net = 0
    WC_cov = 0

    graph_mean = torch.stack(graph.mean)
    if dist.is_initialized():
        graph_num_samples = torch.tensor(graph.num_samples).to(rank)
    else:
        graph_num_samples = torch.tensor(graph.num_samples).cuda()

    if dist.is_initialized():
        dist.all_reduce(graph_mean, op=dist.ReduceOp.SUM)
        dist.all_reduce(graph_num_samples, op=dist.ReduceOp.SUM)
        dist.barrier()

    graph.mean = graph_mean
    graph.num_samples = graph_num_samples

    for c in range(graph.num_classes):
        graph.mean[c] /= graph.num_samples[c]

    for i, (xs, ys) in enumerate(dataloader):
        xs = xs.cuda()
        ys = ys.cuda()

        logits = model(xs)
        hook_data = features.value.data.view(xs.shape[0],-1)

        for c in range(graph.num_classes):
            idxs = (ys == c).nonzero(as_tuple=True)[0]

            if len(idxs) == 0:
                continue
            
            hook_data_per_class = hook_data[idxs, :]
            
            net_pred = torch.argmax(logits[idxs,:], dim=1)

            z = hook_data_per_class - graph.mean[c].unsqueeze(0)
            cov = torch.matmul(z.unsqueeze(-1),
                                z.unsqueeze(1))
            
            WC_cov += torch.sum(cov, dim=0)

            NCC_scores = torch.stack([torch.norm(hook_data_per_class[i,:] - graph.mean,dim=1) \
                                        for i in range(hook_data_per_class.shape[0])])
            NCC_pred = torch.argmin(NCC_scores, dim=1)
            NCC_match_net += sum(NCC_pred==net_pred)

    if dist.is_initialized():
        dist.all_reduce(WC_cov, op=dist.ReduceOp.SUM)
        dist.all_reduce(NCC_match_net, op=dist.ReduceOp.SUM)

    WC_cov /= torch.sum(graph.num_samples)
    NCC_match = NCC_match_net/sum(graph.num_samples)

    graph.WC_cov = WC_cov
    graph.NCC_match = NCC_match


def analysis(graph, classifier):

    M = graph.mean.T # (shape of penultimate output, num_classes)

    # global mean

    mean_global = torch.mean(M, dim=1, keepdim=True)

    # between-class covariance

    M_ = M - mean_global
    BC_cov = torch.matmul(M_, M_.T) / graph.num_classes
    graph.BC_cov = BC_cov

    #avg_norm
    W = classifier.weight

    normalized_M = M_ / torch.norm(M_,'fro')
    normalized_W = W.T / torch.norm(W.T,'fro')
    graph.W_M_dist = (torch.norm(normalized_W - normalized_M)**2).item() # NC 3, Fig 5

    M_norms = torch.norm(M_, dim=0)
    W_norms = torch.norm(W.T, dim=0)

    graph.norm_M_cov = (torch.std(M_norms)/torch.mean(M_norms)).item()
    graph.norm_W_cov = (torch.std(W_norms)/torch.mean(W_norms)).item() # Fig 2

    def calculate_angular(V, C): # Fig. 3, 4
        G = V.T @ V
        G -= torch.diag(torch.diag(G))
        G_std = torch.std(G)
        G += torch.ones((C,C)).cuda() / (C-1)
        return G_std, torch.norm(G,1).item() / (C*(C-1))

    cos_M_std, cos_M_avg = calculate_angular(M_/M_norms, graph.num_classes)
    cos_W_std, cos_W_avg = calculate_angular(W.T/W_norms, graph.num_classes)

    graph.cos_M_std = cos_M_std
    graph.cos_M_avg = cos_M_avg
    graph.cos_W_std = cos_W_std
    graph.cos_W_avg = cos_W_avg


from scipy.sparse.linalg import svds

def log_nc(graph, logger, writer):    

    NCC_match = graph.NCC_match
    WC_cov = graph.WC_cov
    BC_cov = graph.BC_cov

    norm_M_cov = graph.norm_M_cov
    norm_W_cov = graph.norm_W_cov

    cos_M_std = graph.cos_M_std
    cos_M_avg = graph.cos_M_avg
    cos_W_std = graph.cos_W_std
    cos_W_avg = graph.cos_W_avg

    # normed diff between global-centered class-mean of penultiamte output and classifier weight
    # NC3
    W_M_dist = graph.W_M_dist

    # tr{Sw Sb^-1}, Fig 6.
    Sw = WC_cov.cpu().numpy()
    Sb = BC_cov.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=graph.num_classes-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    Sw_invSb = np.trace(Sw @ inv_Sb)

    logger.info(
        f'Fig 2: norm_M_cov {norm_M_cov}, norm_W_cov {norm_W_cov}\n'
        f'Fig 3: cos_M_std {cos_M_std}, cos_W_std {cos_W_std}\n'
        f'Fig 4: cos_M_avg {cos_M_avg}, cos_W_avg {cos_W_avg}\n'
        f'Fig 5: W_M_distance {W_M_dist}\n'
        f'Fig 6: Inverse signal-to-noise {Sw_invSb}\n'
        f'Fig 7: NCC match ratio {NCC_match}'
    )
    

def print_nc(graph):    

    NCC_match = graph.NCC_match
    WC_cov = graph.WC_cov
    BC_cov = graph.BC_cov

    norm_M_cov = graph.norm_M_cov
    norm_W_cov = graph.norm_W_cov

    cos_M_std = graph.cos_M_std
    cos_M_avg = graph.cos_M_avg
    cos_W_std = graph.cos_W_std
    cos_W_avg = graph.cos_W_avg

    # normed diff between global-centered class-mean of penultiamte output and classifier weight
    # NC3
    W_M_dist = graph.W_M_dist

    # tr{Sw Sb^-1}, Fig 6.
    Sw = WC_cov.cpu().numpy()
    Sb = BC_cov.cpu().numpy()
    eigvec, eigval, _ = svds(Sb, k=graph.num_classes-1)
    inv_Sb = eigvec @ np.diag(eigval**(-1)) @ eigvec.T 
    Sw_invSb = np.trace(Sw @ inv_Sb)

    print(
        f'Fig 2: norm_M_cov {norm_M_cov}, norm_W_cov {norm_W_cov}\n'
        f'Fig 3: cos_M_std {cos_M_std}, cos_W_std {cos_W_std}\n'
        f'Fig 4: cos_M_avg {cos_M_avg}, cos_W_avg {cos_W_avg}\n'
        f'Fig 5: W_M_distance {W_M_dist}\n'
        f'Fig 6: Inverse signal-to-noise {Sw_invSb}\n'
        f'Fig 7: NCC mismatch ratio {NCC_match}'
    )







