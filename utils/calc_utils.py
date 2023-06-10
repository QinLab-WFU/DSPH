import torch
import numpy as np
from tqdm import tqdm


def calc_hammingDist(B1, B2):
    q = B2.shape[1]
    if len(B1.shape) < 2:
        B1 = B1.unsqueeze(0)
    distH = 0.5 * (q - B1.mm(B2.transpose(0, 1)))
    return distH


def calc_map_k_matrix(qB, rB, query_L, retrieval_L, k=None, rank=0):
    
    num_query = query_L.shape[0]
    if qB.is_cuda:
        qB = qB.cpu()
        rB = rB.cpu()
    map = 0
    if k is None:
        k = retrieval_L.shape[0]
    for iter in range(num_query):
        gnd = (query_L[iter].unsqueeze(0).mm(retrieval_L.t()) > 0).type(torch.float).squeeze()
        tsum = torch.sum(gnd)
        if tsum == 0:
            continue
        hamm = calc_hammingDist(qB[iter, :], rB)
        _, ind = torch.sort(hamm)
        ind.squeeze_()
        gnd = gnd[ind]
        total = min(k, int(tsum))
        count = torch.arange(1, total + 1).type(torch.float).to(gnd.device)
        tindex = torch.nonzero(gnd)[:total].squeeze().type(torch.float) + 1.0
        map += torch.mean(count / tindex)
    map = map / num_query
    return map


def calc_neighbor(label1, label2):
    # calculate the similar matrix
    Sim = label1.matmul(label2.transpose(0, 1)) > 0
    return Sim.float()


def norm_max_min(x: torch.Tensor, dim=None):
    if dim is None:
        max = torch.max(x)
        min = torch.min(x)
    if dim is not None:
        max = torch.max(x, dim=dim)[0]
        min = torch.min(x, dim=dim)[0]
        if dim > 0:
            max = max.unsqueeze(len(x.shape) - 1)
            min = min.unsqueeze(len(x.shape) - 1)
    norm = (x - min) / (max - min)
    return norm


def norm_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = (x - mean) / std
    return norm


def norm_abs_mean(x: torch.Tensor, dim=None):
    if dim is None:
        mean = torch.mean(x)
        std = torch.std(x)
    if dim is not None:
        mean = torch.mean(x, dim=dim)
        std = torch.std(x, dim=dim)
        if dim > 0:
            mean = mean.unsqueeze(len(x.shape) - 1)
            std = std.unsqueeze(len(x.shape) - 1)
    norm = torch.abs(x - mean) / std
    return norm


def factorial(n):
    if n == 0:
        return 1
    else:
        return n * factorial(n - 1)


def calc_IF(all_bow):
    word_num = torch.sum(all_bow, dim=0)
    total_num = torch.sum(word_num)
    IF = word_num / total_num
    return IF
