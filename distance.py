import numpy as np
import torch
from pytorch3d.ops import knn_points


def surface_distance(x, y, hausdorff=False, percentile=0.9):
    '''
    x: (1,N_x,3)
    y: (1,N_y,3)
    '''
    y_index = knn_points(x,y).idx[0,:,0]
    x_index = knn_points(y,x).idx[0,:,0]

    dist_x = (x-y[:,y_index]).norm(dim=-1)
    dist_y = (y-x[:,x_index]).norm(dim=-1)
    assd = (dist_x.mean() + dist_y.mean()) / 2

    if hausdorff:
        # compute Hausdorff distance
        x_quantile = torch.quantile(dist_x, percentile, dim=-1)  # (B)
        y_quantile = torch.quantile(dist_y, percentile, dim=-1)  # (B)
        hd = torch.maximum(x_quantile, y_quantile).mean()
        return assd, hd
    else:
        return assd


def icp_distance(x,y):
    '''
    x: (1,N_x,3)
    y: (1,N_y,3)
    '''
    n_x, n_y = x.shape[1], y.shape[1]

    y_index = knn_points(x,y).idx[0,:,0]

    # closest point distance
    dist = 0.5*((x-y[:,y_index])**2).sum()
    
    # closest point gradient
    grad = x-y[:,y_index]
    
    return dist, grad


def chamfer_distance(x,y):
    '''
    x: (1,N_x,3)
    y: (1,N_y,3)
    '''
    n_x, n_y = x.shape[1], y.shape[1]

    y_index = knn_points(x,y).idx[0,:,0]
    x_index = knn_points(y,x).idx[0,:,0]

    # chamfer distance
    dist = 0.5*((x-y[:,y_index])**2).sum() + 0.5*((y-x[:,x_index])**2).sum()
    
    # chamfer gradient
    grad = x-y[:,y_index]
    grad.index_add_(1, x_index, x[:,x_index]-y)
    
    return dist/2.0, grad/2.0


def sliced_wasserstein(x, y, n_proj=20):
    '''
    Compute sliced Wasserstein distance and its Wasserstein gradient
    input:
        x (B,N,D)
        y (B,N,D)
        n_proj (int)
    Return:
        dist (B,scalar)
        grad (B,N,D)
    '''

    device = x.device
    
    B = x.shape[0]   # batch size
    D = x.shape[-1]  # dimensionality
    b_id = torch.arange(B)[:,None,None,None]  # indices for batch
    d_id = torch.arange(D)[None,None,:,None]  # indices for dimension
    p_id = torch.arange(n_proj)[None,None,None,:]  # indices for projection
    
    theta = torch.randn(B, D, n_proj, device=device)  # (B, D, n_proj)
    theta = theta / torch.linalg.norm(theta, ord=2, dim=1, keepdim=True)

    x_proj = x @ theta  # (B, N, D) @ (B, D, n_proj) --> (B, N, n_proj)
    y_proj = y @ theta  # (B, N, D) @ (B, D, n_proj) --> (B, N, n_proj)
    x_sort, x_id = x_proj.sort(dim=1)
    y_sort, y_id = y_proj.sort(dim=1)
        
    # 1d Wasserstein distance by sorting
    dist = 0.5*((x_sort - y_sort) ** 2).mean(dim=(-1,-2))
    
    # (B, N, 1, n_proj) * (B, 1, D, n_proj) --> (B, N, D, n_proj)
    grad = (x_sort - y_sort)[:,:,None] * theta[:,None]
    x_id = x_id[:,:,None]
    grad[b_id,x_id,d_id,p_id] = grad.clone()  # permute gradient to original order
    grad = grad.mean(-1)

    return dist, grad

