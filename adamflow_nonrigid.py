import glob
import numpy as np
import argparse
import math
from tqdm import tqdm
import torch

import pyvista as pv

from utils import (
    random_seed,
    sample_mesh_points,
    laplacian
)

from distance import (
    surface_distance,
    chamfer_distance, 
    sliced_wasserstein
)


if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Non-rigid Surface Registration')
    parser.add_argument('--data_dir', default='YOUR_DATA_DIR', type=str, help='data directory')
    parser.add_argument('--save_dir', default='YOUR_SAVE_DIR', type=str, help='save directory')
    parser.add_argument('--organ', default='liver', type=str, help='type of organ')
    parser.add_argument('--method', default='AdamFlow', type=str, help='optimisation method: {Chamfer,SWD,WGF,HBF,Nesterov,AdamFlow}')
    parser.add_argument('--n_pair', default=300, type=int, help='number of registration pairs')
    parser.add_argument('--device', default='cuda:0', type=str, help='cpu or gpu')

    parser.add_argument('--w_lapl', default=2.0, type=float, help="mesh Laplacian regularisation")
    parser.add_argument('--n_proj', default=4, type=int, help='number of projections')
    parser.add_argument('--alpha', default=0.9, type=float, help='alpha')
    parser.add_argument('--beta', default=0.95, type=float, help='beta')
    parser.add_argument('--eps', default=1e-10, type=float, help='small number')
    parser.add_argument('--h', default=1.0, type=float, help='step size')

    parser.add_argument('--eta_swd', default=0.5, type=float, help='learning rate for sliced Wasserstein distance')
    parser.add_argument('--eta_cham', default=0.1, type=float, help='learning rate for Chamfer distance')

    parser.add_argument('--K_swd', default=500, type=int, help='optimisation steps for sliced Wasserstein distance')
    parser.add_argument('--K_cham', default=200, type=int, help='optimisation steps for Chamfer distance')

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    organ = args.organ
    method = args.method
    n_pair = args.n_pair
    device = args.device

    w_lapl = args.w_lapl
    n_proj = args.n_proj
    alpha = args.alpha
    beta = args.beta
    h = args.h
    eps = args.eps

    eta_swd = args.eta_swd
    eta_cham = args.eta_cham
    K_swd = args.K_swd
    K_cham = args.K_cham
    K = K_swd + K_cham

    print('organ:', organ)
    print('method:', method)

    data_dir = data_dir+organ+'/'
    subj_list = sorted(glob.glob(data_dir+'*'))

    n_subj = len(subj_list)
    print('num of subject:', n_subj)
    random_seed(12345)
    align_pair = np.random.randint(0,n_subj,[n_pair, 2])

    assd_err_list = []
    hd90_err_list = []

    for i in tqdm(range(n_pair)):
        assd_err_i = []
        hd90_err_i = []

        id_fix, id_move = align_pair[i]
        subj_id_fix = subj_list[id_fix].split('/')[-1]
        subj_id_move = subj_list[id_move].split('/')[-1]
        mesh_fix = pv.read(data_dir+subj_id_fix+'/mesh_'+organ+'_'+subj_id_fix+'.vtk')
        mesh_move = pv.read(data_dir+subj_id_move+'/mesh_'+organ+'_'+subj_id_move+'.vtk')

        vert_fix, face_fix = mesh_fix.points, mesh_fix.faces.reshape(-1,4)[:,1:]
        vert_move, face_move = mesh_move.points, mesh_move.faces.reshape(-1,4)[:,1:]
        # match center
        vert_move = vert_move - vert_move.mean(0) + vert_fix.mean(0)

        vert_fix = torch.tensor(vert_fix[None], dtype=torch.float32, device=device)
        face_fix = torch.tensor(face_fix[None], dtype=torch.int64, device=device)
        vert_move = torch.tensor(vert_move[None], dtype=torch.float32, device=device)
        face_move = torch.tensor(face_move[None], dtype=torch.int64, device=device)
        L_move = laplacian(face_move)

        random_seed(12345)
        # sample points for evaluation
        point_fix_eval = sample_mesh_points(vert_fix, face_fix, n_sample=50000)
        point_move_eval = sample_mesh_points(vert_move, face_move, n_sample=50000)
        assd, hd90 = surface_distance(point_move_eval, point_fix_eval, hausdorff=True, percentile=0.9)
        assd_err_i.append(assd.item())
        hd90_err_i.append(hd90.item())

        n_move = vert_move.shape[1]
        x_k = vert_move.clone()
        m_k = torch.zeros_like(x_k)
        v_k = torch.zeros_like(x_k)
        eta = eta_swd

        # optimisation
        random_seed(12345)
        for k in range(K):
            point_fix = sample_mesh_points(vert_fix, face_fix, n_sample=n_move)
            if k == K_swd:
                m_k = torch.zeros_like(x_k)
                v_k = torch.zeros_like(x_k)
                eta = eta_cham
            if k < K_swd:
                _, grad_k = sliced_wasserstein(x_k, point_fix, n_proj=n_proj)
            else:
                _, grad_k = chamfer_distance(x_k, point_fix)

            lapl_grad = L_move.bmm(x_k)
            grad_k += w_lapl * lapl_grad

            if method in ['Chamfer', 'SWD', 'WGF']:
                x_k = x_k - eta * grad_k

            if method == 'HBF':
                m_k = m_k + h * (-alpha * m_k - grad_k)
                x_k = x_k + h * eta * m_k

            if method == 'Nesterov':
                t = h * (k+1)
                m_k = m_k + h * (-3/t * m_k - grad_k)
                x_k = x_k + h * eta * m_k

            if method == 'AdamFlow':
                t = h * (k+1)
                m_k = m_k + h * (1 - alpha) * (grad_k - m_k)
                v_k = v_k + h * (1 - beta) * (grad_k**2 - v_k)
                m_hat_k = m_k / (1 - math.exp(-(1-alpha)*t))
                v_hat_k = v_k / (1 - math.exp(-(1-beta)*t))
                x_k = x_k - h * eta * m_hat_k / (v_hat_k.sqrt() + eps)

            if (k+1) % 50 == 0:
                point_move_eval = sample_mesh_points(x_k, face_move, n_sample=50000)
                assd, hd90 = surface_distance(
                    point_move_eval,
                    point_fix_eval,
                    hausdorff=True, percentile=0.9)
                assd_err_i.append(assd.item())
                hd90_err_i.append(hd90.item())

        assd_err_list.append(assd_err_i)
        hd90_err_list.append(hd90_err_i)
    assd_err_list = np.stack(assd_err_list)
    hd90_err_list = np.stack(hd90_err_list)

    print('assd:', assd_err_list[:,-1].mean())
    print('hd90:', hd90_err_list[:,-1].mean())
    np.save(save_dir+'nonrigid_assd_'+method+'_'+organ+'.npy', assd_err_list)
    np.save(save_dir+'nonrigid_hd90_'+method+'_'+organ+'.npy', hd90_err_list)
