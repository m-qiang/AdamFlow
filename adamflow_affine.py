import glob
import numpy as np
import argparse
import math
from tqdm import tqdm
import torch

import pyvista as pv

from utils import (
    random_seed,
    sample_mesh_points
)

from distance import (
    surface_distance,
    icp_distance, 
    sliced_wasserstein
)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Affine Surface Registration')
    parser.add_argument('--data_dir', default='YOUR_DATA_DIR', type=str, help='data directory')
    parser.add_argument('--save_dir', default='YOUR_SAVE_DIR', type=str, help='save directory')
    parser.add_argument('--organ', default='liver', type=str, help='type of organ')
    parser.add_argument('--method', default='AdamFlow', type=str, help='optimisation method: {ICP,WGF,HBF,Nesterov,AdamFlow}')
    parser.add_argument('--n_pair', default=300, type=int, help='number of registration pairs')
    parser.add_argument('--device', default='cuda:0', type=str, help='cpu or gpu')

    parser.add_argument('--n_proj', default=4, type=int, help='number of projections')
    parser.add_argument('--alpha', default=0.9, type=float, help='alpha')
    parser.add_argument('--beta', default=0.95, type=float, help='beta')
    parser.add_argument('--eta', default=1e-2, type=float, help='learning rate')
    parser.add_argument('--K', default=1500, type=int, help='optimisation steps')
    parser.add_argument('--h', default=1.0, type=float, help='step size')
    parser.add_argument('--eps', default=1e-10, type=float, help='small number')

    args = parser.parse_args()

    data_dir = args.data_dir
    save_dir = args.save_dir
    organ = args.organ
    method = args.method
    n_pair = args.n_pair
    device = args.device
    
    n_proj = args.n_proj
    alpha = args.alpha
    beta = args.beta
    eta = args.eta
    K = args.K
    h = args.h
    eps = args.eps

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

        vert_fix = torch.tensor(vert_fix[None], dtype=torch.float32, device=device)
        face_fix = torch.tensor(face_fix[None], dtype=torch.int64, device=device)
        vert_move = torch.tensor(vert_move[None], dtype=torch.float32, device=device)
        face_move = torch.tensor(face_move[None], dtype=torch.int64, device=device)

        random_seed(12345)
        point_fix_eval = sample_mesh_points(vert_fix, face_fix, n_sample=50000)
        point_move_eval = sample_mesh_points(vert_move, face_move, n_sample=50000)
        assd, hd90 = surface_distance(point_move_eval, point_fix_eval, hausdorff=True, percentile=0.9)
        assd_err_i.append(assd.item())
        hd90_err_i.append(hd90.item())

        n_move = vert_move.shape[1]
        A_k = torch.eye(3, dtype=torch.float32, device=device)
        b_k = torch.zeros([3], dtype=torch.float32, device=device)

        m_A_k = torch.zeros_like(A_k)
        v_A_k = torch.zeros_like(A_k)
        m_b_k = torch.zeros_like(b_k)
        v_b_k = torch.zeros_like(b_k)

        # optimisation
        random_seed(12345)

        for k in range(K):
            point_fix = sample_mesh_points(vert_fix, face_fix, n_sample=n_move)
            x_k = vert_move @ A_k.T + b_k
            if method == 'ICP':
                _, grad_k = icp_distance(x_k, point_fix)
            else:
                _, grad_k = sliced_wasserstein(x_k, point_fix, n_proj=n_proj)
            grad_A_k = grad_k[0].T @ vert_move[0] / n_move
            grad_b_k = grad_k[0].sum(0) / n_move

            if method == 'ICP' or method == 'WGF':
                A_k = A_k - eta * grad_A_k
                b_k = b_k - eta * grad_b_k

            if method == 'HBF':
                m_A_k = m_A_k + h * (-alpha * m_A_k - grad_A_k)
                m_b_k = m_b_k + h * (-alpha * m_b_k - grad_b_k)
                A_k = A_k + h * eta * m_A_k
                b_k = b_k + h * eta * m_b_k

            if method == 'Nesterov':
                t = h * (k+1)
                m_A_k = m_A_k + h * (-3/t * m_A_k - grad_A_k)
                m_b_k = m_b_k + h * (-3/t * m_b_k - grad_b_k)
                A_k = A_k + h * eta * m_A_k
                b_k = b_k + h * eta * m_b_k

            if method == 'AdamFlow':
                t = h * (k+1)

                m_A_k = m_A_k + h * (1 - alpha) * (grad_A_k - m_A_k)
                v_A_k = v_A_k + h * (1 - beta) * (grad_A_k**2 - v_A_k)
                m_A_hat_k = m_A_k / (1 - math.exp(-(1-alpha)*t))
                v_A_hat_k = v_A_k / (1 - math.exp(-(1-beta)*t))
                A_k = A_k - h * eta * m_A_hat_k / (v_A_hat_k.sqrt() + eps)

                m_b_k = m_b_k + h * (1 - alpha) * (grad_b_k - m_b_k)
                v_b_k = v_b_k + h * (1 - beta) * (grad_b_k**2 - v_b_k)
                m_b_hat_k = m_b_k / (1 - math.exp(-(1-alpha)*t))
                v_b_hat_k = v_b_k / (1 - math.exp(-(1-beta)*t))
                b_k = b_k - h * eta * m_b_hat_k / (v_b_hat_k.sqrt() + eps)

            if (k+1) % 50 == 0:
                assd, hd90 = surface_distance(
                    point_move_eval @ A_k.T + b_k, 
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
    np.save(save_dir+'affine_assd_'+method+'_'+organ+'.npy', assd_err_list)
    np.save(save_dir+'affine_hd90_'+method+'_'+organ+'.npy', hd90_err_list)
