import torch
import torch.nn.functional as F
import numpy as np
from pytorch3d.ops.knn import knn_points
from skimage.measure import label as connected


def random_seed(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    

def adjacent_faces(face):
    """
    Find the adjacent two faces for each edge
    
    Inputs:
    - face: mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - adj_faces: indices of two adjacent two faces
    for each edge, (|E|, 2) torch.LongTensor
    
    """
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], axis=0)  # (2|E|, 2)
    nf = face.shape[1]
    # map the edge to its belonging face
    fid = torch.arange(nf).to(face.device)
    adj_faces = torch.cat([fid]*3)  # (3|F|)

    edge = edge.cpu().numpy()
    # sort the edge such that v_i < v_j
    edge = np.sort(edge, axis=-1)
    # sort the edge to find the correspondence 
    # between e_ij and e_ji
    eid = np.lexsort((edge[:,1], edge[:,0]))  # (2|E|)

    # map edge to its adjacent two faces
    adj_faces = adj_faces[eid].reshape(-1,2)  # (|E|, 2)
    return adj_faces


def vert_normal(vert, face):
    """
    Compute the normal vector of each vertex.
    
    This function is retrieved from pytorch3d.
    For original code please see: 
    _compute_vertex_normals function in
    https://pytorch3d.readthedocs.io/en/latest/
    _modules/pytorch3d/structures/meshes.html
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - v_normal: vertex normals, (1,|V|,3) torch.Tensor
    """
    
    v_normal = torch.zeros_like(vert)   # normals of vertices
    v_f = vert[:, face[0]]   # vertices of each face

    # compute normals of faces
    f_normal_0 = torch.cross(v_f[:,:,1]-v_f[:,:,0], v_f[:,:,2]-v_f[:,:,0], dim=2) 
    f_normal_1 = torch.cross(v_f[:,:,2]-v_f[:,:,1], v_f[:,:,0]-v_f[:,:,1], dim=2) 
    f_normal_2 = torch.cross(v_f[:,:,0]-v_f[:,:,2], v_f[:,:,1]-v_f[:,:,2], dim=2) 

    # sum the faces normals
    v_normal = v_normal.index_add(1, face[0,:,0], f_normal_0)
    v_normal = v_normal.index_add(1, face[0,:,1], f_normal_1)
    v_normal = v_normal.index_add(1, face[0,:,2], f_normal_2)

    v_normal = v_normal / (torch.norm(v_normal, dim=-1).unsqueeze(-1) + 1e-12)
    return v_normal


def face_normal(vert, face):
    """
    Compute the normal vector of each face.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - f_normal: face normals, (1,|F|,3) torch.Tensor
    """
    
    v_f = vert[:, face[0]]
    # compute normals of faces
    f_normal = torch.cross(v_f[:,:,1]-v_f[:,:,0],
                           v_f[:,:,2]-v_f[:,:,0], dim=2) 
    f_normal = f_normal / (torch.norm(f_normal, dim=-1).unsqueeze(-1) + 1e-12)

    return f_normal
    
    
def mesh_area(vert, face):
    """
    Compute the total area of the mesh

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area: mesh area, float
    """

    v0 = vert[:,face[0,:,0]]
    v1 = vert[:,face[0,:,1]]
    v2 = vert[:,face[0,:,2]]
    area = 0.5*torch.norm(torch.cross(v1-v0, v2-v0), dim=-1)
    return area.sum()


def face_area(vert, face):
    """
    Compute the area of each face

    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - area: face area, (|F|,3) torch.Tensor
    """
    
    v0 = vert[:,face[0,:,0]]
    v1 = vert[:,face[0,:,1]]
    v2 = vert[:,face[0,:,2]]
    area = 0.5*torch.norm(torch.cross(v1-v0, v2-v0), dim=-1)
    return area[0]

    
def laplacian(face):
    """
    Compute Laplacian matrix.
    
    Inputs:
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    
    Returns:
    - L: Laplacian matrix, (1,|V|,|V|) torch.sparse.Tensor
    """
    nv = face.max().item()+1
    edge = torch.cat([face[0,:,[0,1]],
                      face[0,:,[1,2]],
                      face[0,:,[2,0]]], dim=0).T
    # adjacency matrix A
    A = torch.sparse_coo_tensor(
        edge, torch.ones_like(edge[0]).float(), (nv, nv)).unsqueeze(0)

    # number of neighbors for each vertex
    degree = torch.sparse.sum(A, dim=-1).to_dense()[0]
    weight = 1./degree[edge[0]]
    # normalized adjacency matrix
    A_hat = torch.sparse_coo_tensor(
        edge, weight, (nv, nv)).unsqueeze(0)
    
    # normalized degree matrix, i.e., identity matrix
    # set the diagonal entries to one
    self_edge = torch.arange(nv)[None].repeat([2,1]).to(face.device)
    D_hat = torch.sparse_coo_tensor(
        self_edge, torch.ones_like(self_edge[0]).float(), (nv, nv)).unsqueeze(0)
    L = D_hat - A_hat
    return L


def laplacian_smooth(vert, face, lambd=1., n_iters=1):
    """
    Laplacian mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - lambd * L.bmm(vert)
    return vert


def taubin_smooth(vert, face, lambd=0.5, mu=-0.53, n_iters=1):
    """
    Taubin mesh smoothing.
    
    Inputs:
    - vert: input mesh vertices, (1,|V|,3) torch.Tensor
    - face: input mesh faces, (1,|F|,3) torch.LongTensor
    - lambd: strength of mesh smoothing [0,1]
    - mu: strength of mesh smoothing [-1,0]
    - n_iters: number of mesh smoothing iterations
    
    Returns:
    - vert: smoothed mesh vertices, (1,|V|,3) torch.Tensor
    """
    L = laplacian(face)
    for n in range(n_iters):
        vert = vert - mu * L.bmm(vert)
        vert = vert - lambd * L.bmm(vert)
    return vert


def curvature(vert, face, L):
    normal = vert_normal(vert, face)
    # \Laplacian x = -2Hn
    curv = (L.bmm(vert) * normal).sum(-1) / 2
    return curv


def connected_component_filter(seg):
    cc, n_cc = connected(seg, return_num=True)
    # compute the unique values and their counts
    uni, n_uni= np.unique(cc, return_counts=True)
    # remove the background
    uni = uni[1:]
    n_uni = n_uni[1:]
    # preserve the largest connected component apart from the background
    idx = np.argmax(n_uni)
    # only keep the largest connected component
    seg_cc = (cc==uni[idx]).astype(np.float32)

    # also check the reverse image
    cc, n_cc = connected(1-seg_cc, return_num=True)
    # compute the unique values and their counts
    uni, n_uni= np.unique(cc, return_counts=True)
    # remove the background
    uni = uni[1:]
    n_uni = n_uni[1:]
    # preserve the largest connected component apart from the background
    idx = np.argmax(n_uni)
    # only keep the largest connected component
    seg_cc = 1 - (cc==uni[idx]).astype(np.float32)
    return seg_cc


def sample_mesh_points(vert, face, attr=None, n_sample=10000):
    # (1,F,3)
    vert_0 = vert[:,face[0,:,0]]
    vert_1 = vert[:,face[0,:,1]]
    vert_2 = vert[:,face[0,:,2]]
    
    if attr is not None:  # (1,F,d)
        attr_0 = attr[:,face[0,:,0]]
        attr_1 = attr[:,face[0,:,1]]
        attr_2 = attr[:,face[0,:,2]]
        
    # compute face area
    f_area = 0.5*torch.norm(torch.cross(vert_1 - vert_0, vert_2 - vert_0), dim=-1)[0]
    # sample faces based on the face area
    f_sample = torch.multinomial(
        f_area / f_area.sum(), n_sample, replacement=True)

    # (1,n_sample,3)
    vert_0 = vert_0[:,f_sample].clone()
    vert_1 = vert_1[:,f_sample].clone()
    vert_2 = vert_2[:,f_sample].clone()
    
    if attr is not None:  # (1,n_sample,d)
        attr_0 = attr_0[:,f_sample].clone()
        attr_1 = attr_1[:,f_sample].clone()
        attr_2 = attr_2[:,f_sample].clone()

    # randomly sample barycentric coordinates
    x, y = torch.rand(2, n_sample, device=vert.device)
    x_sqrt = x.sqrt()
    w0 = 1.0 - x_sqrt
    w1 = x_sqrt * (1.0 - y)
    w2 = x_sqrt * y
    barycenter = torch.stack([w0, w1, w2], dim=1)[None]

    # compute sampled points
    point_sample = barycenter[:,:,0:1]*vert_0 + barycenter[:,:,1:2]*vert_1 + barycenter[:,:,2:3]*vert_2
    if attr is not None:
        attr_sample  = barycenter[:,:,0:1]*attr_0 + barycenter[:,:,1:2]*attr_1 + barycenter[:,:,2:3]*attr_2
        return point_sample, attr_sample  #, f_sample, barycenter
    else:
        return point_sample  #, f_sample, barycenter
    