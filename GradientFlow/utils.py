import numpy as np
import torch
import ot

def compute_true_Wasserstein_joint(X,Y,p=2,alpha=0.5):
    X1,X2 = X[:,:-3],X[:,-3:]
    Y1, Y2 = Y[:, :-3], Y[:, -3:]
    M1= ot.dist(X1.cpu().detach().numpy(), Y1.cpu().detach().numpy())
    M2 = np.sqrt(ot.dist(X2.cpu().detach().numpy(), Y2.cpu().detach().numpy()))
    M2=(2*np.arcsin(M2/2))**2
    M=alpha*M1+(1-alpha)*M2
    a = np.ones((X.shape[0],)) / X.shape[0]
    b = np.ones((Y.shape[0],)) / Y.shape[0]
    return ot.emd2(a, b, M)
def compute_Wasserstein(M,device='cpu',e=0):
    if(e==0):
        pi = ot.emd([],[],M.cpu().detach().numpy()).astype('float32')
    else:
        pi = ot.sinkhorn([], [], M.cpu().detach().numpy(),reg=e).astype('float32')
    pi = torch.from_numpy(pi).to(device)
    return torch.sum(pi*M)

def projection(U, V):
    return torch.sum(V * U,dim=1,keepdim=True)* U / torch.sum(U * U,dim=1,keepdim=True)
def rand_projections(dim, num_projections=1000,device='cpu'):
    projections = torch.randn((num_projections, dim),device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=1, keepdim=True))
    return projections


def one_dimensional_Wasserstein_prod(X,Y,theta,p):
    X_prod = torch.matmul(X, theta.transpose(0, 1))
    Y_prod = torch.matmul(Y, theta.transpose(0, 1))
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance
def one_dimensional_Wasserstein_cir_prod(X,Y,theta,p,r=1):
    X_prod = torch.cdist(X,theta*r)
    Y_prod = torch.cdist(Y,theta*r)
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    wasserstein_distance = torch.mean(torch.pow(wasserstein_distance, p), dim=0,keepdim=True)
    return wasserstein_distance

def SW(X, Y, L=10, p=2, device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_prod(X,Y,theta,p=p).mean()
    return  torch.pow(sw,1./p)

def CSW(X, Y, L=10, p=2, r=2,device="cpu"):
    dim = X.size(1)
    theta = rand_projections(dim, L,device)
    sw=one_dimensional_Wasserstein_cir_prod(X,Y,theta,p=p,r=r).mean()
    return  torch.pow(sw,1./p)

def JSW(X,Y,L=10,p=2,r=1,f_type='circular',device="cpu"):
    X1, X2 = X[:, :-3], X[:, -3:]
    Y1, Y2 = Y[:, :-3], Y[:, -3:]
    n=X.shape[0]
    dim1=X1.shape[1]
    dim2=X2.shape[1]
    theta1 = rand_projections(dim1, L, device)
    X1_prod = torch.matmul(X1, theta1.transpose(0, 1))
    Y1_prod = torch.matmul(Y1, theta1.transpose(0, 1))
    if(f_type=='circular'):
        theta2 = rand_projections(dim2, L, device)
        X2_prod = torch.cdist(X2, theta2 * r)
        Y2_prod = torch.cdist(Y2, theta2 * r)
    Xjoint = torch.stack([X1_prod,X2_prod],dim=2)
    Yjoint = torch.stack([Y1_prod, Y2_prod], dim=2)
    theta = rand_projections(2, L, device)
    X_prod = torch.sum(Xjoint* theta,dim=-1)
    Y_prod = torch.sum(Yjoint* theta,dim=-1)
    # print(X_prod.shape)
    X_prod = X_prod.view(X_prod.shape[0], -1)
    Y_prod = Y_prod.view(Y_prod.shape[0], -1)
    wasserstein_distance = torch.abs(
        (
                torch.sort(X_prod, dim=0)[0]
                - torch.sort(Y_prod, dim=0)[0]
        )
    )
    sw = torch.mean(torch.pow(wasserstein_distance, p), dim=0, keepdim=True)
    return torch.pow(sw.mean(), 1. / p)
