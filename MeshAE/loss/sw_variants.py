import os.path as osp
import sys

import torch
import torch.nn as nn
from torch.autograd import Variable


sys.path.append(osp.dirname(osp.dirname(osp.abspath(__file__))))


def minibatch_rand_projections(batchsize, dim, num_projections=1000, device='cuda', **kwargs):
    projections = torch.randn((batchsize, num_projections, dim), device=device)
    projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=2, keepdim=True))
    return projections


def proj_onto_unit_sphere(vectors, dim=2):
    """
    input: vectors: [batchsize, num_projs, dim]
    """
    return vectors / torch.sqrt(torch.sum(vectors ** 2, dim=dim, keepdim=True))


def _sample_minibatch_orthogonal_projections(batch_size, dim, num_projections, device='cuda'):
    projections = torch.zeros((batch_size, num_projections, dim), device=device)
    projections = torch.stack([torch.nn.init.orthogonal_(projections[i]) for i in range(projections.shape[0])], dim=0)
    return projections


def compute_practical_moments_sw(x, y, num_projections=30, device="cuda", degree=2.0, p_type='linear',r=1,**kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    dim = x.size(2)
    batch_size = x.size(0)
    projections = minibatch_rand_projections(batch_size, dim, num_projections, device=device)
    # projs.shape: [batchsize, num_projs, dim]
    if(p_type=='linear'):
        xproj = x.bmm(projections.transpose(1, 2))

        yproj = y.bmm(projections.transpose(1, 2))
    elif(p_type=='circular'):
        xproj= torch.cdist(x, projections * r, p=2)
        yproj = torch.cdist(y, projections * r, p=2)

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.mean(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1), 1. / degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment

def compute_practical_moments_jsw(x, y, num_projections=30, device="cuda", degree=2.0, p_type='linear',r=1,**kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    dim = x.size(2)
    batch_size = x.size(0)
    projections1 = minibatch_rand_projections(batch_size, 3, num_projections, device=device)
    projections2 = minibatch_rand_projections(batch_size, 3, num_projections, device=device)
    projections3 = minibatch_rand_projections(batch_size, 2, num_projections, device=device)
    x1, x2 = torch.split(x, 3, dim=2)
    y1, y2 = torch.split(y, 3, dim=2)
    p1, p2= torch.split(projections3, 1, dim=2)
    # projs.shape: [batchsize, num_projs, dim]
    xproj1 = x1.bmm(projections1.transpose(1, 2))
    yproj1 = y1.bmm(projections1.transpose(1, 2))
    xproj2 = torch.cdist(x2, projections2 * r, p=2)
    yproj2 = torch.cdist(y2, projections2 * r, p=2)
    xproj= xproj1.transpose(1, 2)*p1 + xproj2.transpose(1, 2)*p2
    yproj = yproj1.transpose(1, 2) * p1 + yproj2.transpose(1, 2) * p2

    _sort = (torch.sort(xproj)[0] - torch.sort(yproj)[0])

    _sort_pow_p_get_sum = torch.mean(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1), 1. / degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment
def compute_practical_moments_sw_with_predefined_projections(x, y, projections, device="cuda", degree=2.0,p_type = 'linear',r=1, **kwargs):
    """
    x, y: [batch size, num points, dim]
    projections: [batch size, num projs, dim]
    """
    if (p_type == 'linear'):
        xproj = x.bmm(projections.transpose(1, 2))

        yproj = y.bmm(projections.transpose(1, 2))
    elif (p_type == 'circular'):
        xproj = torch.cdist(x, projections * r, p=2)
        yproj = torch.cdist(y, projections * r, p=2)

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.mean(torch.pow(torch.abs(_sort), degree), dim=2)

    first_moment = torch.pow(_sort_pow_p_get_sum.mean(dim=1), 1. / degree)
    second_moment = _sort_pow_p_get_sum.pow(2).mean(dim=1)

    return first_moment, second_moment



class SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device

    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs, device=self.device)
        return {"loss": x.shape[1]*squared_sw_2.mean(dim=0)}
class GSWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs, r,p_type,device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        self.r = r
        self.p_type = p_type
    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        squared_sw_2, _ = compute_practical_moments_sw(x, y, num_projections=self.num_projs, device=self.device,r=self.r,p_type=self.p_type)
        return {"loss": x.shape[1]*squared_sw_2.mean(dim=0)}

class H2SWD(nn.Module):
    """
    Estimate SWD with fixed number of projections
    """

    def __init__(self, num_projs,r,p_type, device="cuda", **kwargs):
        super().__init__()
        self.num_projs = num_projs
        self.device = device
        self.r = r
        self.p_type = p_type
    def forward(self, x, y, **kwargs):
        """
        x, y have the same shape of [batch_size, num_points_in_point_cloud, dim_of_1_point]
        """
        squared_sw_2, _ = compute_practical_moments_jsw(x, y, num_projections=self.num_projs, device=self.device,r=self.r,p_type=self.p_type)
        return {"loss": x.shape[1]*squared_sw_2.mean(dim=0)}
def compute_projected_distances(x, y, projections, degree=2.0,p_type = 'linear',r=1, **kwargs):
    """
    x, y: [batch_size, num_points, dim=3]
    num_projections: integer number
    """
    # projs.shape: [batchsize, num_projs, dim]

    if (p_type == 'linear'):
        xproj = x.bmm(projections.transpose(1, 2))

        yproj = y.bmm(projections.transpose(1, 2))
    elif (p_type == 'circular'):
        xproj = torch.cdist(x, projections * r, p=2)
        yproj = torch.cdist(y, projections * r, p=2)

    _sort = (torch.sort(xproj.transpose(1, 2))[0] - torch.sort(yproj.transpose(1, 2))[0])

    _sort_pow_p_get_sum = torch.mean(torch.pow(torch.abs(_sort), degree), dim=2)

    return _sort_pow_p_get_sum


