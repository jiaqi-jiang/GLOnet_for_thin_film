import torch
import torch.nn as nn
import torch.nn.functional as F



class Generator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dim
        self.thickness_sup = params.thickness_sup
        self.N_layers = params.N_layers
        self.M_materials = params.M_materials
        self.n_database = params.n_database.view(1, 1, params.M_materials, -1).cuda() # 1 x 1 x number of mat x number of freq
        
        self.FC = nn.Sequential(
            nn.Linear(self.noise_dim, 64),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2),
            nn.Linear(64, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 64, bias=False),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, self.N_layers * (self.M_materials + 1), bias=False),
            nn.BatchNorm1d(self.N_layers * (self.M_materials + 1)),
        )

        self.FC_thickness = nn.Sequential(
            nn.Linear(self.N_layers, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, self.N_layers),
        )

    def forward(self, noise, alpha):
        net = self.FC(noise)
        net = net.view(-1, self.N_layers, self.M_materials + 1)
        
        thicknesses = torch.sigmoid(self.FC_thickness(net[:, :, 0])) * self.thickness_sup
        X = net[:, :, 1:]
        
        P = F.softmax(X * alpha, dim = 2).unsqueeze(-1) # batch size x number of layer x number of mat x 1
        refractive_indices = torch.sum(P * self.n_database, dim=2) # batch size x number of layer x number of freq
        
        return (thicknesses, refractive_indices, P.squeeze())

class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, dim=16):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(dim, dim*2, bias=False),
                nn.BatchNorm1d(dim*2),
                nn.LeakyReLU(0.2),
                nn.Linear(dim*2, dim, bias=False),
                nn.BatchNorm1d(dim))

    def forward(self, x):
        return F.leaky_relu(self.block(x) + x, 0.2)

'''
class ResBlock(nn.Module):
    """docstring for ResBlock"""
    def __init__(self, dim=64):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
                nn.Linear(dim, dim, bias=False),
                nn.BatchNorm1d(dim),
                nn.LeakyReLU(0.2))

    def forward(self, x):
        return x + self.block(x)
'''

class ResGenerator(nn.Module):
    def __init__(self, params):
        super().__init__()

        self.noise_dim = params.noise_dim
        self.res_layers = params.res_layers
        self.res_dim = params.res_dim
        self.thickness_sup = params.thickness_sup
        self.N_layers = params.N_layers
        self.M_materials = params.M_materials
        self.n_database = params.n_database.view(1, 1, params.M_materials, -1).cuda() # 1 x 1 x number of mat x number of freq
        
        self.initBLOCK = nn.Sequential(
            nn.Linear(self.noise_dim, self.res_dim),
            nn.LeakyReLU(0.2),
            nn.Dropout(p=0.2)
        )

        self.endBLOCK = nn.Sequential(
            nn.Linear(self.res_dim, self.N_layers * (self.M_materials + 1), bias=False),
            nn.BatchNorm1d(self.N_layers * (self.M_materials + 1)),
        )

        self.ResBLOCK = nn.ModuleList()
        for i in range(params.res_layers):
            self.ResBLOCK.append(ResBlock(self.res_dim))

        self.FC_thickness = nn.Sequential(
            nn.Linear(self.N_layers, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            nn.Linear(16, self.N_layers),
        )

    def forward(self, noise, alpha):
        net = self.initBLOCK(noise)
        for i in range(self.res_layers):
            self.ResBLOCK[i](net)
        net = self.endBLOCK(net)

        net = net.view(-1, self.N_layers, self.M_materials + 1)
        
        thicknesses = torch.sigmoid(self.FC_thickness(net[:, :, 0])) * self.thickness_sup
        X = net[:, :, 1:]
        
        P = F.softmax(X * alpha, dim = 2).unsqueeze(-1) # batch size x number of layer x number of mat x 1
        refractive_indices = torch.sum(P * self.n_database, dim=2) # batch size x number of layer x number of freq
        
        return (thicknesses, refractive_indices, P.squeeze())


class Planar_flow(nn.Module):
    def __init__(self, dim):
        super().__init__() 
        self.u = nn.Parameter(torch.randn(1, dim)*0.01)
        self.w = nn.Parameter(torch.randn(1, dim)*0.01)
        self.b = nn.Parameter(torch.randn(())*0.01)

    def m(self, x):
        return -1 + torch.log(1 + torch.exp(x))

    def u_hat_cal(self):
        w_dot_u = torch.mm(self.u, self.w.t()).view(())
        u_hat = self.u + (self.m(w_dot_u) - w_dot_u) * self.w/torch.pow(torch.norm(self.w),1)
        return u_hat

    def forward(self, z):
        u = self.u_hat_cal()
        affine = torch.mm(z, self.w.t()) + self.b
        x = z + u * torch.tanh(affine)

        psi = (1 - torch.pow(torch.tanh(affine), 2)) * self.w
        det_grad = 1 + torch.mm(psi, u.t())
        LDJ = torch.log(det_grad.abs() + 1e-8)
        return x, LDJ

class GeneratorNF(nn.Module):
    def __init__(self, params):
        super().__init__()
    
        self.N_layers = params.N_layers
        self.thickness_sup = params.thickness_sup
        self.M_materials = params.M_materials
        self.n_database = params.n_database.view(1, 1, params.M_materials, -1).cuda() # 1 x 1 x number of mat x number of freq
        self.dim = self.N_layers * (self.M_materials + 1)
        self.transforms = nn.ModuleList([Planar_flow(self.dim) for _ in range(params.NF_layers)])
        self.MAP = nn.Linear(params.noise_dim,  self.dim)

        self.FC_thickness = nn.Sequential(
            nn.Linear(self.N_layers, 16),
            nn.LeakyReLU(0.2),
            nn.BatchNorm1d(16),
            #nn.Linear(16, 16),
            #nn.LeakyReLU(0.2),
            #nn.BatchNorm1d(16),
            nn.Linear(16, self.N_layers),
        )

    def forward(self, z, alpha):
        z = self.MAP(z)
        log_jacobians = []
        for transform in self.transforms:
            z, LDJ = transform(z)
            log_jacobians.append(LDJ)
        net = z
        sum_log_jacobians = sum(log_jacobians)

        net = net.view(-1, self.N_layers, self.M_materials + 1)
        
        thicknesses = torch.sigmoid(self.FC_thickness(net[:, :, 0])) * self.thickness_sup 
        X = net[:, :, 1:]

        P = F.softmax(X * alpha, dim = 2).unsqueeze(-1) # batch size x number of layer x number of mat x 1
        refractive_indices = torch.sum(P * self.n_database, dim=2) # batch size x number of layer x number of freq
        return (thicknesses, refractive_indices, P.squeeze())