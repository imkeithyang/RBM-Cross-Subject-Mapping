import torch
import torch.nn as nn
import torch.nn.functional as func

class Recognition(nn.Module):
    def __init__(self, recog_dims, additional_input_size):
        super(Recognition,self).__init__()
        self.recog_depth = len(recog_dims)
        if self.recog_depth > 2:
            self.recog_layers = nn.ModuleList([ nn.Linear( recog_dims[i] + int(i==0)*additional_input_size, recog_dims[i+1] ) for i in range(self.recog_depth-1) if i <= self.recog_depth-3 ])
            self.recog_normalizers = nn.ModuleList([ nn.BatchNorm1d(recog_dims[i+1]) for i in range(self.recog_depth-1) if i <= self.recog_depth-3 ])
        self.recog_mu, self.recog_logvar = nn.Linear( recog_dims[-2] + int(self.recog_depth<=2)*additional_input_size, recog_dims[-1] ), nn.Linear( recog_dims[-2] + int(self.recog_depth<=2)*additional_input_size, recog_dims[-1] )

    def forward(self, X):
        for i in range(self.recog_depth):
            if i <= self.recog_depth - 3:
                X = func.relu(self.recog_normalizers[i](self.recog_layers[i](X)))
        return self.recog_mu(X), self.recog_logvar(X)
    
class Prior(nn.Module):
    def __init__(self, prior_dims):
        super(Prior,self).__init__()
        self.prior_depth = len(prior_dims)
        self.prior_layers = nn.ModuleList([ nn.Linear( prior_dims[i], prior_dims[i+1] ) for i in range(self.prior_depth-1) if i <= self.prior_depth-3 ])
        self.prior_normalizers = nn.ModuleList([ nn.BatchNorm1d(prior_dims[i+1]) for i in range(self.prior_depth-1) if i <= self.prior_depth-3 ])
        self.prior_mu, self.prior_logvar = nn.Linear( prior_dims[-2], prior_dims[-1] ), nn.Linear( prior_dims[-2], prior_dims[-1] )

    def forward(self, X):
        for i in range(self.prior_depth):
            if i <= self.prior_depth - 3:
                X = func.relu(self.prior_normalizers[i](self.prior_layers[i](X)))
        return self.prior_mu(X), self.prior_logvar(X)

class Generation(nn.Module):
    def __init__(self, gen_dims, additional_input_size):
        super(Generation,self).__init__()
        self.gen_depth = len(gen_dims)
        if self.gen_depth > 2:
            self.gen_layers = nn.ModuleList([ nn.Linear( gen_dims[i] + int(i==0)*additional_input_size, gen_dims[i+1] ) for i in range(self.gen_depth) if i <= self.gen_depth-3 ])
            self.gen_normalizers = nn.ModuleList([ nn.BatchNorm1d(gen_dims[i+1]) for i in range(self.gen_depth-1) if i <= self.gen_depth-3 ])
        self.gen_mu, self.gen_logvar = nn.Linear( gen_dims[-2] + int(self.gen_depth<=2)*additional_input_size, gen_dims[-1] ), nn.Linear( gen_dims[-2] + int(self.gen_depth<=2)*additional_input_size, gen_dims[-1] )

    def forward(self, X):
        for i in range(self.gen_depth):
            if i <= self.gen_depth - 3:
                X = func.relu(self.gen_normalizers[i](self.gen_layers[i](X)))
        return self.gen_mu(X), self.gen_logvar(X)

class ConditionalVAE(nn.Module):
    def __init__(self, recog_dims, prior_dims, gen_dims):
        super(ConditionalVAE,self).__init__()
        self.CVAE_in_dim, self.CVAE_out_dim = recog_dims[0], gen_dims[-1]
        self.recognizer = Recognition( recog_dims, self.CVAE_out_dim )
        self.prior = Prior( prior_dims )
        self.generator = Generation( gen_dims, self.CVAE_in_dim )

    def forward(self, X, Y):
        Zr_mu, Zr_logvar = self.recognizer( torch.cat((X,Y),dim=1) )
        Zp_mu, Zp_logvar = self.prior( X )
        if self.training:
            r_eps, p_eps = torch.randn_like(Zr_mu), torch.randn_like(Zp_mu)
            Zr, Zp = r_eps.mul( torch.exp(Zr_logvar/2) ).add_(Zr_mu), p_eps.mul( torch.exp(Zp_logvar/2) ).add_( Zp_mu )
            Yg_mu, Yg_logvar = self.generator( torch.cat((Zr,X),dim=1) )
            Yp_mu, Yp_logvar = self.generator( torch.cat((Zp,X),dim=1) )
            return Yg_mu, Yg_logvar, Yp_mu, Yp_logvar, Zr_mu, Zr_logvar, Zp_mu, Zp_logvar
        else:
            Y_mu, Y_logvar = self.generator( torch.cat((Zp_mu,X),dim=1) )
            return Y_mu, Y_logvar
        
##################################################################################################

class ConditionalVAE_simplified(nn.Module):
    def __init__(self, recog_dims, gen_dims):
        super(ConditionalVAE_simplified,self).__init__()
        self.CVAE_in_dim, self.CVAE_out_dim = recog_dims[0], gen_dims[-1]
        self.recognizer = Recognition( recog_dims, self.CVAE_out_dim )
        self.generator = Generation( gen_dims, self.CVAE_in_dim )

    def forward(self, X, Y, K=1):
        Z_mu, Z_logvar = self.recognizer(torch.cat((X,Y),dim=1))
        if self.training:
            eps = torch.randn_like( Z_mu )
            Z = eps.mul( torch.exp(Z_logvar/2) ).add_( Z_mu )
            return self.generator( torch.cat((Z, X), dim=1) ), Z_mu, Z_logvar
        else:
            eps = torch.randn_like( Z_mu )
            Z = eps 
            return self.generator( torch.cat((Z, X), dim=1) )
        
####################################################################################################

# GaussBernRBM here

class GaussBernRBM_withUnitVariances(nn.Module):
    def __init__(self, v_dim, h_dim, tol, device):
        super(GaussBernRBM_withUnitVariances, self).__init__()
        self.W = nn.Parameter(torch.randn(h_dim,v_dim) * tol)
        self.v_bias = nn.Parameter(torch.zeros(v_dim))
        self.h_bias = nn.Parameter(torch.zeros(h_dim))
        self.device = device
    
    def sample_Bern(self, p):
        u = torch.rand(p.size()).to(self.device)
        z = func.relu( torch.sign(p - u) )
        return z
    
    def sample_Gauss(self, mu): # the covariance matrix is identity
        e = torch.randn_like(mu)
        z = mu + e
        return z
    
    def pdf_h_given_v(self, v):
        phv = torch.sigmoid(func.linear(v, self.W, self.h_bias))
        return phv
    
    def pdf_v_given_h(self, h):
        mu = func.linear(h, self.W.t(), self.v_bias)
        return mu

    def free_energy(self, v):
        visible_term = v.mv(self.v_bias) - 0.5*v.pow(2).sum(dim=1)
        wv_b = func.linear(v, self.W, self.h_bias)
        hidden_term = func.softplus(wv_b).sum(dim=1)
        return (- hidden_term - visible_term).mean()
    
    def fisher_score(self, v):
        sigmau, dsigmau_du = torch.sigmoid(func.linear(v, self.W, self.h_bias)), torch.sigmoid(func.linear(v, self.W, self.h_bias))*(1-torch.sigmoid(func.linear(v, self.W, self.h_bias)))
        S1 = 0.5 * func.mse_loss( v, func.linear( sigmau, self.W.t(), self.v_bias ), reduction = 'sum' )
        S2 = torch.diagonal( torch.matmul( torch.transpose( torch.matmul( torch.diag_embed(dsigmau_du), self.W ), 1,2 ), self.W ), dim1=-2, dim2=-1 ).sum();
        return (S1 + S2)/(dsigmau_du.size(0))
    
    def forward(self, v, k):
        vh_k = v
        for _ in range(k):
            phv_k = self.pdf_h_given_v(vh_k)
            hv_k = self.sample_Bern(phv_k)
            mu_k = self.pdf_v_given_h(hv_k)
            vh_k = self.sample_Gauss(mu_k)
        return vh_k, mu_k
    
####################################################################################################

# GaussBernRBM here

class GaussBernRBM_withNonUnitVariances(nn.Module):
    def __init__(self, v_dim, h_dim, tol, device):
        super(GaussBernRBM_withNonUnitVariances, self).__init__()
        self.W = nn.Parameter(torch.randn(h_dim,v_dim) * tol)
        self.v_bias = nn.Parameter(torch.randn(v_dim) * tol)
        self.h_bias = nn.Parameter(torch.randn(h_dim) * tol)
        self.z = nn.Parameter(torch.randn(v_dim) * tol)
        self.device = device
    
    def sample_Bern(self, p):
        u = torch.rand(p.size()).to(self.device)
        z = func.relu( torch.sign(p - u) )
        return z
    
    def sample_Gauss(self, mu, std): # the covariance matrix is identity
        e = torch.randn_like(mu)
        z = e.mul( std ).add_( mu )
        return z
    
    def pdf_h_given_v(self, v):
        phv = torch.sigmoid(func.linear(torch.mul(v,torch.exp(-self.z)), self.W, self.h_bias))
        return phv
    
    def pdf_v_given_h(self, h):
        mu = func.linear(h, self.W.t(), self.v_bias)
        std = torch.exp(self.z/2)
        return mu, std

    def free_energy(self, v):
        Sv = torch.mul(v,torch.exp(-self.z))
        visible_term = Sv.mv(self.v_bias) - 0.5*torch.mul(v,Sv).sum(dim=1)
        wv_b = func.linear(Sv, self.W, self.h_bias)
        hidden_term = func.softplus(wv_b).sum(dim=1)
        return (- hidden_term - visible_term).mean()
    
    def fisher_score(self, v):
        Sv = torch.mul(v,torch.exp(-self.z))
        sigmau, dsigmau_du = torch.sigmoid(func.linear(Sv, self.W, self.h_bias)), torch.sigmoid(func.linear(Sv, self.W, self.h_bias))*(1-torch.sigmoid(func.linear(Sv, self.W, self.h_bias)))
        S1 = 0.5 * func.mse_loss( Sv, torch.mul(func.linear( sigmau, self.W.t(), self.v_bias ),torch.exp(-self.z)), reduction = 'sum' )
        WS = torch.mul(self.W,torch.exp(-self.z))
        S2 = torch.diagonal( torch.matmul( torch.transpose( torch.matmul( torch.diag_embed(dsigmau_du), WS ), 1,2 ), WS ), dim1=-2, dim2=-1 ).sum() - torch.exp(-self.z).sum();
        return (S1 + S2)/(dsigmau_du.size(0))
    
    def forward(self, v, k):
        vh_k = v
        for _ in range(k):
            phv_k = self.pdf_h_given_v(vh_k)
            hv_k = self.sample_Bern(phv_k)
            mu_k, std_k = self.pdf_v_given_h(hv_k)
            vh_k = self.sample_Gauss(mu_k, std_k)
        return vh_k, mu_k

####################################################################################################

class MultiLayerPerceptron(nn.Module):
    def __init__(self, mlp_dims): # required len(arr) > 2
        super(MultiLayerPerceptron, self).__init__()
        self.mlp_depth = len(mlp_dims)
        self.mlp_layers = nn.ModuleList([ nn.Linear(mlp_dims[i], mlp_dims[i+1]) for i in range(self.mlp_depth - 1) ])
        self.mlp_normalizers = nn.ModuleList([ nn.BatchNorm1d( mlp_dims[i+1] ) for i in range(self.mlp_depth-1) if i <= self.mlp_depth-1 ])
    
    def forward(self, X):
        for i in range(self.mlp_depth-1):
            if i < self.mlp_depth - 1:
                X = func.relu(self.mlp_normalizers[i](self.mlp_layers[i](X)))
            else:
                X = func.softmax(self.mlp_normalizers[i](self.mlp_layers[i](X)), 1)
        return X
    
####################################################################################################


    

    
