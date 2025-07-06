import tntorch as tn
import torch
from scipy.stats import norm, multivariate_normal
from torch.distributions.multivariate_normal import MultivariateNormal
import numpy as np



min = -5
max = 5
basis_size = 3
base = 10
coefs = [base**i for i in range(basis_size)]
delta_z = (max - min) / (np.sum(coefs) * (base - 1))

N = 6
m = MultivariateNormal(torch.zeros(N), torch.eye(N))

def multivariate_normal_wrapper(*args):
    z = torch.stack(args)
    z = torch.reshape(z, (N, basis_size, -1))
    z = min + torch.tensordot(z, torch.tensor(coefs, dtype=z.dtype), dims=([1],[0])) * delta_z

    test = torch.exp(m.log_prob(torch.transpose(z, 0, 1)))

    return test

domain = [torch.arange(0, base) for _ in range(N * len(coefs))]
t = tn.cross(function=multivariate_normal_wrapper, domain=domain, eps=1e-10, rmax=100, max_iter=2)  # ranks_tt=[10] * (basis_size * N -  1))
output = t.sum() * delta_z**N
print("Output:", output.item())
