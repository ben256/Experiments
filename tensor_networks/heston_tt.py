import numpy as np
import torch
import tntorch as tn

from tensor_networks.heston_fft import heston_pricer_fft


def price_from_tt(S, K, T, v0, tt_tensor, param_deltas, base, basis_size):

    prices = np.zeros(S.size)

    for index, _ in enumerate(prices):

        i_S = int(round((S - S_min) / param_deltas[0]))
        i_K = int(round((K - K_min) / param_deltas[1]))
        i_T = int(round((T - T_min) / param_deltas[2]))
        i_v0 = int(round((v0 - v0_min) / param_deltas[3]))

        num_points = base ** basis_size
        i_S = max(0, min(i_S, num_points - 1))
        i_K = max(0, min(i_K, num_points - 1))
        i_T = max(0, min(i_T, num_points - 1))
        i_v0 = max(0, min(i_v0, num_points - 1))

        grid_indices = [i_S, i_K, i_T, i_v0]
        tt_indices = []

        for grid_index in grid_indices:
            temp_index = grid_index
            digits_for_index = []
            for _ in range(basis_size):
                digit = temp_index % base
                digits_for_index.append(digit)
                temp_index //= base
            tt_indices.extend(digits_for_index[::-1])

        prices[index] =  tt_tensor[tuple(tt_indices)].item()

    return prices


# Variable Parameters
S_min, S_max = 80.0, 120.0
K_min, K_max = 85.0, 115.0
T_min, T_max = 0.0, 1.0
v0_min, v0_max = 0.01, 0.04
variable_params = {
    'S': [S_min, S_max],
    'K': [K_min, K_max],
    'T': [T_min, T_max],
    'v0': [v0_min, v0_max],
}

# Fixed Parameters
kappa = 1.0
theta = 0.02
rho = -0.7
sigma_v = 0.1
r = 0.05

# Domain Setup
base = 10
basis_size = 2
N = len(variable_params)

domain = [torch.arange(0, base) for _ in range(N * basis_size)]
coefficients = [base ** i for i in range(basis_size)]
param_deltas = [(v[1] - v[0]) / (np.sum(coefficients) * (base - 1)) for v in variable_params.values()]


def heston_wrapper(*args):
    z = torch.stack(args)
    z = torch.reshape(z, (N, basis_size, -1))

    indices = torch.tensordot(z, torch.tensor(coefficients, dtype=z.dtype), dims=([1], [0]))

    S_np = (S_min + indices[0] * param_deltas[0]).numpy()
    K_np = (K_min + indices[1] * param_deltas[1]).numpy()
    T_np = (T_min + indices[2] * param_deltas[2]).numpy()
    v0_np = (v0_min + indices[3] * param_deltas[3]).numpy()

    prices_np = heston_pricer_fft(S_np, K_np, T_np, sigma_v, kappa, rho, theta, v0_np, r, 0.0)

    return torch.from_numpy(prices_np)


t_bs = tn.cross(
    function=heston_wrapper,
    domain=domain,
    eps=1e-4,
    rmax=30,
    max_iter=20
)
print(t_bs.ranks_tt)

S_test, K_test, T_test, v0_test = np.array([105.0]), np.array([100.0]), np.array([0.5]), np.array([0.04])
tt_price = price_from_tt(S_test, K_test, T_test, v0_test, t_bs, param_deltas)
analytical_price = heston_pricer_fft(S_test, K_test, T_test, sigma_v, kappa, rho, theta, v0_test, r, 0.0)
print(f"TT Price: {tt_price}")
print(f"Analytical Price: {analytical_price}")
print(f"Error: {abs(tt_price - analytical_price):.2e}")