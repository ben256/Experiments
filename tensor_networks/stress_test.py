import argparse
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm

from tensor_networks.heston_tt import build_heston_tt, price_from_tt
from tensor_networks.utils.heston_fft import heston_pricer_fft


def perform_stress_test_on_parameter(
        parameter_name: str,
        parameter_limits: tuple,
        n_samples: int = 10,
        S: float = 100.0,
        K: float = 100.0,
        T: float = 1.0,
        v0: float = 0.02,
        kappa: float = 1.0,
        theta: float = 0.02,
        rho: float = -0.7,
        sigma_v: float = 0.1,
        rate: float = 0.05,
        div: float = 0.0
):

    test_cases = {
        'S': [S] * n_samples,
        'K': [K] * n_samples,
        'T': [T] * n_samples,
        'v0': [v0] * n_samples,
        'kappa': [kappa] * n_samples,
        'theta': [theta] * n_samples,
        'rho': [rho] * n_samples,
        'sigma_v': [sigma_v] * n_samples,
        'rate': [rate] * n_samples,
        'div': [div] * n_samples
    }
    test_cases[parameter_name] = np.linspace(parameter_limits[0], parameter_limits[1], n_samples)

    ranks = [2, 3, 4, 5, 6, 7]

    errors = np.zeros((n_samples, len(ranks)))

    for i in tqdm(range(n_samples)):
        params = {key: test_cases[key][i] for key in test_cases}
        analytic_prices = heston_pricer_fft(**params)

        for j, r in enumerate(ranks):
            tt_heston, info = build_heston_tt(
                base=10,
                basis_size=3,
                tt_tolerance=1e-6,
                tt_rmax=r,
                tt_max_iter=50,
                **params
            )
            tt_prices = price_from_tt(params['K'], params['T'], tt_heston, info)
            errors[i, j] = np.sqrt(np.mean((tt_prices - analytic_prices) ** 2))

    data_for_boxplot = [errors[:, j] for j in range(len(ranks))]

    fig, ax = plt.subplots(figsize=(10, 7), tight_layout=True)
    ax.boxplot(data_for_boxplot, tick_labels=ranks, showfliers=True)
    ax.set_xlabel('TT Rank')
    ax.set_ylabel('RMS Pricing Error')
    ax.set_title(f'RMS Pricing Error vs TT Rank ($\\{parameter_name}$ stress)')
    ax.set_yscale('log')
    ax.grid(True, which='both', alpha=0.3)

    try:
        plt.savefig(f'./output/plots/stress_test_{parameter_name}.png', dpi=300)
    except Exception:
        print("Could not save the plot. Ensure the output directory exists.")

    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("parameter_name", type=str)
    parser.add_argument("parameter_limits", type=float, nargs=2)
    parser.add_argument("--n_samples", type=int, default=10)
    parser.add_argument("--S", type=float, default=100.0)
    parser.add_argument("--K", type=float, default=100.0)
    parser.add_argument("--T", type=float, default=1.0)
    parser.add_argument("--v0", type=float, default=0.02)
    parser.add_argument("--kappa", type=float, default=1.0)
    parser.add_argument("--theta", type=float, default=0.02)
    parser.add_argument("--rho", type=float, default=-0.7)
    parser.add_argument("--sigma_v", type=float, default=0.1)
    parser.add_argument("--rate", type=float, default=0.05)
    parser.add_argument("--div", type=float, default=0.0)

    args = parser.parse_args()

    perform_stress_test_on_parameter(
        parameter_name=args.parameter_name,
        parameter_limits=tuple(args.parameter_limits),
        n_samples=args.n_samples,
        S=args.S,
        K=args.K,
        T=args.T,
        v0=args.v0,
        kappa=args.kappa,
        theta=args.theta,
        rho=args.rho,
        sigma_v=args.sigma_v,
        rate=args.rate,
        div=args.div
    )
