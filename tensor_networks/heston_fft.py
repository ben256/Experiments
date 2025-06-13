import numpy as np


def HestonCf(phi, S, T, kappa, rho, volvol, theta, var0, rate, div):

    batch_size = S.size
    S_r = S.reshape(1, batch_size)
    T_r = T.reshape(1, batch_size)
    var0_r = var0.reshape(1, batch_size)
    phi_r = phi.reshape(-1, 1)

    xx = np.log(S_r)
    gamma = kappa - rho * volvol * phi_r * 1j
    zeta = -0.5 * (np.power(phi_r, 2) + 1j * phi_r)
    psi = np.power(np.power(gamma, 2) - 2 * np.power(volvol, 2) * zeta, 0.5)

    CCaux = (2 * psi - (psi - gamma) * (1 - np.exp(-psi * T_r))) / (2 * psi)
    CC = (-(kappa * theta) / np.power(volvol, 2)) * (2 * np.log(CCaux) + (psi - gamma) * T_r)
    BB = (2 * zeta * (1 - np.exp(-psi * T_r)) * var0_r) / (2 * psi - (psi - gamma) * (1 - np.exp(-psi * T_r)))
    AA = 1j * phi_r * (xx + rate * T_r)

    return np.exp(AA + BB + CC)


def heston_pricer_fft(S, K_strike, T, volvol, kappa, rho, theta, var0, rate, div):

    alpha = 1.25
    NN = 4096
    cc = 600
    eta = cc / NN
    Lambda = (2 * np.pi) / (NN * eta)
    bb = (NN * Lambda) / 2

    # Vectorized call to the characteristic function
    jj = np.arange(1, NN + 1)
    phi = eta * (jj - 1)
    NewPhi = phi - (alpha + 1) * 1j

    CF = HestonCf(NewPhi, S, T, kappa, rho, volvol, theta, var0, rate, div)
    phi_r = phi.reshape(-1, 1)

    ModCF = (np.exp(-rate * T) * CF) / (np.power(alpha, 2) + alpha - np.power(phi_r, 2) + 1j * phi_r * (2 * alpha + 1))

    # FFT execution
    delta = np.zeros(NN)
    delta[0] = 1
    Simpson = (eta / 3) * (3 + np.power(-1j, jj) - delta)
    Simpson_r = Simpson.reshape(-1, 1)

    FuncFFT = np.exp(1j * bb * phi_r) * ModCF * Simpson_r
    Payoff = np.real(np.fft.fft(FuncFFT, axis=0))

    # Interpolation to find the price for the specific strike K_strike
    ku = -bb + Lambda * (jj - 1)
    ku_r = ku.reshape(-1, 1)
    CallPrices = (np.exp(-alpha * ku_r) / np.pi) * Payoff

    interp_positions = ((np.log(K_strike) + bb) / Lambda) + 1
    prices = np.zeros(S.size)
    for i in range(S.size):
        prices[i] = np.interp(interp_positions[i], jj, CallPrices[:, i])

    return prices
