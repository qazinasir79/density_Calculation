import numpy as np

R = 8.314462618

# Ethane properties
TC = 305.32
PC = 4.872e6
omega = 0.0995
M = 30.069e-3

def _solve_cubic(coeffs):
    roots = np.roots(coeffs)
    real = np.real(roots[np.abs(np.imag(roots)) < 1e-8])
    real = real[real > 1e-6]
    real.sort()
    return real

def _Psat_antoine(T):
    if T >= TC:
        return None
    return 10 ** (6.06426 - 789.22 / (T + 247.86)) * 0.1


def density_PR(T_K, P_MPa, phase='auto'):
    P = P_MPa * 1e6
    Tr = T_K / TC
    a = 0.45724 * R**2 * TC**2 / PC
    b = 0.07780 * R * TC / PC
    kappa = 0.37464 + 1.54226 * omega - 0.26992 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
    A = a * alpha * P / (R**2 * T_K**2)
    B = b * P / (R * T_K)
    coeffs = [1, -(1 - B), A - 2*B - 3*B**2, -(A*B - B**2 - B**3)]
    roots = _solve_cubic(coeffs)
    if len(roots) == 0:
        return np.nan
    if len(roots) == 1:
        Z = roots[0]
    else:
        if phase == 'liquid':
            Z = roots[0]
        elif phase == 'vapor':
            Z = roots[-1]
        else:
            if T_K < TC:
                Psat = _Psat_antoine(T_K)
                if Psat is not None:
                    if P_MPa >= Psat:
                        Z = roots[0]
                    else:
                        Z = roots[-1]
                else:
                    Z = roots[0] if P_MPa > 3 else roots[-1]
            else:
                Z = roots[-1]
    V = Z * R * T_K / P
    return M / V


def density_SRK(T_K, P_MPa, phase='auto'):
    P = P_MPa * 1e6
    Tr = T_K / TC
    a = 0.42747 * R**2 * TC**2 / PC
    b = 0.08664 * R * TC / PC
    kappa = 0.480 + 1.574 * omega - 0.176 * omega**2
    alpha = (1 + kappa * (1 - np.sqrt(Tr)))**2
    A = a * alpha * P / (R**2 * T_K**2)
    B = b * P / (R * T_K)
    coeffs = [1, -1, A - B - B**2, -A*B]
    roots = _solve_cubic(coeffs)
    if len(roots) == 0:
        return np.nan
    if len(roots) == 1:
        Z = roots[0]
    else:
        if phase == 'liquid':
            Z = roots[0]
        elif phase == 'vapor':
            Z = roots[-1]
        else:
            if T_K < TC:
                Psat = _Psat_antoine(T_K)
                if Psat is not None:
                    Z = roots[0] if P_MPa >= Psat else roots[-1]
                else:
                    Z = roots[0] if P_MPa > 3 else roots[-1]
            else:
                Z = roots[-1]
    V = Z * R * T_K / P
    return M / V
