import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp


def elastic_pendulum(t, y, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = y
    d2r = r * dtheta ** 2 + g * np.cos(theta) - (k / m) * (r - L0) + epsilon / r ** 2
    d2theta = -2 * dr * dtheta / r - (g / r) * np.sin(theta)
    return np.array([dr, dtheta, d2r, d2theta])


def variational_equations(t, z, g, k, m, L0, epsilon):
    n = z.shape[1]
    dzdt = np.zeros_like(z)

    for i in range(n):
        y = z[:4, i]
        delta = z[4:, i]
        r, theta, dr, dtheta = y

        dfdy = np.array([
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [dtheta ** 2 - k / m - 2 * epsilon / r ** 3, -g * np.sin(theta), 0, 2 * r * dtheta],
            [(2 * dr * dtheta - g * np.sin(theta)) / r ** 2, -(g / r) * np.cos(theta), -2 * dtheta / r, -2 * dr / r]
        ])

        dydt = elastic_pendulum(t, y, g, k, m, L0, epsilon)
        ddelta_dt = dfdy @ delta

        dzdt[:, i] = np.concatenate((dydt, ddelta_dt))

    return dzdt


def compute_lyapunov_exponent(initial_condition: np.ndarray,
                              perturbation: np.ndarray,
                              params: tuple,
                              t_max: float = 100,
                              dt: float = 0.1) -> np.ndarray:

    g, k, m, L0, epsilon = params
    initial_condition = initial_condition.reshape(4, -1)
    perturbation = perturbation.reshape(4, -1)

    num_conditions = initial_condition.shape[1]
    lyapunov_exponents = np.zeros(num_conditions)

    for idx in range(num_conditions):
        y0 = np.concatenate((initial_condition[:, idx], perturbation[:, idx]))
        sol = solve_ivp(variational_equations, [0, t_max], y0, method='RK45', t_eval=[0, t_max],
                        rtol=1e-12, atol=1e-12, args=(g, k, m, L0, epsilon), vectorized=True)

        delta_T = sol.y[4:, -1]
        delta_0 = sol.y[4:, 0]

        exponent = np.log(np.linalg.norm(delta_T) / np.linalg.norm(delta_0)) / t_max
        lyapunov_exponents[idx] = exponent

    return lyapunov_exponents


if __name__ == '__main__':
    params = (9.81, 40.0, 1.0, 1.0, 0.0)

    initial_conditions = np.array([
        [1.0, np.pi / 6, 0.0, 0.0],
        [1.0, np.pi / 3, 0.0, 0.0]
    ]).T

    perturbations = np.array([
        [1, 0, 0, 0],
        [1, 0, 0, 0]
    ]).T

    lyapunov_exp = compute_lyapunov_exponent(initial_conditions, perturbations, params)

    for idx, exp in enumerate(lyapunov_exp):
        print(f'Lyapunov exponent for initial condition {idx+1}: {exp:.6f}')
