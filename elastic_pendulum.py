import numpy as np
from numpy import ndarray
from scipy.integrate import solve_ivp

# from equations_of_motion import t_span


def elastic_pendulum(t, y, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = y  # radial pos., angular pos., radial vel., angular vel.
    d2r = r * dtheta ** 2 + g * np.cos(theta) - (k / m) * (r - L0) + epsilon / r ** 2  # radial acceleration
    d2theta = -2 * dr * dtheta / r - (g / r) * np.sin(theta)  # angular acceleration
    return np.array([dr, dtheta, d2r, d2theta])  # state derivatives


def variational_equations(t, z, g, k, m, L0, epsilon):
    n = z.shape[1]  # number of trajectories
    dzdt = np.zeros_like(z)  # derivative placeholder

    for i in range(n):
        y = z[:4, i]  # original state variables
        delta = z[4:, i]  # perturbation variables
        r, theta, dr, dtheta = y  # unpack state variables

        dfdy = np.array([  # Jacobian matrix
            [0, 0, 1, 0],
            [0, 0, 0, 1],
            [dtheta ** 2 - k / m - 2 * epsilon / r ** 3, -g * np.sin(theta), 0, 2 * r * dtheta],
            [(2 * dr * dtheta + g * np.sin(theta)) / r ** 2, -(g / r) * np.cos(theta), -2 * dtheta / r, -2 * dr / r]
        ])

        dydt = elastic_pendulum(t, y, g, k, m, L0, epsilon)  # derivatives of state variables
        ddelta_dt = dfdy @ delta  # derivatives of perturbation variables

        dzdt[:, i] = np.concatenate((dydt, ddelta_dt))  # combined state and perturbation derivatives

    return dzdt


def compute_lyapunov_exponent(initial_condition: np.ndarray,
                              perturbation: np.ndarray,
                              params: tuple,
                              t_max: float = 100,
                              dt: float = 0.1) -> list:

    g, k, m, L0, epsilon = params  # system parameters
    initial_condition = initial_condition.reshape(4, -1)  # reshape initial conditions
    perturbation = perturbation.reshape(4, -1)  # reshape perturbations

    num_conditions = initial_condition.shape[1]  # number of initial conditions
    lyapunov_exponents = []  # list to store exponents

    for idx in range(num_conditions):
        y0 = np.concatenate((initial_condition[:, idx], perturbation[:, idx]))  # combined initial state and perturbation
        sol = solve_ivp(fun=variational_equations,
                        t_span=[0, t_max],
                        y0=y0,
                        method='RK45',
                        t_eval=[0, t_max],
                        rtol=1e-12,
                        atol=1e-12,
                        args=(g, k, m, L0, epsilon),
                        vectorized=True)

        delta_T = sol.y[4:, -1]  # final perturbation
        delta_0 = sol.y[4:, 0]  # initial perturbation

        exponent = np.log(np.linalg.norm(delta_T) / np.linalg.norm(delta_0)) / t_max  # Lyapunov exponent
        lyapunov_exponents.append(exponent)

    return lyapunov_exponents


if __name__ == '__main__':
    params = (9.81, 40.0, 1.0, 1.0, 0.0)  # gravitational accel., spring constant, mass, natural length, additional potential

    initial_conditions = np.array([
        [1.0, np.pi / 6, 0.0, 0.0],  # initial state 1
        [1.0, np.pi / 3, 0.0, 0.0]   # initial state 2
    ]).T

    perturbations = np.array([
        [1, 0, 0, 0],  # perturbation state 1
        [1, 0, 0, 0]   # perturbation state 2
    ]).T

    lyapunov_exp = compute_lyapunov_exponent(initial_conditions, perturbations, params)

    for idx, exp in enumerate(lyapunov_exp):
        print(f'Lyapunov exponent for initial condition {idx+1}: {exp:.6f}')
