import numpy as np;
from numpy import ndarray;
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt


def elastic_pendulum(t, y):
    k = 40.0  # spring constant
    r, theta, dr, dtheta = y  # state variables
    d2r = r * dtheta ** 2 + g * np.cos(theta) - (k / m) * (r - L0) + epsilon / r ** 2  # radial acceleration
    d2theta = -2 * dr * dtheta / r - (g / r) * np.sin(theta)  # angular acceleration
    dY = np.array([dr, dtheta, d2r, d2theta])
    return dY


def variational_equations(t, z):
    k = 40.0  # spring constant
    m = 1.0  # mass
    y = z[:4]  # original state variables
    delta = z[4:].reshape((4,))  # perturbation variables
    r, theta, dr, dtheta = y  # unpack state variables

    dfdy = np.array([  # Jacobian matrix
        [0, 0, 1, 0],
        [0, 0, 0, 1],
        [dtheta ** 2 - k / m - 2 * epsilon / r ** 3, -g * np.sin(theta), 0, 2 * r * dtheta],
        [(2 * dr * dtheta - g * np.sin(theta)) / r ** 2, -(g / r) * np.cos(theta), -2 * dtheta / r, -2 * dr / r]
    ])

    dydt = elastic_pendulum(t, y)  # derivatives of state variables
    ddelta_dt = dfdy @ delta  # derivatives of perturbation variables

    return np.concatenate((dydt, ddelta_dt))  # combined derivatives


def compute_lyapunov_exponent(initial_conditions: np.ndarray,   # initial conditions [r, theta, dr, dtheta]
                              perturbations: np.ndarray,        # perturbations [delta_r, delta_theta, delta_dr, delta_dtheta]
                              params: tuple,                    # parameters tuple (g, k, m, L0, epsilon)
                              t_max: float = 100,               # maximum time for integration
                              dt: float = 0.1) -> list:         # time step for evaluation
    g, k, m, L0, epsilon = params  # unpack parameters
    ext_initial_conditions = np.vstack((initial_conditions, perturbations))


    lyapunov_exponents: list = []  # initialize Lyapunov exponents list

    for y0 in ext_initial_conditions.T: # loop over initial conditions and perturbations
        sol = solve_ivp(fun=variational_equations,  # numerical integration
                        t_span=[0, t_max],
                        y0=y0,
                        method='RK45',
                        t_eval=np.arange(0, t_max, dt),
                        rtol=1e-12,
                        atol=1e-12)

        Y = sol.y[0:4, ]
        E = energy(Y, g, k, m, L0, epsilon)
        save_chart(sol)

        delta_T = sol.y[4:, -1]  #   final perturbation vector
        delta_0 = sol.y[4:, +1]  # initial perturbation vector
        exponent = np.log(np.linalg.norm(delta_T, axis=0) / np.linalg.norm(delta_0, axis=0)) / t_max  # Lyapunov exponent calculation
        lyapunov_exponents.append(exponent)  # store exponent

    return lyapunov_exponents  # return list of exponents


def energy(Y, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = Y.reshape(4, -1)
    T = 0.5 * m * (dr ** 2 + r ** 2 * dtheta ** 2)
    U = -m * g * r * np.cos(theta) + 0.5 * k * (r - L0) ** 2 + epsilon / r
    E = T + U
    return E


def save_chart(sol):
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    labels = ["Case 1"]
    axes[0].plot(sol.t, sol.y[1])
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Radial Distance r (m)")
    axes[0].grid()
    plt.suptitle("Elastic Pendulum Motion for Different Initial Conditions (Vectorized)")
    plt.savefig("elastic_pendulum_vectorized.png", dpi=300, bbox_inches='tight')
    plt.close()
    return None


if __name__ == '__main__':
    g = 9.81  # gravitational acceleration
    k = 40.0  # spring constant
    m = 1.0  # mass
    L0 = 1.0  # natural length of the spring
    epsilon = 0.0  # additional potential parameter
    params = (g, k, m, L0, epsilon)  # parameters tuple

    initial_conditions: ndarray = np.array(  # initial state conditions
        [[1., 1.],
         [0.52359878, 1.04719755],
         [0., 0.],
         [0., 0.]])

    perturbations: ndarray = np.array([  # initial perturbation conditions
        [1, 1],
        [0, 0],
        [0, 0],
        [0, 0]
    ])

    lyapunov_exps: list = compute_lyapunov_exponent(initial_conditions, perturbations, params)  # compute exponents

    for idx, exp in enumerate(lyapunov_exps):  # print results
        print(f'Largest Lyapunov exponent for initial condition {idx + 1}: {exp:.6f}')
