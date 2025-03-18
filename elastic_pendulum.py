import numpy as np; from numpy import ndarray; from scipy.integrate import solve_ivp

def compute_lyapunov_exponent(initial_conditions: np.ndarray,  # initial conditions [r, theta, dr, dtheta]
                              perturbations: np.ndarray,       # perturbations [delta_r, delta_theta, delta_dr, delta_dtheta]
                              params: tuple,                   # parameters tuple (g, k, m, L0, epsilon)
                              t_max: float = 100,              # maximum time for integration
                              dt: float = 0.1) -> np.ndarray:  # time step for evaluation
    g, k, m, L0, epsilon = params  # unpack parameters

    def elastic_pendulum(t, y):
        r, theta, dr, dtheta = y  # state variables
        d2r = r * dtheta**2 + g * np.cos(theta) - (k/m)*(r - L0) + epsilon/r**2  # radial acceleration
        d2theta = -2 * dr * dtheta / r - (g / r) * np.sin(theta)  # angular acceleration
        dY = np.array([dr, dtheta, d2r, d2theta])
        return dY

    lyapunov_exponents = []  # initialize Lyapunov exponents list

    for ic, pert in zip(initial_conditions, perturbations):  # loop over initial conditions and perturbations
        y0 = np.concatenate((ic, pert))  # concatenate state and perturbation

        def variational_equations(t, z):
            y = z[:4]  # original state variables
            delta = z[4:].reshape((4,))  # perturbation variables
            r, theta, dr, dtheta = y  # unpack state variables

            dfdy = np.array([  # Jacobian matrix
                [0, 0, 1, 0],
                [0, 0, 0, 1],
                [dtheta**2 - k/m - 2*epsilon/r**3, -g*np.sin(theta), 0, 2*r*dtheta],
                [(2*dr*dtheta - g*np.sin(theta))/r**2, -(g/r)*np.cos(theta), -2*dtheta/r, -2*dr/r]
            ])

            dydt = elastic_pendulum(t, y)  # derivatives of state variables
            ddelta_dt = dfdy @ delta  # derivatives of perturbation variables

            return np.concatenate((dydt, ddelta_dt))  # combined derivatives

        sol = solve_ivp(fun=variational_equations,  # numerical integration
                        t_span=[0, t_max],
                        y0=y0, method='RK45',
                        t_eval=np.arange(0, t_max, dt),
                        rtol=1e-12,
                        atol=1e-12)

        delta_final = sol.y[4:, -1]  # final perturbation vector
        exponent = np.log(np.linalg.norm(delta_final) / np.linalg.norm(pert)) / t_max  # Lyapunov exponent calculation
        lyapunov_exponents.append(exponent)  # store exponent

    return np.array(lyapunov_exponents)  # return array of exponents


if __name__ == '__main__':
    g = 9.81          # gravitational acceleration
    k = 40.0          # spring constant
    m = 1.0           # mass
    L0 = 1.0          # natural length of the spring
    epsilon = 0.0     # additional potential parameter
    params = (g, k, m, L0, epsilon)  # parameters tuple

    initial_conditions: ndarray = np.array([  # initial state conditions
        [1.0, np.pi/6, 0.0, 0.0],
        [1.2, np.pi/4, 0.0, 0.0]
    ])

    perturbations: ndarray = np.array([  # initial perturbation conditions
        [0.0, 1e-8, 0.0, 0.0],
        [0.0, 1e-8, 0.0, 0.0]
    ])

    lyapunov_exps: ndarray = compute_lyapunov_exponent(initial_conditions, perturbations, params)  # compute exponents

    for idx, exp in enumerate(lyapunov_exps):  # print results
        print(f'Largest Lyapunov exponent for initial condition {idx+1}: {exp:.6f}')
