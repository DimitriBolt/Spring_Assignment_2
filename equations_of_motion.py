import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

def elastic_pendulum(t, Y, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = Y.reshape(4, -1)
    d2r = r * dtheta**2 + g * np.cos(theta) - (k/m) * (r - L0) + epsilon / r**2
    d2theta = -2 * dr * dtheta / r - (g / r) * np.sin(theta)
    dY = np.array([dr, dtheta, d2r, d2theta])
    return dY

def energy(Y, n_ic, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = Y.reshape(4, n_ic, -1)
    T = 0.5 * m * (dr**2 + r**2 * dtheta**2)
    U = -m * g * r * np.cos(theta) + 0.5 * k * (r - L0)**2 + epsilon / r
    E = T + U
    return E

g, k, m, L0, epsilon = 9.81, 1.0, 1.0, 10.0, -1

initial_conditions = np.array([
    [1 +0   , np.pi/6, 0.0, 0.0],
    [1 +1e-4, np.pi/6, 0.0, 0.0],
    ]).transpose()

# Check if dimensions are consistent for vectorized solver
assert elastic_pendulum(1, initial_conditions, g, k, m, L0, epsilon).shape == initial_conditions.shape

tmax = 130
n_eval = 10000
t_span, t_eval = (0, tmax), np.linspace(0, tmax, n_eval)

sol = solve_ivp(
    fun = elastic_pendulum,
    t_span=t_span,
    y0=initial_conditions.flatten(),
    args=(g, k, m, L0, epsilon),
    t_eval=t_eval,
    method='RK45',
    atol=1e-12,
    rtol=1e-12,
    vectorized=True
)

fig, axes = plt.subplots(5, 1, figsize=(10, 10), sharex=True)
labels = ["Case 1", "Case 2", "Case 3", "Case 4"]

n_ic = initial_conditions.shape[1]
Y = sol.y.reshape(4, n_ic, -1)
dY = np.diff(Y, axis=1).squeeze() # difference on second dimension
norm_dY = np.linalg.norm(dY, axis=0)

lyapunov_exponent = 1/tmax * np.log(norm_dY[-1]/norm_dY[0])
print(f'Largest Lyapunov exponent for initial condition {0+1}: {lyapunov_exponent:.6f}')

E = energy(Y, n_ic, g, k, m, L0, epsilon)
for i in range(n_ic):
    assert np.max(abs(E[i] - E[i][0])) < 1e-8
# Charts:
# r
axes[0].plot(sol.t, Y[0][0]) # Radial Distance r (m)
axes[0].plot(sol.t, Y[0][1]) # Radial Distance r (m)
axes[0].set_xlabel("Time (s)")
axes[0].set_ylabel("Radial Distance r (m)")
# difference of r
axes[1].plot(sol.t, dY[0])
axes[1].set_xlabel("Time (s)")
axes[1].set_ylabel("delta Distance")

# theta
axes[2].plot(sol.t, Y[1][0]) # Angle (rad)
axes[2].plot(sol.t, Y[1][1]) # Angle (rad)
axes[2].set_xlabel("Time (s)")
axes[2].set_ylabel("Angle (rad)")
# difference of theta
axes[3].plot(sol.t, dY[1])
axes[3].set_xlabel("Time (s)")
axes[3].set_ylabel("delta Angle")

# ||dY||
axes[4].plot(sol.t, norm_dY)
axes[4].set_xlabel("Time (s)")
axes[4].set_ylabel("||dY||")
axes[4].set_yscale("log")


axes[0].grid()
axes[1].grid()
axes[2].grid()
axes[3].grid()
axes[4].grid()

plt.suptitle("Elastic Pendulum Motion for Different Initial Conditions (Vectorized)")
plt.savefig("elastic_pendulum_vectorized.png", dpi=300, bbox_inches='tight')
plt.close()
