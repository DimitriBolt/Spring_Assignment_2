import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

# Parameters from provided script
g = 9.81       # gravitational acceleration (m/s^2)
k = 1.0        # spring constant (N/m)
m = 1.0        # mass (kg)
L0 = 10.0      # natural length of the pendulum (m)
epsilon = -1   # additional potential parameter
params = (g, k, m, L0, epsilon)

# Equations of motion from provided script
def elastic_pendulum(t, Y, g, k, m, L0, epsilon):
    r, theta, dr, dtheta = Y
    d2r = r * dtheta**2 + g * np.cos(theta) - (k/m) * (r - L0) + epsilon / r**2
    d2theta = -2 * dr * dtheta / r - (g/r) * np.sin(theta)
    return [dr, dtheta, d2r, d2theta]

# Event function for Poincaré section (theta=0, dtheta>0)
def poincare_event(t, Y, *args):
    return Y[1]  # theta crossing zero

poincare_event.terminal = False
poincare_event.direction = 1  # only crossings from negative to positive (dtheta > 0)

# Initial condition (only the first set)
initial_condition = [1.0, np.pi/6, 0.0, 0.0]

# Integration settings
tmax = 130
t_span = (0, tmax)
t_eval = np.linspace(0, tmax, 10000)

# Integrate equations with event detection
sol = solve_ivp(fun=elastic_pendulum,
                t_span=t_span,
                y0=initial_condition,
                args=params,
                events=poincare_event,
                method='RK45',
                atol=1e-12,
                rtol=1e-12)

# Extract event points for Poincaré section
poincare_r = sol.y_events[0][:, 0]
poincare_dr = sol.y_events[0][:, 2]

# Plotting the Poincaré section
plt.figure(figsize=(8, 6))
plt.scatter(poincare_r, poincare_dr, s=10, c='blue')
plt.title('Poincaré Section of Elastic Pendulum (θ=0, dθ>0)')
plt.xlabel('Radial Distance r (m)')
plt.ylabel('Radial Velocity dr/dt (m/s)')
plt.grid()
plt.tight_layout()
plt.show()
