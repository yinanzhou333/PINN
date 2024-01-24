import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt

class Pendulum:
    def __init__(self, length=1.0, mass=1.0, gravity=9.81):
        self.length = length
        self.mass = mass
        self.gravity = gravity
        self.torque_profile = lambda t: 0.0  # Default torque profile (constant torque)

    def set_torque_profile(self, profile):
        self.torque_profile = profile

    def pendulum_equations(self, t, y):
        theta, omega = y
        torque = self.torque_profile(t)
        dydt = [omega, -(self.gravity / self.length) * np.sin(theta) + torque/(self.mass*self.length*self.length)]
        return dydt

    def simulate(self, initial_state, simulation_time):
        solution = solve_ivp(
            fun=self.pendulum_equations,
            t_span=simulation_time,
            y0=initial_state,
            max_step=0.001,
        )
        return solution
    def mytest(self, initial_state, t_span, step=0.0001):
        T=np.arange(t_span[0],t_span[1],step)
        theta=initial_state[0]
        omega=initial_state[1]
        self.theta_array=np.empty(T.shape, dtype=float)
        self.omega_array=np.empty(T.shape, dtype=float)
        self.time_array=T
        for i, t in enumerate(T):
            self.theta_array[i]=theta
            self.omega_array[i]=omega
            omega=omega+step*(self.torque_profile(t)/(self.mass*self.length)-self.gravity*np.sin(theta))/self.length
            theta=theta+omega*step
            
            
    
    def plot_simulation(self, solution):
        plt.plot(solution.t, solution.y[0], label='Theta')
        plt.plot(self.time_array, self.theta_array, linestyle="--", label='Theta_validation')
        #plt.plot(solution.t, solution.y[1], label='Omega')
        #plt.plot(solution.t, np.mod(solution.y[0] + np.pi, 2 * np.pi) - np.pi, label='Theta')
        plt.plot(solution.t, [self.torque_profile(t) for t in solution.t], label='Torque')
        plt.title('Pendulum Motion with Time-Dependent Torque')
        plt.xlabel('Time (s)')
        plt.legend()
        plt.grid(True)
        plt.show()

# Example usage
if __name__ == "__main__":
    pendulum = Pendulum(length=1.0, mass=0.1, gravity=9.81)

    # Simulation parameters
    initial_state = [0, 0]  # Initial state [theta, omega]
    simulation_time = (0, 10)  # Simulation time (start, end)
    
    # Define a time-dependent torque profile (e.g., torque increases linearly with time)
    def linear_torque(t):
        return 1.2*t
    def square_wave_torque(t, frequency=0.2, amplitude=1):
        return amplitude * np.sign(np.sin(2 * np.pi * frequency * t))
    
    pendulum.set_torque_profile(square_wave_torque)

    # Simulate and plot
    solution = pendulum.simulate(initial_state, simulation_time)
    pendulum.mytest(initial_state, simulation_time, step=0.0001)
    pendulum.plot_simulation(solution)

