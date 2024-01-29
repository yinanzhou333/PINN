import json
import os
import numpy as np
from pendulum_simulation import Pendulum

if __name__ == "__main__":
    pendulum = Pendulum(length=1.0, mass=1, gravity=9.81)
    initial_state = [0, 0]  # Initial state [theta, omega]
    simulation_time = (0, 10)  # Simulation time (start, end)
    with open("mydata.json", 'r') as file:
        existing_data = json.load(file)
        for data in existing_data:
            torque_data = np.column_stack((data['time'], data['torque']))
            pendulum.run_simulation(torque_data, initial_state, simulation_time)
            pendulum.mytest(initial_state, simulation_time, step=0.001)
            pendulum.plot_simulation(data)
