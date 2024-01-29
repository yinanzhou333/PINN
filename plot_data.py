import json
import os
import numpy as np
from pendulum_simulation import Pendulum

def cosine_of_last_theta(entry):
    return np.cos(entry["theta"][-1])
if __name__ == "__main__":
    pendulum = Pendulum(length=1.0, mass=1, gravity=9.81)
    initial_state = [0, 0]  # Initial state [theta, omega]
    simulation_time = (0, 10)  # Simulation time (start, end)
    with open("mydata.json", 'r') as file:
        lines = file.readlines()
        # Process each line as a separate JSON object
        json_data = [json.loads(line) for line in lines]
        #json_data = json.load(file)
        sorted_json_data = sorted(json_data, key=cosine_of_last_theta)
        for data in [sorted_json_data[0],sorted_json_data[1],sorted_json_data[2],sorted_json_data[-1]]:
            print(f'cos(theta) at the last time point is {np.cos(data["theta"][-1])} ')
            torque_data = np.column_stack((data['time'], data['torque']))
            pendulum.run_simulation(torque_data, initial_state, simulation_time)
            pendulum.mytest(initial_state, simulation_time, step=0.001)
            pendulum.plot_simulation(data)
