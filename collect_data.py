from pendulum_simulation import Pendulum

if __name__ == "__main__":
    pendulum = Pendulum(length=1.0, mass=1, gravity=9.81)

    # Simulation parameters
    initial_state = [0, 0]  # Initial state [theta, omega]
    simulation_time = (0, 10)  # Simulation time (start, end)
    for i in range(5, 20):
    	print(f'number of sample torque points: {i}')
    	for j in range(1000):
    		torque_data = pendulum.generate_random_torque_data(num_points=i)
    		data = pendulum.run_simulation(torque_data, initial_state, simulation_time, save_file='mydata.json')

