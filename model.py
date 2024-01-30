import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class TorquePredictionModel:
    def __init__(self, input_shape, output_shape):
        self.model = self._build_model(input_shape, output_shape)

    def _build_model(self, input_shape, output_shape):
        model = models.Sequential()
        model.add(layers.Dense(64, activation='relu', input_shape=(input_shape,)))
        model.add(layers.Dense(32, activation='relu'))
        model.add(layers.Dense(output_shape, activation='linear'))  # Output layer for regression
        model.compile(optimizer='adam', loss='mean_squared_error')
        return model

    def _custom_loss_function(self, y_true, y_pred):
        cos_theta_true = y_true[:, -1]
        cos_theta_pred = y_pred[:, -1]
        loss = tf.keras.losses.mean_squared_error(cos_theta_true, cos_theta_pred)
        return loss

    def train(self, input_data, target_data, validation_data=None, epochs=50, batch_size=32):
        history = self.model.fit(input_data, target_data, epochs=epochs, batch_size=batch_size, validation_data=validation_data)
        return history

    def predict_torque(self, input_for_prediction):
        return self.model.predict(input_for_prediction)

def load_dataset(file_path):
    with open(file_path, 'r') as file:
        lines = file.readlines()

    data = [json.loads(line) for line in lines]
    return data

def prepare_data(data):
    flattened_times = [np.ravel(sample['time']) for sample in data]
    max_time_length = max(len(t) for t in flattened_times)
    padded_times = [np.pad(t, (0, max_time_length - len(t)), 'constant') for t in flattened_times]
    
    times = np.vstack(padded_times)
    
    flattened_thetas = [np.ravel(sample['theta']) for sample in data]
    max_theta_length = max(len(t) for t in flattened_thetas)
    padded_thetas = [np.pad(t, (0, max_theta_length - len(t)), 'constant') for t in flattened_thetas]
    thetas = np.vstack(padded_thetas)

    flattened_omegas = [np.ravel(sample['omega']) for sample in data]
    max_omega_length = max(len(t) for t in flattened_omegas)
    padded_omegas = [np.pad(t, (0, max_omega_length - len(t)), 'constant') for t in flattened_omegas]
    omegas = np.vstack(padded_omegas)

    flattened_torques = [np.ravel(sample['torque']) for sample in data]
    max_torque_length = max(len(t) for t in flattened_torques)
    padded_torques = [np.pad(t, (0, max_torque_length - len(t)), 'constant') for t in flattened_torques]
    torques = np.vstack(padded_torques)

    return times, thetas, omegas, torques

def main():
    # Load your dataset
    dataset_path = 'mydata.json'  # Replace with the actual path to your JSON file
    dataset = load_dataset(dataset_path)

    # Prepare the data
    times, thetas, omegas, torques = prepare_data(dataset)

    # Randomly split the data into training and validation sets
    test_size = 0.2  # 20% for validation
    random_seed = 42  # Set a seed for reproducibility
    train_times, val_times, train_thetas, val_thetas, train_omegas, val_omegas, train_torques, val_torques = \
        train_test_split(times, thetas, omegas, torques, test_size=test_size, random_state=random_seed)

    # Create and train the neural network model
    input_shape = len(train_times[0]) + len(train_thetas[0])  # Time array + Theta array
    output_shape = len(train_torques[0])

    torque_model = TorquePredictionModel(input_shape, output_shape)
    input_data_train = np.column_stack((train_times, train_thetas))
    input_data_val = np.column_stack((val_times, val_thetas))
    target_data_train = train_torques
    validation_data = (input_data_val, val_torques)

    history = torque_model.train(input_data_train, target_data_train, validation_data)

    # Make predictions on the testing set
    
    #input_data_test = np.column_stack((val_times[0], val_thetas[0]))  # Use the first sample from the validation set
    predicted_torque = torque_model.predict_torque(input_data_val[0:102])

    # Plot the results for the first sample in the validation set
    plt.figure()
    plt.plot(val_times[100], val_torques[100], label='True Torque', marker='o')
    plt.plot(val_times[100], predicted_torque[100], label='Predicted Torque', marker='x')
    print(val_times[100])
    print(predicted_torque[100])
    plt.xlabel('Time')
    plt.ylabel('Torque Value')
    plt.title('Validation Sample 1')
    plt.legend()
    plt.show()

    # Print loss for each epoch
    print("Training loss:", history.history['loss'])
    print("Validation loss:", history.history['val_loss'])

if __name__ == "__main__":
    main()



