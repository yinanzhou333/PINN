"""Backend supported: tensorflow.compat.v1, tensorflow, pytorch, jax, paddle"""
import deepxde as dde
import numpy as np
import sys
import os
import matplotlib.pyplot as plt

def func(x):
    """
    x: array_like, N x D_in
    y: array_like, N x D_out
    """
    return x * np.sin(5 * x)

geom = dde.geometry.Interval(-1, 1)
num_train = 10
num_test = 100
data = dde.data.Function(geom, func, num_train, num_test)

activation = "tanh"
initializer = "Glorot uniform"
net = dde.nn.FNN([1] + [20] * 3 + [1], activation, initializer)

model = dde.Model(data, net)
model.compile("adam", lr=0.001, metrics=["l2 relative error"])
 
def train_model():
    losshistory, train_state = model.train(iterations=10000)
    #dde.saveplot(losshistory, train_state, issave=True, isplot=True)
    model.save('model1',protocol='backend', verbose=0)
def use_model(model_filename):
    model.restore(model_filename) #model_filename = 'model1-10000.ckpt'

def plot_result():
    x = np.linspace(-1, 1, 201)
    x = x.reshape(-1, 1)
    y_pred = model.predict(x)
    y_true = func(x)

    plt.figure(figsize=(8, 6))
    plt.plot(x, y_true, label='True Function', linestyle='-')
    plt.plot(x, y_pred, label='Predicted Values', linestyle='--')
    plt.title('True Function vs Predicted Values')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.legend()
    plt.show()

if __name__ == "__main__":
    if len(sys.argv) == 1:
        # No command line argument, run train
        train_model()
    elif len(sys.argv) == 2:
        # One command line argument, assume it's the model filename to restore
        model_filename = sys.argv[1]
        use_model(model_filename)
    else:
        print("Usage: python script.py [model_filename]")
        sys.exit(1)

    # Regardless of 'train' or 'use', plot the results
    plot_result()

