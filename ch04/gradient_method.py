import sys, os
sys.path.append(os.pardir)
import numpy as np
from common.gradient import _numerical_gradient_1d

def gradient_descent(f, init_x, lr, step_num):
    x = init_x

    for i in range(step_num):
        grad = _numerical_gradient_1d(f, x)
        x = x - lr * grad

    return x

def function_2(x):
    return np.sum(x**2)

if __name__ == "__main__":
    init_x = np.array([-3.0, 4.0])
    x = gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
    print(x)
