import matplotlib.pyplot as plt
import numpy as np
from matplotlib import style
import random

style.use('fivethirtyeight')
# style.use('ggplot')

# Xs = np.array([1, 2, 3, 4, 5, 6], dtype=np.float64)
# # Ys = np.array([49, 64, 81, 100, 121, 144], dtype=np.float64)
# Ys = np.array([5, 4, 6, 5, 6, 7], dtype=np.float64)


def create_dataset(hm, variance, step=2, correlation=False):
    val = 1
    Ys = []
    for i in range(hm):
        y = val + random.randrange(-variance, variance)
        Ys.append(y)
        if correlation and correlation == 'pos':
            val += step
        elif correlation and correlation == 'neg':
            val -= step

    Xs = [i for i in range(len(Ys))]
    return np.array(Xs, dtype=np.float64), np.array(Ys, dtype=np.float64)


def get_slope(x, y):
    m = ((((np.sum(x) / len(x)) * (np.sum(y) / len(y))) - (np.sum(x * y) / len(x * y))) /
         (((np.sum(x) / len(x)) ** 2) - (np.sum(x ** 2) / len(x ** 2))))
    return m


def coefficient_of_determination(orig_y, line_y):
    mean_line = [(np.sum(orig_y) / len(orig_y)) for i in orig_y]
    sqaured_error_pred_line = sum((line_y - orig_y) ** 2)
    sqaured_error_mean_line = sum((mean_line - orig_y) ** 2)
    return (1 - (sqaured_error_pred_line / sqaured_error_mean_line))


Xs, Ys = create_dataset(40, 10, step=2, correlation='pos')

m = get_slope(Xs, Ys)
print(f"Slope: {m}")
b = (np.sum(Ys) / len(Ys)) - m * (np.sum(Xs) / len(Xs))
print(f"Coefficient: {b}")


def predict(x):
    return ((m*x) + b)


to_prdict = 8
print(f"Prediction of {to_prdict}: {predict(to_prdict)}")

reg_line = [(m * x) + b for x in Xs]

r_squared = coefficient_of_determination(Ys, reg_line)

print(f"R Squared Error: {r_squared}")

plt.title('Custom Linear Regression Model')
mean_line = [(np.sum(Ys) / len(Ys)) for i in Ys]
plt.plot(Xs, mean_line, color='b', label='Mean Line')
plt.plot(Xs, reg_line, color='r', label='Regression Line')
plt.scatter(to_prdict, predict(to_prdict), color='g', label=f"Prediction of {to_prdict}", s=100)
plt.scatter(Xs, Ys, label='Real Data points')
plt.legend(loc=4)
plt.show()
