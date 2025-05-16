import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

def plot_cos_function(steps=100):
    x = np.arange(0, steps + 1, 1) / steps  # Creates values from 0 to 1 in steps of 0.01
    y = np.cos(0.5 * np.pi * (x))
    # y = 1-np.cos(0.5 * np.pi * (x))
    # y = (2 / np.pi) * np.arccos(1-x)
    # x, y = x[:-1], y[1:]  /y[:-1]
    # y = np.log(y)
    # y = np.exp(np.cumsum(y))
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'cos(0.5π((x/{steps})))')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of the Given Function')
    plt.legend()
    plt.grid()
    plt.show()

def plot_power_function(power):
    x = np.linspace(0, 1, 500)  # Creates values from 0 to 1 in steps of 0.01
    y = -x ** power + 1
    plt.figure(figsize=(8, 5))
    plt.plot(x, y, label=f'Power: {power}')
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Plot of the Given Function')
    plt.legend()
    plt.grid()
    plt.show()

# Example usage
plot_power_function(0.5)


def plot_overlapping_normals(x_value):
    mean1, std1 = 0, 1
    mean2, std2 = .5, 1
    x = np.linspace(0, 2, 500)

    y1 = norm.pdf(x, mean1, std1)
    y2 = norm.pdf(x, mean2, std2)

    plt.figure(figsize=(8, 5))
    plt.plot(x, y1, 'g', label="$p_{data}$")
    plt.plot(x, y2, 'r', label="$p_{pred}$")

    plt.fill_between(x, y1, color='green', alpha=0.5)
    plt.fill_between(x, y2, color='red', alpha=0.5)
    y1_val = norm.pdf(x_value, mean1, std1)
    y2_val = norm.pdf(x_value, mean2, std2)
    plt.vlines(x_value, 0, y1_val, colors='darkgreen', linestyles='dashed')
    plt.vlines(x_value+0.005, 0, y2_val, colors='darkred', linestyles='dashed')
    plt.scatter([x_value], [y1_val], color='g', s=50)
    plt.scatter([x_value], [y2_val], color='r', s=50)

    text1 = '$α_t$ $p_{data}$'
    text2 = '$α_t$ $p_{pred}$'
    plt.text(x_value - 0.1, y1_val / 2, text1, color='darkgreen', fontsize=14, rotation=90, verticalalignment='center')
    plt.text(x_value + 0.05, y2_val / 2, text2, color='darkred', fontsize=14, rotation=90, verticalalignment='center')

    plt.xlabel("X")
    plt.ylabel("Probability Density")
    plt.title("")
    plt.legend()
    plt.show()

plot_overlapping_normals(1)


def plot_overlapping_barplot(bin):
    mean1, std1 = 0, 1
    mean2, std2 = 0.5, 1
    bins = np.linspace(-1, 2, 8)  # Create bins for histogram

    # Generate sample data
    data1 = np.random.normal(mean1, std1, 10000)
    data2 = np.random.normal(mean2, std2, 10000)*0.5

    plt.figure(figsize=(8, 5))
    y1_val, _, _ = plt.hist(data1, bins=bins, density=True, alpha=0.5, color='green', label="$p_{data}$", edgecolor='black')
    y2_val, _, _ = plt.hist(data2, bins=bins, density=True, alpha=0.5, color='red', label="$p_{pred}$", edgecolor='black')

    #y1_val = norm.pdf(x_value, mean1, std1)
    #y2_val = norm.pdf(x_value, mean2, std2)
    y1_val = y1_val[bin]
    y2_val = y2_val[bin]
    bin_width = bins[1] - bins[0]
    x_value = (bins[bin]+0.5*bin_width)
    plt.vlines(x_value - 0.1*bin_width, 0, y1_val, colors='darkgreen', linestyles='dashed')
    plt.vlines(x_value + 0.1*bin_width, 0, y2_val, colors='darkred', linestyles='dashed')
    plt.scatter([x_value]- 0.1*bin_width, [y1_val], color='g', s=50)
    plt.scatter([x_value]+ 0.1*bin_width, [y2_val], color='r', s=50)

    text1 = '$α_t$ $p_{data}$'
    text2 = '$α_t$ $p_{pred}$'
    plt.text(x_value - 0.4*bin_width, y1_val / 2, text1, color='darkgreen', fontsize=16, rotation=90, verticalalignment='center')
    plt.text(x_value + 0.2*bin_width, y2_val / 2, text2, color='darkred', fontsize=16, rotation=90, verticalalignment='center')

    plt.xlabel("$\mathcal{Z}$", fontsize=20)
    plt.ylabel("PMF", fontsize=16)
    #plt.title("")
    #plt.legend()
    plt.xticks([])
    plt.yticks([])
    # plt.gca().spines['top'].set_visible(False)
    # plt.gca().spines['right'].set_visible(False)
    # plt.gca().spines['left'].set_visible(False)
    # plt.gca().spines['bottom'].set_visible(False)
    plt.show()


# Example usage
plot_overlapping_barplot(bin=2)