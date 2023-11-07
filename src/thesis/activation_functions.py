import numpy as np                    
import matplotlib.pyplot as plt

def linear(x, a=2, c=0.5):
    f = a*x + c
    df = []
    for i in range(len(x)):
        df.append(a)
    return f, df

def sigmoid(x):
    f = 1 / (1 + np.exp(-x))
    df = f * (1-f)  
    return f, df

def tanh(x):
    f = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))
    df = 1 - (f**2)
    return f, df

def relu(x):
    f, df = [], []
    for x_t in x:
        f.append(x_t) if x_t >= 0 else f.append(0)
        df.append(1) if x_t >= 0 else df.append(0)
    return f, df

def activation_functions(include_derivatives = False):
    x = np.arange(-6,6,0.01)
    fig, axs = plt.subplots(2, 2, figsize = (12,6))
    axs[0,0].plot(x, linear(x)[0], color = "#307EC7", linewidth = 3, label = "f(x) (Linear; a=2, b=0.5)")
    axs[0,0].set_title("Linear\n", size = 16)
    axs[0,1].plot(x, sigmoid(x)[0], color = "#307EC7", linewidth = 3, label = "f(x) (Sigmoid)")
    axs[0,1].set_title("Sigmoid\n", size = 16)
    axs[1,0].plot(x, tanh(x)[0], color = "#307EC7", linewidth = 3, label = "f(x) (Tanh)")
    axs[1,0].set_title("\nTanh\n", size = 16)
    axs[1,1].plot(x, relu(x)[0], color = "#307EC7", linewidth = 3, label = "f(x) (ReLU)")
    axs[1,1].set_title("\nReLU\n", size = 16)
    if include_derivatives:
        axs[0,0].plot(x, linear(x)[1], color = "#9621E2", linewidth = 3, label = "f'(x) (Linear; a=2, b=0.5)")
        axs[0,1].plot(x, sigmoid(x)[1], color = "#9621E2", linewidth = 3, label = "f'(x) (Sigmoid)")
        axs[1,0].plot(x, tanh(x)[1], color = "#9621E2", linewidth = 3, label = "f'(x) (Tanh)")
        axs[1,1].plot(x, relu(x)[1], color = "#9621E2", linewidth = 3, label = "f'(x) (ReLU)")
    for i in range(2):
        for j in range(2):
            axs[i,j].spines['left'].set_position('center')
            axs[i,j].spines['right'].set_color('none')
            axs[i,j].spines['top'].set_color('none')
            axs[i,j].xaxis.set_ticks_position('bottom')
            axs[i,j].yaxis.set_ticks_position('left')
            axs[i,j].legend(loc = "upper left", frameon = False)
    plt.tight_layout()
    plt.savefig("../figures_tables/Activation Functions & Derivatives.pdf") if include_derivatives else plt.savefig("../figures_tables/Activation Functions.pdf")

if __name__ == "__main__":
    activation_functions(include_derivatives = False)
    activation_functions(include_derivatives = True)