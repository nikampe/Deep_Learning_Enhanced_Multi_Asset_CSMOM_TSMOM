import numpy as np                    
import matplotlib.pyplot as plt

def training(x):
    return np.exp(-x)

def testing(x):
    x_left, x_right = x[0:int(len(x)*0.7)], x[int(len(x)*0.7):len(x)]
    y_left, y_right = np.exp(-0.75*x_left), np.exp(0.75*x_left)
    return 0.08*(x-3)**2 + 0.15

def train_test_plot():
    x = np.arange(1,4,0.1)
    y_training = training(x)
    y_testing = testing(x)
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xlabel("Model Complexity", size = 16)
    ax.set_ylabel("Model Error", size = 16)
    ax.plot(x, y_testing, color="#9621E2", linewidth=3, label="Testing")
    ax.plot(x, y_training, color="#307EC7", linewidth=3, label="Training")
    ax.axvline(x[min(range(len(y_testing)),key=y_testing.__getitem__)], ymax=0.8, color="red", linestyle="--", linewidth="2")
    ax.text(x=x[min(range(len(y_testing)),key=y_testing.__getitem__)], y=0.44, s="Optimal\n Complexity", ha="center", va="center", color="red", size=14)
    ax.text(x=x[int(len(x)*0.6)], y=0.33, s="Underfitting", ha="right", va="center", color="black", size=14)
    ax.arrow(x=x[int(len(x)*0.6)], y=0.3, dx=-0.55, dy=0, head_width=0.01, length_includes_head=True, color="black")
    ax.text(x=x[int(len(x)*0.75)], y=0.33, s="Overfitting", ha="left", va="center", color="black", size=14)
    ax.arrow(x=x[int(len(x)*0.75)], y=0.3, dx=0.5, dy=0, head_width=0.01, length_includes_head=True, color="black")
    ax.legend(loc="lower left", frameon=False, prop={'size': 14})
    plt.savefig("../Figures/Overfitting & Underfitting.pdf")

if __name__ == "__main__":
    train_test_plot()